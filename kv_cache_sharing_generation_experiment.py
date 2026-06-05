"""End-to-end translated KV-cache sharing generation experiment.

This script tests the closest current approximation to the original KV-sharing
goal:

1. sender/receiver features are collected on a long context X;
2. existing sparse-anchor block selection chooses sender evidence blocks;
3. existing multi-slot + head-wise translators map sender block memory to
   receiver-readable K/V slots;
4. translated slots are packed into a receiver DynamicCache;
5. receiver decodes from only the last prompt token plus translated cache;
6. generated continuations are compared with full receiver prefill+generate.

Important limitation: current translators are trained on projected K/V tensors
from the existing QKVExtractor path, while HuggingFace caches normally store
post-RoPE key states. This experiment intentionally keeps the existing
translator stack unchanged and measures whether that packed cache is useful as
a first real-cache integration test.
"""

import argparse
import csv
import os
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from block_memory_method_sweep import build_items, parse_method
from evidence_recall_selective_recompute import LLAMA_3_2_1B, collect_features, load_bundle, load_text_files
from generation_effect_experiment import train_translators_return_models


def infer_kv_heads(model):
    cfg = model.config
    q_heads = int(getattr(cfg, "num_attention_heads"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", q_heads))
    return q_heads, kv_heads


def q_heads_to_kv_heads(x, q_heads, kv_heads):
    if q_heads == kv_heads:
        return x
    if q_heads % kv_heads != 0:
        raise ValueError(f"Cannot fold {q_heads} query heads into {kv_heads} KV heads.")
    bsz, heads, seq_len, dim = x.shape
    group = q_heads // kv_heads
    return x.view(bsz, kv_heads, group, seq_len, dim).mean(dim=2).contiguous()


@torch.no_grad()
def translated_slots(item, tk, tv, prep_input, denorm_pred):
    pred_k = denorm_pred(tk(prep_input(item, "sender_k_slots")), "target_k_slots")
    pred_v = denorm_pred(tv(prep_input(item, "sender_v_slots")), "target_v_slots")
    valid = item["valid_slots"][None, None, :, None]
    return pred_k.masked_fill(~valid, 0.0), pred_v.masked_fill(~valid, 0.0)


def build_translated_cache(per_layer, sample_id, receiver_model):
    q_heads, kv_heads = infer_kv_heads(receiver_model)
    cache_data = []
    for layer in range(receiver_model.config.num_hidden_layers):
        data = per_layer[layer]
        k, v = translated_slots(
            data["items"][sample_id],
            data["tk"],
            data["tv"],
            data["prep_input"],
            data["denorm_pred"],
        )
        k = q_heads_to_kv_heads(k, q_heads, kv_heads).to(receiver_model.dtype)
        v = q_heads_to_kv_heads(v, q_heads, kv_heads).to(receiver_model.dtype)
        cache_data.append((k.contiguous(), v.contiguous()))
    return DynamicCache(ddp_cache_data=cache_data, config=receiver_model.config)


def encode_prompt(tokenizer, text, max_length, device):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def greedy_full_receiver(receiver, input_ids, attention_mask, max_new_tokens):
    out = receiver.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=receiver.tokenizer.eos_token_id,
    )
    return out[:, input_ids.shape[1] :]


@torch.no_grad()
def greedy_with_translated_cache(receiver, input_ids, translated_cache, max_new_tokens, original_context_len):
    device = input_ids.device
    generated = []
    cur = input_ids[:, -1:]
    cache = translated_cache
    cache_len = cache.get_seq_length()

    for step in range(max_new_tokens):
        attn_mask = torch.ones((1, cache_len + cur.shape[1]), dtype=torch.long, device=device)
        position_ids = torch.tensor([[original_context_len - 1 + step]], dtype=torch.long, device=device)
        out = receiver.model(
            input_ids=cur,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_id)
        cur = next_id
        cache = out.past_key_values
        cache_len = cache.get_seq_length()

    return torch.cat(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


def token_f1(a, b):
    a_list = [int(x) for x in a]
    b_list = [int(x) for x in b]
    if not a_list and not b_list:
        return 1.0
    if not a_list or not b_list:
        return 0.0
    used = set()
    overlap = 0
    for x in a_list:
        for i, y in enumerate(b_list):
            if i not in used and x == y:
                used.add(i)
                overlap += 1
                break
    precision = overlap / len(a_list)
    recall = overlap / len(b_list)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def seq_similarity(a, b):
    return SequenceMatcher(a=[int(x) for x in a], b=[int(x) for x in b]).ratio()


def continuation_ce(receiver, generated_ids, input_ids, attention_mask):
    full = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=1)
    logits = receiver.model(input_ids=full, attention_mask=full_mask, use_cache=False).logits
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    pred = logits[:, start:end, :].contiguous()
    labels = generated_ids.contiguous()
    return F.cross_entropy(pred.view(-1, pred.shape[-1]), labels.view(-1), reduction="mean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--method", default="multislot_headwise_norm")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/kv_cache_sharing_generation.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    receiver.model.eval()

    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)
    method = parse_method(args.method, args.slots_per_block)
    if args.layers == "all":
        layers = list(range(receiver.model.config.num_hidden_layers))
    else:
        layers = parse_int_list(args.layers)
    missing = set(range(receiver.model.config.num_hidden_layers)) - set(layers)
    if missing:
        raise ValueError("Real cache generation requires every receiver layer. Use --layers all.")

    per_layer = {}
    for layer in layers:
        items = build_items(
            features_s,
            features_r,
            layer,
            args.block_size,
            args.block_score_mode,
            args.anchor_tokens,
            args.budget_ratio,
            method,
            args.value_pool_mode,
            device,
        )
        tk, tv, prep_input, denorm_pred = train_translators_return_models(
            items,
            method,
            args.epochs,
            args.lr,
            args.hidden,
            args.kv_loss_weight,
            args.output_loss_weight,
        )
        per_layer[layer] = {
            "items": items,
            "tk": tk,
            "tv": tv,
            "prep_input": prep_input,
            "denorm_pred": denorm_pred,
        }
        print(f"trained layer {layer:02d}")

    rows = []
    for sample_id, text in enumerate(texts):
        input_ids, attention_mask = encode_prompt(receiver.tokenizer, text, args.max_length, device)
        context_len = int(attention_mask.sum().item())
        input_ids = input_ids[:, :context_len]
        attention_mask = attention_mask[:, :context_len]

        full_new = greedy_full_receiver(receiver, input_ids, attention_mask, args.max_new_tokens)
        cache = build_translated_cache(per_layer, sample_id, receiver.model)
        shared_new = greedy_with_translated_cache(receiver, input_ids, cache, args.max_new_tokens, context_len)

        full_ids = full_new[0].detach().cpu().tolist()
        shared_ids = shared_new[0].detach().cpu().tolist()
        full_text = receiver.tokenizer.decode(full_ids, skip_special_tokens=True)
        shared_text = receiver.tokenizer.decode(shared_ids, skip_special_tokens=True)

        row = {
            "sample_id": sample_id,
            "context_tokens": context_len,
            "cache_slots": cache.get_seq_length(),
            "max_new_tokens": args.max_new_tokens,
            "first_token_match": int(bool(full_ids and shared_ids and full_ids[0] == shared_ids[0])),
            "exact_match": int(full_ids == shared_ids),
            "prefix_match_tokens": next((i for i, (a, b) in enumerate(zip(full_ids, shared_ids)) if a != b), min(len(full_ids), len(shared_ids))),
            "token_f1": token_f1(shared_ids, full_ids),
            "sequence_similarity": seq_similarity(shared_ids, full_ids),
            "full_continuation_ce": float(continuation_ce(receiver, full_new, input_ids, attention_mask).cpu()),
            "shared_continuation_ce": float(continuation_ce(receiver, shared_new, input_ids, attention_mask).cpu()),
            "full_text": full_text.replace("\n", "\\n"),
            "shared_text": shared_text.replace("\n", "\\n"),
        }
        rows.append(row)
        print(
            f"sample={sample_id} first={row['first_token_match']} exact={row['exact_match']} "
            f"prefix={row['prefix_match_tokens']} f1={row['token_f1']:.3f} "
            f"sim={row['sequence_similarity']:.3f} cache_slots={row['cache_slots']}"
        )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
        "context_tokens",
        "cache_slots",
        "max_new_tokens",
        "first_token_match",
        "exact_match",
        "prefix_match_tokens",
        "token_f1",
        "sequence_similarity",
        "full_continuation_ce",
        "shared_continuation_ce",
        "full_text",
        "shared_text",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = max(len(rows), 1)
    print(f"Wrote CSV: {args.csv}")
    print(
        "mean "
        f"first={sum(r['first_token_match'] for r in rows) / n:.3f} "
        f"exact={sum(r['exact_match'] for r in rows) / n:.3f} "
        f"token_f1={sum(r['token_f1'] for r in rows) / n:.3f} "
        f"seq_sim={sum(r['sequence_similarity'] for r in rows) / n:.3f}"
    )


if __name__ == "__main__":
    main()
