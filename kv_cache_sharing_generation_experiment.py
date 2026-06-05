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

By default the translator is trained in real HuggingFace cache space: sender
and receiver K/V come from model(..., use_cache=True).past_key_values. For
training compatibility those KV-head tensors are repeated/aligned to receiver
query heads, and folded back to receiver KV heads before DynamicCache packing.
"""

import argparse
import csv
import os
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from attention_preserving_kv_translation_experiment import (
    LOCAL_SENDER_MODEL,
    align_heads,
    attention_output,
    attention_probs,
    parse_int_list,
)
from block_memory_method_sweep import (
    build_items,
    build_receiver_slot_targets,
    build_slots_for_block,
    parse_method,
    slot_attention,
)
from block_translated_memory_recovery import (
    receiver_block_sender_masks,
    score_blocks,
    select_blocks,
    selected_block_candidate_mask,
    sender_token_weights,
    sender_value_weights,
    sparse_normalize,
)
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
def collect_hf_caches(bundle, texts, max_length, device):
    caches = []
    for text in texts:
        enc = bundle.tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = bundle.model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        cache = out.past_key_values
        caches.append(
            {
                "k": [layer.keys.detach().float() for layer in cache.layers],
                "v": [layer.values.detach().float() for layer in cache.layers],
            }
        )
    return caches


def cache_heads_to_q_heads(x, target_heads):
    return align_heads(x, target_heads)


def build_cache_space_items(
    features_s,
    features_r,
    caches_s,
    caches_r,
    layer,
    block_size,
    block_score_mode,
    anchor_tokens,
    budget_ratio,
    config,
    value_pool_mode,
    device,
):
    items = []
    for fs, fr, cs, cr in zip(features_s, features_r, caches_s, caches_r):
        q_s = align_heads(fs["qkv"].q[layer].to(device), fr["qkv"].q[layer].shape[1])
        q_r = fr["qkv"].q[layer].to(device)
        target_heads = q_r.shape[1]

        # Real cache-space K/V. They are post-cache tensors with native KV
        # heads; repeat/align to receiver Q heads for existing slot attention
        # and translator training.
        k_s = cache_heads_to_q_heads(cs["k"][layer].to(device), target_heads)
        v_s = cache_heads_to_q_heads(cs["v"][layer].to(device), target_heads)
        k_r = cache_heads_to_q_heads(cr["k"][layer].to(device), target_heads)
        v_r = cache_heads_to_q_heads(cr["v"][layer].to(device), target_heads)

        s_mask = fs["mask"].to(device)
        r_mask = fr["mask"].to(device)
        seq_len = min(q_s.shape[-2], q_r.shape[-2], k_s.shape[-2], k_r.shape[-2], s_mask.shape[-1], r_mask.shape[-1])
        q_s, q_r = q_s[:, :, :seq_len, :], q_r[:, :, :seq_len, :]
        k_s, v_s = k_s[:, :, :seq_len, :], v_s[:, :, :seq_len, :]
        k_r, v_r = k_r[:, :, :seq_len, :], v_r[:, :, :seq_len, :]
        s_mask, r_mask = s_mask[:, :seq_len], r_mask[:, :seq_len]

        attn_s = attention_probs(q_s, k_s, s_mask)
        attn_r = attention_probs(q_r, k_r, r_mask)
        full_out = attention_output(attn_r, v_r)
        routing_weights = sender_token_weights(k_s, attn_s, s_mask, "kxreceived")
        value_weights = sender_value_weights(v_s, attn_s, s_mask, value_pool_mode)
        block_sender_masks, block_starts, block_ends, valid_from_sender = receiver_block_sender_masks(
            fs["offsets"].to(device),
            fr["offsets"].to(device),
            r_mask,
            block_size,
            device,
        )
        valid_recv = torch.tensor([end > start for start, end in zip(block_starts.tolist(), block_ends.tolist())], dtype=torch.bool, device=device)
        candidate_blocks = valid_from_sender & valid_recv
        block_scores = score_blocks(
            routing_weights,
            block_sender_masks,
            candidate_blocks,
            mode=block_score_mode,
            anchor_tokens=anchor_tokens,
        )
        valid_blocks = select_blocks(block_scores, candidate_blocks, keep_blocks=None, budget_ratio=budget_ratio)

        k_slots, v_slots, slot_scores, slot_starts, valid_slots = [], [], [], [], []
        for block_id in range(valid_blocks.shape[0]):
            bk, bv, bs = build_slots_for_block(
                k_s,
                v_s,
                routing_weights,
                value_weights,
                block_sender_masks[block_id],
                valid_blocks[block_id],
                config,
            )
            k_slots.extend(bk)
            v_slots.extend(bv)
            slot_scores.extend(bs)
            for _ in range(config.slots_per_block):
                slot_starts.append(block_starts[block_id])
                valid_slots.append(valid_blocks[block_id])

        sender_k_slots = torch.stack(k_slots, dim=2)
        sender_v_slots = torch.stack(v_slots, dim=2)
        slot_scores = torch.stack(slot_scores).to(device)
        slot_starts = torch.stack(slot_starts).to(device)
        valid_slots = torch.stack(valid_slots).to(device)
        target_k_slots, target_v_slots = build_receiver_slot_targets(k_r, v_r, valid_blocks, block_starts, block_ends, config.slots_per_block)
        candidate_mask = selected_block_candidate_mask(attn_r.shape, valid_blocks, block_starts, block_ends, r_mask, device)
        selective_oracle_out = attention_output(sparse_normalize(attn_r, candidate_mask), v_r)

        items.append(
            {
                "sender_k_slots": sender_k_slots,
                "sender_v_slots": sender_v_slots,
                "target_k_slots": target_k_slots,
                "target_v_slots": target_v_slots,
                "q_r": q_r,
                "r_mask": r_mask,
                "full_out": full_out,
                "selective_oracle_out": selective_oracle_out,
                "valid_slots": valid_slots,
                "slot_starts": slot_starts,
                "slot_scores": slot_scores,
                "selected_ratio": float(valid_blocks.float().mean().cpu()),
            }
        )
    return items


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
    parser.add_argument("--cache_space", choices=["hf_cache", "raw_extractor"], default="hf_cache")
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
    caches_s = caches_r = None
    if args.cache_space == "hf_cache":
        print("Collecting sender/receiver HF past_key_values...")
        caches_s = collect_hf_caches(sender, texts, args.max_length, device)
        caches_r = collect_hf_caches(receiver, texts, args.max_length, device)
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
        if args.cache_space == "hf_cache":
            items = build_cache_space_items(
                features_s,
                features_r,
                caches_s,
                caches_r,
                layer,
                args.block_size,
                args.block_score_mode,
                args.anchor_tokens,
                args.budget_ratio,
                method,
                args.value_pool_mode,
                device,
            )
        else:
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
            "cache_space": args.cache_space,
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
        "cache_space",
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
