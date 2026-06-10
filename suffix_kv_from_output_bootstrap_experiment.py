"""Decode with only suffix-layer KV produced after attention-output patching.

This tests whether translated attention outputs can bootstrap receiver-native
KV in later layers, while earlier layers keep no historical KV at decode time.
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
from generation_effect_experiment import train_translators_return_models, translated_attention_hidden


def token_f1(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    used = set()
    overlap = 0
    for x in a:
        for i, y in enumerate(b):
            if i not in used and x == y:
                used.add(i)
                overlap += 1
                break
    p = overlap / len(a)
    r = overlap / len(b)
    return 2 * p * r / max(p + r, 1e-8)


def seq_similarity(a, b):
    return SequenceMatcher(a=[int(x) for x in a], b=[int(x) for x in b]).ratio()


def continuation_ce(receiver, generated_ids, input_ids, attention_mask):
    full = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=1)
    logits = receiver.model(input_ids=full, attention_mask=full_mask, use_cache=False).logits
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    pred = logits[:, start:end, :].contiguous()
    return F.cross_entropy(pred.view(-1, pred.shape[-1]), generated_ids.reshape(-1), reduction="mean")


@torch.no_grad()
def greedy_full(receiver, input_ids, attention_mask, max_new_tokens):
    out = receiver.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=receiver.tokenizer.eos_token_id,
    )
    return out[:, input_ids.shape[1] :]


def max_cache_len(cache):
    return max(int(layer.keys.shape[-2]) for layer in cache.layers)


@torch.no_grad()
def greedy_from_mixed_cache(receiver, input_ids, cache, context_len, max_new_tokens):
    cur = input_ids[:, -1:]
    generated = []
    device = input_ids.device
    for step in range(max_new_tokens):
        attn_len = max_cache_len(cache) + 1
        attention_mask = torch.ones((1, attn_len), dtype=torch.long, device=device)
        position_ids = torch.tensor([[context_len - 1 + step]], dtype=torch.long, device=device)
        out = receiver.model(
            input_ids=cur,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur)
        cache = out.past_key_values
    return torch.cat(generated, dim=1)


def suffix_cache(cache, start_layer, receiver_model):
    data = []
    for idx, layer in enumerate(cache.layers):
        k = layer.keys.detach()
        v = layer.values.detach()
        if idx < start_layer:
            k = k[:, :, :0, :]
            v = v[:, :, :0, :]
        data.append((k.contiguous(), v.contiguous()))
    return DynamicCache(ddp_cache_data=data, config=receiver_model.config)


@torch.no_grad()
def patched_prefill(receiver, input_ids, attention_mask, per_layer, sample_id, alpha, method):
    handles = []
    replacements = {}
    for layer, data in per_layer.items():
        replacements[layer] = translated_attention_hidden(
            data["items"][sample_id],
            data["tk"],
            data["tv"],
            data["prep_input"],
            data["denorm_pred"],
            method,
            data["attn_module"],
        )

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                original = output[0] if isinstance(output, tuple) else output
                repl = replacements[layer_id].to(original.dtype)
                patched = alpha * repl + (1.0 - alpha) * original
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            return hook

        handles.append(data["attn_module"].register_forward_hook(make_hook(layer)))

    out = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    for handle in handles:
        handle.remove()
    return out


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
    parser.add_argument("--patch_layer_sets", default="8;12;8,12,15")
    parser.add_argument("--suffix_starts", default="9,13")
    parser.add_argument("--alphas", default="1.0,0.5")
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
    parser.add_argument("--csv", default="runs/suffix_kv_from_output_bootstrap.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)
    method = parse_method(args.method, args.slots_per_block)
    patch_sets = [parse_int_list(chunk) for chunk in args.patch_layer_sets.split(";") if chunk.strip()]
    suffix_starts = parse_int_list(args.suffix_starts)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    enc = receiver.tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    input_ids_all = enc["input_ids"].to(device)
    mask_all = enc["attention_mask"].to(device)
    rows = []

    for patch_layers in patch_sets:
        patch_key = ",".join(str(x) for x in patch_layers)
        per_layer = {}
        for layer in patch_layers:
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
                "attn_module": receiver.extractor.attn_layers[layer][1],
            }

        for sample_id in range(len(texts)):
            input_ids = input_ids_all[sample_id : sample_id + 1]
            attention_mask = mask_all[sample_id : sample_id + 1]
            context_len = int(attention_mask.sum().item())
            input_ids = input_ids[:, :context_len]
            attention_mask = attention_mask[:, :context_len]
            ref = greedy_full(receiver, input_ids, attention_mask, args.max_new_tokens)
            ref_ids = ref[0].detach().cpu().tolist()
            ref_text = receiver.tokenizer.decode(ref_ids, skip_special_tokens=True)
            ref_ce = float(continuation_ce(receiver, ref, input_ids, attention_mask).cpu())
            for alpha in alphas:
                prefill = patched_prefill(receiver, input_ids, attention_mask, per_layer, sample_id, alpha, method)
                for suffix_start in suffix_starts:
                    cache = suffix_cache(prefill.past_key_values, suffix_start, receiver.model)
                    pred = greedy_from_mixed_cache(receiver, input_ids, cache, context_len, args.max_new_tokens)
                    pred_ids = pred[0].detach().cpu().tolist()
                    pred_text = receiver.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    row = {
                        "sample_id": sample_id,
                        "patch_layers": patch_key,
                        "suffix_start": suffix_start,
                        "alpha": alpha,
                        "context_tokens": context_len,
                        "max_new_tokens": args.max_new_tokens,
                        "first_token_match": int(bool(ref_ids and pred_ids and ref_ids[0] == pred_ids[0])),
                        "exact_match": int(ref_ids == pred_ids),
                        "prefix_match_tokens": next((i for i, (a, b) in enumerate(zip(ref_ids, pred_ids)) if a != b), min(len(ref_ids), len(pred_ids))),
                        "token_f1": token_f1(pred_ids, ref_ids),
                        "sequence_similarity": seq_similarity(pred_ids, ref_ids),
                        "reference_continuation_ce": ref_ce,
                        "suffix_continuation_ce": float(continuation_ce(receiver, pred, input_ids, attention_mask).cpu()),
                        "reference_text": ref_text.replace("\n", "\\n"),
                        "suffix_text": pred_text.replace("\n", "\\n"),
                    }
                    rows.append(row)
                    print(
                        f"patch={patch_key} suffix={suffix_start} sample={sample_id} alpha={alpha:.2f} "
                        f"first={row['first_token_match']} exact={row['exact_match']} "
                        f"f1={row['token_f1']:.3f} sim={row['sequence_similarity']:.3f}"
                    )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
        "patch_layers",
        "suffix_start",
        "alpha",
        "context_tokens",
        "max_new_tokens",
        "first_token_match",
        "exact_match",
        "prefix_match_tokens",
        "token_f1",
        "sequence_similarity",
        "reference_continuation_ce",
        "suffix_continuation_ce",
        "reference_text",
        "suffix_text",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
