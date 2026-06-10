"""Diagnose whether failures come from cross-layer KV sharing or output-made KV.

The experiment compares four cache variants against normal receiver full-cache
decode:

* native_full: unmodified receiver prefix cache sanity check.
* output_full: full prefix cache produced after translated attention-output
  patching during receiver prefill.
* native_reconstructed: keep native anchor layers, reconstruct other layers.
* output_reconstructed: keep output-patched anchor layers, reconstruct others.
"""

import argparse
import csv
import os
from collections import defaultdict

import torch

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from block_memory_method_sweep import build_items, parse_method
from cross_layer_kv_reconstruction_experiment import (
    clone_cache_data,
    continuation_ce,
    greedy_from_cache,
    prefix_cache,
    reconstruct_cache,
    seq_similarity,
    token_f1,
    train_adapters,
)
from evidence_recall_selective_recompute import LLAMA_3_2_1B, collect_features, load_bundle, load_text_files
from generation_effect_experiment import train_translators_return_models, translated_attention_hidden
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def patched_prefix_cache(receiver, input_ids, attention_mask, per_layer, sample_id, alpha, method):
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
                repl = replacements[layer_id][:, : original.shape[1], :].to(original.dtype)
                patched = alpha * repl + (1.0 - alpha) * original
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            return hook

        handles.append(data["attn_module"].register_forward_hook(make_hook(layer)))

    out = receiver.model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True)
    for handle in handles:
        handle.remove()
    return out.past_key_values


def score_row(receiver, sample_id, variant, method, anchors, cache_data, ref_ids, ref_text, ref_ce, input_ids, attention_mask, context_len, max_new_tokens):
    pred = greedy_from_cache(receiver, input_ids, DynamicCache(ddp_cache_data=cache_data, config=receiver.model.config), max_new_tokens, context_len)
    pred_ids = pred[0].detach().cpu().tolist()
    pred_text = receiver.tokenizer.decode(pred_ids, skip_special_tokens=True)
    return {
        "sample_id": sample_id,
        "variant": variant,
        "method": method,
        "anchor_layers": ",".join(str(x) for x in anchors),
        "context_tokens": context_len,
        "max_new_tokens": max_new_tokens,
        "first_token_match": int(bool(ref_ids and pred_ids and ref_ids[0] == pred_ids[0])),
        "exact_match": int(ref_ids == pred_ids),
        "prefix_match_tokens": next((i for i, (a, b) in enumerate(zip(ref_ids, pred_ids)) if a != b), min(len(ref_ids), len(pred_ids))),
        "token_f1": token_f1(pred_ids, ref_ids),
        "sequence_similarity": seq_similarity(pred_ids, ref_ids),
        "reference_continuation_ce": ref_ce,
        "variant_continuation_ce": float(continuation_ce(receiver, pred, input_ids, attention_mask).cpu()),
        "reference_text": ref_text.replace("\n", "\\n"),
        "variant_text": pred_text.replace("\n", "\\n"),
    }


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
    parser.add_argument("--anchor_layers", default="8,12,15,20")
    parser.add_argument("--reconstruct_methods", default="copy_nearest,interp,adapter")
    parser.add_argument("--patch_layers", default="12,15")
    parser.add_argument("--alpha", type=float, default=0.5)
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
    parser.add_argument("--adapter_epochs", type=int, default=50)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--csv", default="runs/diagnose_cross_layer_vs_output_kv.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)
    method = parse_method(args.method, args.slots_per_block)
    patch_layers = parse_int_list(args.patch_layers)
    anchors = sorted({x for x in parse_int_list(args.anchor_layers) if 0 <= x < receiver.model.config.num_hidden_layers})
    reconstruct_methods = [x.strip() for x in args.reconstruct_methods.split(",") if x.strip()]

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

    prompts, native_rows, output_rows, refs = [], [], [], []
    for sample_id, text in enumerate(texts):
        enc = receiver.tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        context_len = int(attention_mask.sum().item())
        input_ids = input_ids[:, :context_len]
        attention_mask = attention_mask[:, :context_len]
        native_cache = prefix_cache(receiver, input_ids, attention_mask)
        output_cache = patched_prefix_cache(receiver, input_ids, attention_mask, per_layer, sample_id, args.alpha, method)
        native_data = clone_cache_data(native_cache)
        output_data = clone_cache_data(output_cache)
        ref = greedy_from_cache(receiver, input_ids, DynamicCache(ddp_cache_data=native_data, config=receiver.model.config), args.max_new_tokens, context_len)
        prompts.append((input_ids, attention_mask, context_len))
        native_rows.append(native_data)
        output_rows.append(output_data)
        refs.append(ref)

    native_adapters = train_adapters(native_rows, anchors, args.adapter_epochs, args.adapter_lr) if "adapter" in reconstruct_methods else {}
    output_adapters = train_adapters(output_rows, anchors, args.adapter_epochs, args.adapter_lr) if "adapter" in reconstruct_methods else {}

    rows = []
    for sample_id, (input_ids, attention_mask, context_len) in enumerate(prompts):
        ref_ids = refs[sample_id][0].detach().cpu().tolist()
        ref_text = receiver.tokenizer.decode(ref_ids, skip_special_tokens=True)
        ref_ce = float(continuation_ce(receiver, refs[sample_id], input_ids, attention_mask).cpu())
        rows.append(score_row(receiver, sample_id, "native_full", "full", anchors, native_rows[sample_id], ref_ids, ref_text, ref_ce, input_ids, attention_mask, context_len, args.max_new_tokens))
        rows.append(score_row(receiver, sample_id, "output_full", "full", anchors, output_rows[sample_id], ref_ids, ref_text, ref_ce, input_ids, attention_mask, context_len, args.max_new_tokens))
        for rec_method in reconstruct_methods:
            native_rec = reconstruct_cache(native_rows[sample_id], anchors, rec_method, native_adapters, receiver.model)
            output_rec = reconstruct_cache(output_rows[sample_id], anchors, rec_method, output_adapters, receiver.model)
            rows.append(score_row(receiver, sample_id, "native_reconstructed", rec_method, anchors, clone_cache_data(native_rec), ref_ids, ref_text, ref_ce, input_ids, attention_mask, context_len, args.max_new_tokens))
            rows.append(score_row(receiver, sample_id, "output_reconstructed", rec_method, anchors, clone_cache_data(output_rec), ref_ids, ref_text, ref_ce, input_ids, attention_mask, context_len, args.max_new_tokens))
            print(f"sample={sample_id} method={rec_method}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
        "variant",
        "method",
        "anchor_layers",
        "context_tokens",
        "max_new_tokens",
        "first_token_match",
        "exact_match",
        "prefix_match_tokens",
        "token_f1",
        "sequence_similarity",
        "reference_continuation_ce",
        "variant_continuation_ce",
        "reference_text",
        "variant_text",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["method"])].append(row)
    for key, vals in sorted(groups.items()):
        print(
            key,
            "first",
            sum(float(x["first_token_match"]) for x in vals) / len(vals),
            "exact",
            sum(float(x["exact_match"]) for x in vals) / len(vals),
            "f1",
            sum(float(x["token_f1"]) for x in vals) / len(vals),
        )
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
