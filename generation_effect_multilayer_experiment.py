"""Multi-layer generation-proxy evaluation for translated block memory.

Train one translated block-memory adapter per selected receiver layer, then
patch all selected attention layers in the same receiver forward pass.
"""

import argparse
import csv
import os

import torch

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from block_memory_method_sweep import build_items, parse_method
from evidence_recall_selective_recompute import LLAMA_3_2_1B, collect_features, load_bundle, load_text_files
from generation_effect_experiment import (
    logit_kl_and_match,
    masked_ce,
    train_translators_return_models,
    translated_attention_hidden,
)


@torch.no_grad()
def oracle_attention_hidden(item, attn_module):
    out_heads = item["selective_oracle_out"]
    hidden = out_heads.transpose(1, 2).contiguous().view(out_heads.shape[0], out_heads.shape[2], -1)
    return attn_module.o_proj(hidden.to(attn_module.o_proj.weight.dtype))


def eval_multilayer_patch(receiver, input_ids_all, mask_all, texts, per_layer, replacement_fn, alpha):
    totals = {"full_ce": 0.0, "patched_ce": 0.0, "logit_kl": 0.0, "top1_match": 0.0}
    for sample_id in range(len(texts)):
        input_ids = input_ids_all[sample_id : sample_id + 1]
        attention_mask = mask_all[sample_id : sample_id + 1]
        full_logits = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        handles = []
        replacements = {}
        for layer, data in per_layer.items():
            replacements[layer] = replacement_fn(layer, data, sample_id)

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

        patched_logits = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        for handle in handles:
            handle.remove()

        full_ce = masked_ce(full_logits, input_ids, attention_mask)
        patched_ce = masked_ce(patched_logits, input_ids, attention_mask)
        kl, match = logit_kl_and_match(full_logits, patched_logits, attention_mask)
        totals["full_ce"] += float(full_ce.cpu())
        totals["patched_ce"] += float(patched_ce.cpu())
        totals["logit_kl"] += kl
        totals["top1_match"] += match
    n = len(texts)
    return {
        "full_ce": totals["full_ce"] / n,
        "patched_ce": totals["patched_ce"] / n,
        "ce_delta": totals["patched_ce"] / n - totals["full_ce"] / n,
        "logit_kl": totals["logit_kl"] / n,
        "top1_match": totals["top1_match"] / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layer_sets", default="12,15;8,12,15")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--methods", default="baseline_1slot_linear,multislot_headwise_norm")
    parser.add_argument("--include_selective_oracle", action="store_true")
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/generation_effect_multilayer.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)
    enc = receiver.tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    input_ids_all = enc["input_ids"].to(device)
    mask_all = enc["attention_mask"].to(device)

    rows = []
    method_configs = [parse_method(name.strip(), args.slots_per_block) for name in args.methods.split(",") if name.strip()]
    layer_sets = [parse_int_list(chunk) for chunk in args.layer_sets.split(";") if chunk.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    for layers in layer_sets:
        layer_key = ",".join(str(x) for x in layers)
        oracle_items = None
        if args.include_selective_oracle:
            oracle_method = parse_method("baseline_1slot_linear", args.slots_per_block)
            oracle_items = {}
            for layer in layers:
                oracle_items[layer] = {
                    "items": build_items(
                        features_s,
                        features_r,
                        layer,
                        args.block_size,
                        args.block_score_mode,
                        args.anchor_tokens,
                        args.budget_ratio,
                        oracle_method,
                        args.value_pool_mode,
                        device,
                    ),
                    "attn_module": receiver.extractor.attn_layers[layer][1],
                }
            for alpha in alphas:
                metrics = eval_multilayer_patch(
                    receiver,
                    input_ids_all,
                    mask_all,
                    texts,
                    oracle_items,
                    lambda layer, data, sample_id: oracle_attention_hidden(data["items"][sample_id], data["attn_module"]),
                    alpha,
                )
                row = {"layers": layer_key, "method": "selective_oracle", "alpha": alpha, **metrics}
                rows.append(row)
                print(
                    f"L[{layer_key}] {'selective_oracle':<24} alpha={alpha:.2f} "
                    f"full_ce={row['full_ce']:.4f} patched_ce={row['patched_ce']:.4f} "
                    f"delta={row['ce_delta']:+.4f} kl={row['logit_kl']:.4f} top1={row['top1_match']:.3f}"
                )

        for method in method_configs:
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
                    "attn_module": receiver.extractor.attn_layers[layer][1],
                }

            for alpha in alphas:
                metrics = eval_multilayer_patch(
                    receiver,
                    input_ids_all,
                    mask_all,
                    texts,
                    per_layer,
                    lambda layer, data, sample_id: translated_attention_hidden(
                            data["items"][sample_id],
                            data["tk"],
                            data["tv"],
                            data["prep_input"],
                            data["denorm_pred"],
                            method,
                            data["attn_module"],
                    ),
                    alpha,
                )
                row = {
                    "layers": layer_key,
                    "method": method.name,
                    "alpha": alpha,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"L[{layer_key}] {method.name:<24} alpha={alpha:.2f} "
                    f"full_ce={row['full_ce']:.4f} patched_ce={row['patched_ce']:.4f} "
                    f"delta={row['ce_delta']:+.4f} kl={row['logit_kl']:.4f} top1={row['top1_match']:.3f}"
                )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = ["layers", "method", "alpha", "full_ce", "patched_ce", "ce_delta", "logit_kl", "top1_match"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
