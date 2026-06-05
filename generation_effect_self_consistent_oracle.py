"""Self-consistent selective-oracle patch evaluation.

Previous multi-layer oracle patches used replacement tensors precomputed on the
original receiver hidden trajectory. This script tests a stricter oracle:

* selected blocks are still chosen once from sender routing prior;
* during the receiver forward pass, each patched layer recomputes Q/K/V from the
  current hidden states seen by that layer;
* attention is restricted to the selected blocks;
* the replacement can be blended with the original attention output.

This isolates whether multi-layer quality drops because the oracle replacement
was not trajectory-consistent.
"""

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from block_memory_method_sweep import build_items, parse_method
from evidence_recall_selective_recompute import LLAMA_3_2_1B, collect_features, load_bundle, load_text_files
from generation_effect_experiment import logit_kl_and_match, masked_ce


def infer_heads(attn_module):
    cfg = getattr(attn_module, "config", None)
    q_heads = (
        getattr(attn_module, "num_heads", None)
        or getattr(attn_module, "num_attention_heads", None)
        or getattr(cfg, "num_attention_heads", None)
    )
    kv_heads = (
        getattr(attn_module, "num_key_value_heads", None)
        or getattr(cfg, "num_key_value_heads", None)
        or q_heads
    )
    if q_heads is None:
        raise RuntimeError("Cannot infer attention head count.")
    return int(q_heads), int(kv_heads)


def reshape_heads(x, heads):
    bsz, seq_len, hidden = x.shape
    head_dim = hidden // heads
    return x.view(bsz, seq_len, heads, head_dim).transpose(1, 2).contiguous()


def current_qkv(attn_module, hidden_states):
    q_heads, kv_heads = infer_heads(attn_module)
    q = reshape_heads(attn_module.q_proj(hidden_states).float(), q_heads)
    k = reshape_heads(attn_module.k_proj(hidden_states).float(), kv_heads)
    v = reshape_heads(attn_module.v_proj(hidden_states).float(), kv_heads)
    if kv_heads != q_heads:
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return q, k, v


def selected_key_mask_from_item(item, block_size, seq_len, device):
    key_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    starts = item["slot_starts"]
    valid = item["valid_slots"]
    for start in torch.unique(starts[valid]):
        s = int(start.item())
        key_mask[s : min(s + block_size, seq_len)] = True
    return key_mask


def restricted_attention_output(q, k, v, key_selected, attention_mask):
    d = q.shape[-1]
    seq_len = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
    valid_key = attention_mask[:, None, None, :].bool()
    selected = key_selected[None, None, None, :]
    scores = scores.masked_fill(causal[None, None, :, :] | ~valid_key | ~selected, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@torch.no_grad()
def static_oracle_hidden(item, attn_module):
    out_heads = item["selective_oracle_out"]
    hidden = out_heads.transpose(1, 2).contiguous().view(out_heads.shape[0], out_heads.shape[2], -1)
    return attn_module.o_proj(hidden.to(attn_module.o_proj.weight.dtype))


@torch.no_grad()
def self_consistent_hidden(attn_module, hidden_states, item, block_size, attention_mask):
    q, k, v = current_qkv(attn_module, hidden_states)
    key_selected = selected_key_mask_from_item(item, block_size, q.shape[-2], q.device)
    out_heads = restricted_attention_output(q, k, v, key_selected, attention_mask)
    hidden = out_heads.transpose(1, 2).contiguous().view(out_heads.shape[0], out_heads.shape[2], -1)
    return attn_module.o_proj(hidden.to(attn_module.o_proj.weight.dtype))


def eval_patch(receiver, input_ids_all, mask_all, texts, per_layer, mode, alpha, block_size):
    totals = {"full_ce": 0.0, "patched_ce": 0.0, "logit_kl": 0.0, "top1_match": 0.0}
    for sample_id in range(len(texts)):
        input_ids = input_ids_all[sample_id : sample_id + 1]
        attention_mask = mask_all[sample_id : sample_id + 1]
        full_logits = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        handles = []
        static_replacements = {}
        if mode == "static":
            for layer, data in per_layer.items():
                static_replacements[layer] = static_oracle_hidden(data["items"][sample_id], data["attn_module"])

        for layer, data in per_layer.items():
            def make_hook(layer_id, layer_data):
                def hook(module, inputs, kwargs, output):
                    original = output[0] if isinstance(output, tuple) else output
                    if mode == "static":
                        replacement = static_replacements[layer_id]
                    else:
                        hidden_states = kwargs.get("hidden_states")
                        if hidden_states is None:
                            hidden_states = inputs[0]
                        replacement = self_consistent_hidden(module, hidden_states, layer_data["items"][sample_id], block_size, attention_mask)
                    replacement = replacement.to(original.dtype)
                    patched = alpha * replacement + (1.0 - alpha) * original
                    if isinstance(output, tuple):
                        return (patched,) + output[1:]
                    return patched

                return hook

            handles.append(data["attn_module"].register_forward_hook(make_hook(layer, data), with_kwargs=True))

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
    parser.add_argument("--layer_sets", default="12,15;8,12,15;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--alphas", default="1.0,0.75,0.5")
    parser.add_argument("--modes", default="static,self_consistent")
    parser.add_argument("--csv", default="runs/generation_effect_self_consistent_oracle.csv")
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
    method = parse_method("baseline_1slot_linear", 1)
    layer_sets = [parse_int_list(chunk) for chunk in args.layer_sets.split(";") if chunk.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for layers in layer_sets:
        layer_key = ",".join(str(x) for x in layers)
        per_layer = {}
        for layer in layers:
            per_layer[layer] = {
                "items": build_items(
                    features_s,
                    features_r,
                    layer,
                    args.block_size,
                    args.block_score_mode,
                    args.anchor_tokens,
                    args.budget_ratio,
                    method,
                    "uniform",
                    device,
                ),
                "attn_module": receiver.extractor.attn_layers[layer][1],
            }
        for mode in modes:
            for alpha in alphas:
                metrics = eval_patch(receiver, input_ids_all, mask_all, texts, per_layer, mode, alpha, args.block_size)
                row = {"layers": layer_key, "mode": mode, "alpha": alpha, **metrics}
                rows.append(row)
                print(
                    f"L[{layer_key}] {mode:<16} alpha={alpha:.2f} "
                    f"full_ce={row['full_ce']:.4f} patched_ce={row['patched_ce']:.4f} "
                    f"delta={row['ce_delta']:+.4f} kl={row['logit_kl']:.4f} top1={row['top1_match']:.3f}"
                )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = ["layers", "mode", "alpha", "full_ce", "patched_ce", "ce_delta", "logit_kl", "top1_match"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
