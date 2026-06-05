"""Rule-based K-routing prior ablation.

Compares K norm, received attention mass, K norm x received, K outlier score,
local K variance, and block K energy as sender-side routing priors.
"""

import argparse
import csv
import os

import torch

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from evidence_recall_selective_recompute import (
    LLAMA_3_2_1B,
    attention_probs,
    collect_features,
    evaluate_candidate_mode,
    load_bundle,
    load_text_files,
    map_sender_mask_to_receiver,
    topk_mask_from_scores,
)


def local_variance(x, window):
    pad = window // 2
    unfolded = torch.nn.functional.pad(x.transpose(-1, -2), (pad, pad), mode="constant", value=0.0).unfold(-1, window, 1)
    return unfolded.var(dim=-1).mean(dim=-2)


def expand_to_blocks(mask, block_size):
    out = torch.zeros_like(mask)
    for start in range(0, mask.shape[-1], block_size):
        end = min(start + block_size, mask.shape[-1])
        out[..., start:end] = mask[..., start:end].any(dim=-1, keepdim=True)
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layers", default="0,4,8,12,15")
    parser.add_argument("--keep_tokens", type=int, default=64)
    parser.add_argument("--receiver_block_size", type=int, default=32)
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--csv", default="runs/rule_k_routing_prior.csv")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    fs = collect_features(sender, texts, args.max_length, device)
    fr = collect_features(receiver, texts, args.max_length, device)
    rows = []
    for layer in parse_int_list(args.layers):
        if layer >= sender.model.config.num_hidden_layers or layer >= receiver.model.config.num_hidden_layers:
            continue
        totals, counts = {}, {}
        for s, r in zip(fs, fr):
            q_s, k_s = s["qkv"].q[layer].to(device), s["qkv"].k[layer].to(device)
            q_r, k_r, v_r = r["qkv"].q[layer].to(device), r["qkv"].k[layer].to(device), r["qkv"].v[layer].to(device)
            s_mask, r_mask = s["mask"].to(device), r["mask"].to(device)
            if q_s.shape[1] < q_r.shape[1]:
                q_s = q_s.repeat_interleave(q_r.shape[1] // q_s.shape[1], dim=1)
                k_s = k_s.repeat_interleave(q_r.shape[1] // k_s.shape[1], dim=1)
            else:
                q_s, k_s = q_s[:, : q_r.shape[1]], k_s[:, : q_r.shape[1]]
            attn_s = attention_probs(q_s, k_s, s_mask)
            attn_r = attention_probs(q_r, k_r, r_mask)
            k_norm = k_s.norm(dim=-1)
            received = (attn_s * s_mask[:, None, :, None].float()).sum(dim=2)
            outlier = (k_s.abs() > (k_s.abs().mean(dim=-1, keepdim=True) + 2 * k_s.abs().std(dim=-1, keepdim=True))).float().sum(dim=-1)
            variance = local_variance(k_s, 7)
            priors = {
                "k_norm": k_norm,
                "received_mass": received,
                "k_norm_x_received": k_norm * received,
                "k_outlier": outlier,
                "local_k_variance": variance,
            }
            for name, scores in priors.items():
                mask_s = topk_mask_from_scores(scores[:, :, None, :].expand_as(attn_s), args.keep_tokens)
                mask_r = map_sender_mask_to_receiver(mask_s, s["offsets"].to(device), r["offsets"].to(device))
                mask_r = expand_to_blocks(mask_r, args.receiver_block_size)
                mask_r = mask_r & r_mask[:, None, :, None].bool() & r_mask[:, None, None, :].bool()
                row = evaluate_candidate_mode(name, mask_r, attn_r, q_r, k_r, v_r, r_mask, args.route_k)
                mode = row.pop("mode")
                counts[mode] = counts.get(mode, 0) + 1
                for key, value in row.items():
                    totals[(mode, key)] = totals.get((mode, key), 0.0) + value
        for mode in sorted(counts):
            row = {"layer": layer, "mode": mode}
            for (m, key), value in totals.items():
                if m == mode:
                    row[key] = value / counts[mode]
            print(f"L{layer:02d} {mode:<20} sel={row['selected_ratio']:.3f} mass={row['candidate_mass']:.4f} cos={row['selective_recompute_cosine']:.4f}")
            rows.append(row)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = ["layer", "mode", "selected_tokens", "selected_ratio", "candidate_mass", "gold_topk_recall", "oracle_out_mse", "oracle_out_cosine", "selective_recompute_mse", "selective_recompute_cosine"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
