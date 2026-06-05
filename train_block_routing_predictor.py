"""Train a lightweight sender-K -> receiver evidence-block predictor.

This is an MVP predictor: for each receiver block, features are sender block
statistics mapped by character span overlap; labels are receiver blocks whose
gold attention mass is above the top block budget.
"""

import argparse
import csv
import os

import torch
import torch.nn as nn

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from evidence_recall_selective_recompute import (
    LLAMA_3_2_1B,
    attention_probs,
    collect_features,
    evaluate_candidate_mode,
    load_bundle,
    load_text_files,
    map_sender_mask_to_receiver,
)


class BlockPredictor(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def block_features(k_s, attn_s, s_mask, block_size):
    bsz, heads, seq_len, dim = k_s.shape
    rows = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        k_blk = k_s[..., start:end, :]
        received = (attn_s[..., start:end] * s_mask[:, None, :, None].float()).sum(dim=2)
        rows.append(
            torch.stack(
                [
                    k_blk.norm(dim=-1).mean(dim=-1),
                    k_blk.norm(dim=-1).max(dim=-1).values,
                    k_blk.var(dim=-2).mean(dim=-1),
                    received.mean(dim=-1),
                    received.max(dim=-1).values,
                ],
                dim=-1,
            )
        )
    return torch.stack(rows, dim=2)


def block_mask_from_logits(logits, keep_blocks, seq_len, block_size):
    idx = torch.topk(logits, k=min(keep_blocks, logits.shape[-1]), dim=-1).indices
    block_mask = torch.zeros_like(logits, dtype=torch.bool)
    block_mask.scatter_(-1, idx, True)
    token_mask = torch.zeros(logits.shape[0], logits.shape[1], seq_len, dtype=torch.bool, device=logits.device)
    for block_id, start in enumerate(range(0, seq_len, block_size)):
        end = min(start + block_size, seq_len)
        token_mask[..., start:end] = block_mask[..., block_id : block_id + 1]
    return token_mask[:, :, None, :].expand(logits.shape[0], logits.shape[1], seq_len, seq_len)


@torch.no_grad()
def receiver_oracle_block_labels(attn_r, r_mask, keep_blocks, block_size):
    seq_len = attn_r.shape[-1]
    scores = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        scores.append(attn_r[..., start:end].sum(dim=-1).mean(dim=-1))
    scores = torch.stack(scores, dim=-1)
    scores = scores.masked_fill(torch.isnan(scores), 0.0)
    idx = torch.topk(scores, k=min(keep_blocks, scores.shape[-1]), dim=-1).indices
    labels = torch.zeros_like(scores)
    labels.scatter_(-1, idx, 1.0)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layers", default="0,4,8,12,15")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--keep_blocks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--csv", default="runs/train_block_routing_predictor.csv")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, 20000)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    fs = collect_features(sender, texts, args.max_length, device)
    fr = collect_features(receiver, texts, args.max_length, device)
    rows = []
    for layer in parse_int_list(args.layers):
        if layer >= sender.model.config.num_hidden_layers or layer >= receiver.model.config.num_hidden_layers:
            continue
        model = BlockPredictor().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_data = []
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
            x = block_features(k_s, attn_s, s_mask, args.block_size)
            y = receiver_oracle_block_labels(attn_r, r_mask, args.keep_blocks, args.block_size)
            train_data.append((x, y, s, r, q_r, k_r, v_r, r_mask))
        for _ in range(args.epochs):
            for x, y, *_ in train_data:
                loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        totals, counts = {}, {}
        for x, _, s, r, q_r, k_r, v_r, r_mask in train_data:
            logits = model(x)
            mask_s = block_mask_from_logits(logits, args.keep_blocks, q_r.shape[-2], args.block_size)
            mask_r = map_sender_mask_to_receiver(mask_s, s["offsets"].to(device), r["offsets"].to(device))
            mask_r = mask_r & r_mask[:, None, :, None].bool() & r_mask[:, None, None, :].bool()
            row = evaluate_candidate_mode("learned_block_predictor", mask_r, attention_probs(q_r, k_r, r_mask), q_r, k_r, v_r, r_mask, args.route_k)
            mode = row.pop("mode")
            counts[mode] = counts.get(mode, 0) + 1
            for key, value in row.items():
                totals[(mode, key)] = totals.get((mode, key), 0.0) + value
        row = {"layer": layer, "mode": "learned_block_predictor"}
        for (m, key), value in totals.items():
            row[key] = value / counts[m]
        print(f"L{layer:02d} learned sel={row['selected_ratio']:.3f} mass={row['candidate_mass']:.4f} cos={row['selective_recompute_cosine']:.4f}")
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
