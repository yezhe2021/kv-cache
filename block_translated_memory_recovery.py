"""Block-level translated sender memory recovery.

This MVP tests a stronger variant than selective recompute:

    sender K/V + sender routing prior -> sender block memories
    sender block memories -> lightweight translated receiver block memories
    receiver Q attends translated block memories to approximate full output

It does not require receiver historical K/V during evaluation, except for
building the full-output target and supervised training targets.
"""

import argparse
import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_preserving_kv_translation_experiment import (
    LOCAL_SENDER_MODEL,
    LinearKVTranslator,
    align_heads,
    attention_output,
    attention_probs,
    output_cosine,
    output_mse,
    parse_int_list,
)
from evidence_recall_selective_recompute import (
    LLAMA_3_2_1B,
    collect_features,
    load_bundle,
    load_text_files,
    offset_overlap_matrix,
)


def receiver_block_sender_masks(sender_offsets, receiver_offsets, receiver_mask, block_size, device):
    overlap = offset_overlap_matrix(sender_offsets, receiver_offsets).to(device)
    valid_len = int(receiver_mask[0].sum().item())
    seq_len = receiver_offsets.shape[1]
    masks = []
    block_starts = []
    block_ends = []
    valid_blocks = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        recv_valid = torch.zeros(seq_len, dtype=torch.bool, device=device)
        recv_valid[start:min(end, valid_len)] = True
        if recv_valid.any():
            sender_mask = overlap[recv_valid].any(dim=0)
            valid = bool(sender_mask.any().item())
        else:
            sender_mask = torch.zeros(overlap.shape[1], dtype=torch.bool, device=device)
            valid = False
        masks.append(sender_mask)
        block_starts.append(start)
        block_ends.append(min(end, valid_len))
        valid_blocks.append(valid)
    return (
        torch.stack(masks, dim=0),
        torch.tensor(block_starts, device=device),
        torch.tensor(block_ends, device=device),
        torch.tensor(valid_blocks, dtype=torch.bool, device=device),
    )


def pool_by_masks(x, token_weights, block_token_masks, valid_blocks):
    # x: [1, heads, seq, dim], token_weights: [1, heads, seq]
    memories = []
    for block_id in range(block_token_masks.shape[0]):
        mask = block_token_masks[block_id][None, None, :].to(x.device)
        weights = token_weights.masked_fill(~mask, 0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (x * weights[..., None]).sum(dim=-2) / denom
        if not bool(valid_blocks[block_id].item()):
            pooled = torch.zeros_like(pooled)
        memories.append(pooled)
    return torch.stack(memories, dim=2)


def score_blocks(token_weights, block_token_masks, valid_blocks, mode="anchor_count", anchor_tokens=64, topk=4):
    if mode == "anchor_count":
        valid_token_scores = token_weights.mean(dim=1)
        keep = min(anchor_tokens, valid_token_scores.shape[-1])
        anchor_idx = torch.topk(valid_token_scores, k=keep, dim=-1).indices
        anchor_mask = torch.zeros_like(valid_token_scores, dtype=torch.bool)
        anchor_mask.scatter_(-1, anchor_idx, True)

    scores = []
    for block_id in range(block_token_masks.shape[0]):
        mask = block_token_masks[block_id][None, None, :].to(token_weights.device)
        if mode == "mean":
            masked = token_weights.masked_fill(~mask, 0.0)
            denom = mask.float().sum(dim=-1).clamp_min(1.0)
            score = (masked.sum(dim=-1) / denom).mean()
        elif mode == "sum":
            score = token_weights.masked_fill(~mask, 0.0).sum(dim=-1).mean()
        elif mode == "max":
            score = token_weights.masked_fill(~mask, float("-inf")).amax(dim=-1).mean()
        elif mode in {"topk_mean", "topk_sum"}:
            masked = token_weights.masked_fill(~mask, float("-inf"))
            k = min(topk, int(mask.sum().item()))
            if k <= 0:
                score = torch.tensor(float("-inf"), device=token_weights.device)
            else:
                values = torch.topk(masked, k=k, dim=-1).values
                score = values.mean() if mode == "topk_mean" else values.sum(dim=-1).mean()
        elif mode == "anchor_count":
            block_anchor_count = (anchor_mask & block_token_masks[block_id][None, :]).float().sum(dim=-1)
            score = block_anchor_count.mean()
        else:
            raise ValueError(f"Unknown block score mode: {mode}")
        if not bool(valid_blocks[block_id].item()):
            score = torch.tensor(float("-inf"), device=token_weights.device)
        scores.append(score)
    return torch.stack(scores, dim=0)


def select_blocks(block_scores, valid_blocks, keep_blocks=None, budget_ratio=None):
    num_valid = int(valid_blocks.sum().item())
    if num_valid == 0:
        return valid_blocks
    if keep_blocks is None:
        keep = max(1, int(round(num_valid * budget_ratio)))
    else:
        keep = keep_blocks
    keep = max(1, min(keep, num_valid))
    masked_scores = block_scores.masked_fill(~valid_blocks, float("-inf"))
    idx = torch.topk(masked_scores, k=keep).indices
    selected = torch.zeros_like(valid_blocks)
    selected[idx] = True
    return selected & valid_blocks


def receiver_block_targets(x, receiver_mask, block_size):
    seq_len = x.shape[-2]
    valid_len = int(receiver_mask[0].sum().item())
    memories = []
    valid_blocks = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, valid_len)
        if end > start:
            memories.append(x[..., start:end, :].mean(dim=-2))
            valid_blocks.append(True)
        else:
            memories.append(torch.zeros_like(x[..., 0, :]))
            valid_blocks.append(False)
    return torch.stack(memories, dim=2), torch.tensor(valid_blocks, dtype=torch.bool, device=x.device)


def sender_token_weights(k_s, attn_s, sender_mask, mode):
    valid = sender_mask[:, None, :].to(k_s.device).bool()
    if mode == "uniform":
        scores = torch.ones(k_s.shape[:3], dtype=k_s.dtype, device=k_s.device)
    elif mode == "k_norm":
        scores = k_s.norm(dim=-1)
    elif mode == "received":
        scores = (attn_s * valid[:, :, None, :].float()).sum(dim=2)
    elif mode == "kxreceived":
        received = (attn_s * valid[:, :, None, :].float()).sum(dim=2)
        scores = received * k_s.norm(dim=-1)
    else:
        raise ValueError(f"Unknown pool mode: {mode}")
    return scores.masked_fill(~valid, 0.0)


def sender_value_weights(v_s, attn_s, sender_mask, mode):
    valid = sender_mask[:, None, :].to(v_s.device).bool()
    if mode == "uniform":
        scores = torch.ones(v_s.shape[:3], dtype=v_s.dtype, device=v_s.device)
    elif mode == "v_norm":
        scores = v_s.norm(dim=-1)
    elif mode == "received":
        scores = (attn_s * valid[:, :, None, :].float()).sum(dim=2)
    else:
        raise ValueError(f"Unknown value pool mode: {mode}")
    return scores.masked_fill(~valid, 0.0)


def translated_block_attention(q_r, k_mem, v_mem, valid_blocks, block_starts, receiver_mask):
    # Block-start causal masking keeps blocks whose first token is not in the future.
    d = q_r.shape[-1]
    scores = torch.matmul(q_r, k_mem.transpose(-1, -2)) / math.sqrt(d)
    seq_len = q_r.shape[-2]
    query_pos = torch.arange(seq_len, device=q_r.device)
    causal_blocks = block_starts[None, :] <= query_pos[:, None]
    block_mask = valid_blocks[None, None, None, :] & causal_blocks[None, None, :, :]
    scores = scores.masked_fill(~block_mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_mem)
    return out, attn


def selected_block_candidate_mask(attn_shape, valid_blocks, block_starts, block_ends, receiver_mask, device):
    key_selected = torch.zeros(attn_shape[-1], dtype=torch.bool, device=device)
    for block_id in range(valid_blocks.shape[0]):
        if bool(valid_blocks[block_id].item()):
            key_selected[int(block_starts[block_id].item()) : int(block_ends[block_id].item())] = True
    candidate = key_selected[None, None, None, :].expand(attn_shape)
    candidate = candidate & receiver_mask[:, None, None, :].bool()
    return candidate


def sparse_normalize(weights, mask):
    weights = weights.masked_fill(~mask, 0.0)
    denom = weights.sum(dim=-1, keepdim=True)
    return weights / denom.clamp_min(1e-8)


def collect_block_items(
    features_s,
    features_r,
    layer,
    block_size,
    routing_pool_mode,
    value_pool_mode,
    keep_blocks,
    budget_ratio,
    block_score_mode,
    anchor_tokens,
    block_score_topk,
    device,
):
    items = []
    for fs, fr in zip(features_s, features_r):
        q_s = align_heads(fs["qkv"].q[layer].to(device), fr["qkv"].q[layer].shape[1])
        k_s = align_heads(fs["qkv"].k[layer].to(device), fr["qkv"].q[layer].shape[1])
        v_s = align_heads(fs["qkv"].v[layer].to(device), fr["qkv"].q[layer].shape[1])
        q_r = fr["qkv"].q[layer].to(device)
        k_r = fr["qkv"].k[layer].to(device)
        v_r = fr["qkv"].v[layer].to(device)
        s_mask = fs["mask"].to(device)
        r_mask = fr["mask"].to(device)

        attn_s = attention_probs(q_s, k_s, s_mask)
        attn_r = attention_probs(q_r, k_r, r_mask)
        full_out = attention_output(attn_r, v_r)

        block_sender_masks, block_starts, block_ends, valid_from_sender = receiver_block_sender_masks(
            fs["offsets"].to(device),
            fr["offsets"].to(device),
            r_mask,
            block_size,
            device,
        )
        routing_weights = sender_token_weights(k_s, attn_s, s_mask, routing_pool_mode)
        value_weights = sender_value_weights(v_s, attn_s, s_mask, value_pool_mode)
        sender_k_mem = pool_by_masks(k_s, routing_weights, block_sender_masks, valid_from_sender)
        sender_v_mem = pool_by_masks(v_s, value_weights, block_sender_masks, valid_from_sender)
        recv_k_mem, valid_recv = receiver_block_targets(k_r, r_mask, block_size)
        recv_v_mem, _ = receiver_block_targets(v_r, r_mask, block_size)
        candidate_blocks = valid_from_sender & valid_recv
        block_scores = score_blocks(
            routing_weights,
            block_sender_masks,
            candidate_blocks,
            mode=block_score_mode,
            anchor_tokens=anchor_tokens,
            topk=block_score_topk,
        )
        valid_blocks = select_blocks(block_scores, candidate_blocks, keep_blocks, budget_ratio)
        selective_candidate_mask = selected_block_candidate_mask(
            attn_r.shape,
            valid_blocks,
            block_starts,
            block_ends,
            r_mask,
            device,
        )
        selective_oracle_attn = sparse_normalize(attn_r, selective_candidate_mask)
        selective_oracle_out = attention_output(selective_oracle_attn, v_r)
        items.append(
            {
                "sender_k_mem": sender_k_mem,
                "sender_v_mem": sender_v_mem,
                "recv_k_mem": recv_k_mem,
                "recv_v_mem": recv_v_mem,
                "q_r": q_r,
                "v_r": v_r,
                "r_mask": r_mask,
                "full_out": full_out,
                "selective_oracle_out": selective_oracle_out,
                "valid_blocks": valid_blocks,
                "block_starts": block_starts,
                "block_ends": block_ends,
            }
        )
    return items


def train_translators(items, sender_dim, receiver_dim, epochs, lr, kv_loss_weight, output_loss_weight, device):
    tk = LinearKVTranslator(sender_dim, receiver_dim).to(device)
    tv = LinearKVTranslator(sender_dim, receiver_dim).to(device)
    opt = torch.optim.AdamW(list(tk.parameters()) + list(tv.parameters()), lr=lr)
    for _ in range(epochs):
        for item in items:
            valid = item["valid_blocks"][None, None, :, None]
            pred_k = tk(item["sender_k_mem"])
            pred_v = tv(item["sender_v_mem"])
            loss_terms = []
            if kv_loss_weight > 0:
                loss_k = F.mse_loss(pred_k.masked_select(valid), item["recv_k_mem"].masked_select(valid))
                loss_v = F.mse_loss(pred_v.masked_select(valid), item["recv_v_mem"].masked_select(valid))
                loss_terms.append(kv_loss_weight * (loss_k + loss_v))
            if output_loss_weight > 0:
                out_hat, _ = translated_block_attention(
                    item["q_r"],
                    pred_k,
                    pred_v,
                    item["valid_blocks"],
                    item["block_starts"],
                    item["r_mask"],
                )
                qmask = item["r_mask"].bool()[:, None, :, None].to(out_hat.device)
                out_loss = F.mse_loss(out_hat.masked_select(qmask), item["full_out"].masked_select(qmask))
                loss_terms.append(output_loss_weight * out_loss)
            loss = sum(loss_terms)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return tk, tv


@torch.no_grad()
def evaluate_items(items, tk, tv):
    totals = {
        "translated_mse": 0.0,
        "translated_cosine": 0.0,
        "mean_block_mse": 0.0,
        "mean_block_cosine": 0.0,
        "selective_oracle_mse": 0.0,
        "selective_oracle_cosine": 0.0,
        "selected_ratio": 0.0,
    }
    for item in items:
        translated_k = tk(item["sender_k_mem"])
        translated_v = tv(item["sender_v_mem"])
        out_hat, _ = translated_block_attention(
            item["q_r"],
            translated_k,
            translated_v,
            item["valid_blocks"],
            item["block_starts"],
            item["r_mask"],
        )
        mean_block_out, _ = translated_block_attention(
            item["q_r"],
            item["recv_k_mem"],
            item["recv_v_mem"],
            item["valid_blocks"],
            item["block_starts"],
            item["r_mask"],
        )
        qmask = item["r_mask"].bool()
        totals["translated_mse"] += float(output_mse(out_hat, item["full_out"], qmask).cpu())
        totals["translated_cosine"] += float(output_cosine(out_hat, item["full_out"], qmask).cpu())
        totals["mean_block_mse"] += float(output_mse(mean_block_out, item["full_out"], qmask).cpu())
        totals["mean_block_cosine"] += float(output_cosine(mean_block_out, item["full_out"], qmask).cpu())
        totals["selective_oracle_mse"] += float(output_mse(item["selective_oracle_out"], item["full_out"], qmask).cpu())
        totals["selective_oracle_cosine"] += float(output_cosine(item["selective_oracle_out"], item["full_out"], qmask).cpu())
        totals["selected_ratio"] += float(item["valid_blocks"].float().mean().cpu())
    count = max(1, len(items))
    return {key: value / count for key, value in totals.items()}


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
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--keep_blocks", type=int, default=None)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument(
        "--block_score_mode",
        default="anchor_count",
        choices=["anchor_count", "max", "topk_mean", "topk_sum", "mean", "sum"],
    )
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--block_score_topk", type=int, default=4)
    parser.add_argument("--pool_modes", default="uniform,received,kxreceived")
    parser.add_argument("--value_pool_mode", default="uniform", choices=["uniform", "v_norm", "received"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/block_translated_memory_recovery.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)

    rows = []
    for layer in parse_int_list(args.layers):
        if layer >= sender.model.config.num_hidden_layers or layer >= receiver.model.config.num_hidden_layers:
            continue
        for pool_mode in [x.strip() for x in args.pool_modes.split(",") if x.strip()]:
            items = collect_block_items(
                features_s,
                features_r,
                layer,
                args.block_size,
                pool_mode,
                args.value_pool_mode,
                args.keep_blocks,
                args.budget_ratio,
                args.block_score_mode,
                args.anchor_tokens,
                args.block_score_topk,
                device,
            )
            sender_dim = items[0]["sender_k_mem"].shape[-1]
            receiver_dim = items[0]["recv_k_mem"].shape[-1]
            tk, tv = train_translators(
                items,
                sender_dim,
                receiver_dim,
                args.epochs,
                args.lr,
                args.kv_loss_weight,
                args.output_loss_weight,
                device,
            )
            metrics = evaluate_items(items, tk, tv)
            row = {
                "layer": layer,
                "routing_pool_mode": pool_mode,
                "value_pool_mode": args.value_pool_mode,
                "block_size": args.block_size,
                "keep_blocks": args.keep_blocks if args.keep_blocks is not None else "",
                "budget_ratio": args.budget_ratio,
                "block_score_mode": args.block_score_mode,
                "anchor_tokens": args.anchor_tokens,
                "block_score_topk": args.block_score_topk,
                **metrics,
            }
            rows.append(row)
            print(
                f"L{layer:02d} {pool_mode:<10} "
                f"ratio={row['selected_ratio']:.3f} "
                f"trans_cos={row['translated_cosine']:.4f} "
                f"selective_oracle_cos={row['selective_oracle_cosine']:.4f} "
                f"mean_block_cos={row['mean_block_cosine']:.4f} "
                f"trans_mse={row['translated_mse']:.6f}"
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "layer",
        "routing_pool_mode",
        "value_pool_mode",
        "block_size",
        "keep_blocks",
        "budget_ratio",
        "block_score_mode",
        "anchor_tokens",
        "block_score_topk",
        "selected_ratio",
        "translated_mse",
        "translated_cosine",
        "mean_block_mse",
        "mean_block_cosine",
        "selective_oracle_mse",
        "selective_oracle_cosine",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
