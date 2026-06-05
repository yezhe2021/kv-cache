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
    make_global_important_mask,
    make_k_saliency_mask,
    receiver_candidate_scores,
    topk_mask_from_scores,
)


def block_ids(seq_len, block_size, device):
    return torch.arange(seq_len, device=device) // block_size


def expand_token_mask_to_blocks(token_mask, block_size):
    out = torch.zeros_like(token_mask)
    key_len = token_mask.shape[-1]
    for start in range(0, key_len, block_size):
        end = min(start + block_size, key_len)
        keep = token_mask[..., start:end].any(dim=-1, keepdim=True)
        out[..., start:end] = keep
    return out


def fixed_topk_mask(scores, keep_tokens):
    return topk_mask_from_scores(scores, keep_tokens)


def fixed_top_blocks_from_token_scores(token_scores, keep_tokens, block_size):
    seq_len = token_scores.shape[-1]
    num_blocks = (seq_len + block_size - 1) // block_size
    keep_blocks = max(1, min(num_blocks, (keep_tokens + block_size - 1) // block_size))
    block_score = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        block_score.append(token_scores[..., start:end].mean(dim=-1))
    block_score = torch.stack(block_score, dim=-1)
    block_idx = torch.topk(block_score, k=keep_blocks, dim=-1).indices
    block_mask = torch.zeros_like(block_score, dtype=torch.bool)
    block_mask.scatter_(-1, block_idx, True)
    ids = block_ids(seq_len, block_size, token_scores.device)
    return block_mask.gather(-1, ids.view(*([1] * (block_mask.ndim - 1)), -1))


def block_mask_from_block_scores(block_scores, seq_len, block_size, keep_blocks):
    keep_blocks = max(1, min(keep_blocks, block_scores.shape[-1]))
    idx = torch.topk(block_scores, k=keep_blocks, dim=-1).indices
    chosen = torch.zeros_like(block_scores, dtype=torch.bool)
    chosen.scatter_(-1, idx, True)
    out = torch.zeros(*block_scores.shape[:-1], seq_len, dtype=torch.bool, device=block_scores.device)
    for block_id, start in enumerate(range(0, seq_len, block_size)):
        end = min(start + block_size, seq_len)
        out[..., start:end] = chosen[..., block_id : block_id + 1]
    return out


def query_key_block_mask_from_token_scores(token_scores, block_size, keep_blocks):
    seq_len = token_scores.shape[-1]
    block_scores = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        block_scores.append(token_scores[..., start:end].mean(dim=-1))
    block_scores = torch.stack(block_scores, dim=-1)
    key_mask = block_mask_from_block_scores(block_scores, seq_len, block_size, keep_blocks)
    return key_mask[:, :, None, :].expand(token_scores.shape[0], token_scores.shape[1], seq_len, seq_len)


def block_scores_from_candidate_mask(mask, block_size):
    scores = []
    for start in range(0, mask.shape[-1], block_size):
        end = min(start + block_size, mask.shape[-1])
        scores.append(mask[..., start:end].float().mean(dim=-1))
    return torch.stack(scores, dim=-1)


def block_mask_from_candidate_mask(mask, block_size, keep_blocks):
    block_scores = block_scores_from_candidate_mask(mask, block_size)
    key_mask = block_mask_from_block_scores(block_scores, mask.shape[-1], block_size, keep_blocks)
    return key_mask


def recent_token_mask(shape, r_mask, keep_tokens):
    mask = torch.zeros(shape, dtype=torch.bool, device=r_mask.device)
    valid_len = int(r_mask[0].sum().item())
    start = max(0, valid_len - keep_tokens)
    mask[..., start:valid_len] = True
    return mask


def random_token_mask(shape, r_mask, keep_tokens, seed):
    gen = torch.Generator(device=r_mask.device).manual_seed(seed)
    scores = torch.rand(shape, generator=gen, device=r_mask.device)
    scores = scores.masked_fill(~r_mask[:, None, None, :].bool(), -1.0)
    return fixed_topk_mask(scores, keep_tokens)


def random_block_mask(shape, block_size, keep_blocks, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = (shape[-1] + block_size - 1) // block_size
    scores = torch.rand(*shape[:-1], num_blocks, generator=gen, device=device)
    key_mask = block_mask_from_block_scores(scores, shape[-1], block_size, keep_blocks)
    return key_mask


def shape_device(shape):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recent_block_mask(shape, valid_len, block_size, keep_blocks, device):
    num_blocks = (shape[-1] + block_size - 1) // block_size
    start_block = max(0, (valid_len - 1) // block_size - keep_blocks + 1)
    scores = torch.full((*shape[:-1], num_blocks), -1.0, device=device)
    scores[..., start_block : (valid_len + block_size - 1) // block_size] = 1.0
    return block_mask_from_block_scores(scores, shape[-1], block_size, keep_blocks)


def uniform_block_mask(shape, block_size, keep_blocks, device):
    num_blocks = (shape[-1] + block_size - 1) // block_size
    chosen = torch.linspace(0, num_blocks - 1, steps=keep_blocks, device=device).round().long().unique()
    scores = torch.full((*shape[:-1], num_blocks), -1.0, device=device)
    scores[..., chosen] = 1.0
    return block_mask_from_block_scores(scores, shape[-1], block_size, keep_blocks)


@torch.no_grad()
def run_layer(features_s, features_r, layer, args, device, ratio):
    totals, counts = {}, {}
    for sample_id, (fs, fr) in enumerate(zip(features_s, features_r)):
        q_s = fs["qkv"].q[layer].to(device)
        k_s = fs["qkv"].k[layer].to(device)
        q_r = fr["qkv"].q[layer].to(device)
        k_r = fr["qkv"].k[layer].to(device)
        v_r = fr["qkv"].v[layer].to(device)
        s_mask = fs["mask"].to(device)
        r_mask = fr["mask"].to(device)
        q_s = q_s.repeat_interleave(q_r.shape[1] // q_s.shape[1], dim=1) if q_s.shape[1] < q_r.shape[1] else q_s[:, : q_r.shape[1]]
        k_s = k_s.repeat_interleave(q_r.shape[1] // k_s.shape[1], dim=1) if k_s.shape[1] < q_r.shape[1] else k_s[:, : q_r.shape[1]]
        attn_s = attention_probs(q_s, k_s, s_mask)
        attn_r = attention_probs(q_r, k_r, r_mask)
        valid_len = int(r_mask[0].sum().item())
        keep_tokens = max(1, int(round(valid_len * ratio)))
        num_blocks = (valid_len + args.receiver_block_size - 1) // args.receiver_block_size
        keep_blocks = max(1, min(num_blocks, int(round(num_blocks * ratio))))

        received = (attn_s * s_mask[:, None, :, None].float()).sum(dim=(1, 2))
        received = received[:, None, None, :].expand_as(attn_s)
        k_norm = k_s.norm(dim=-1)
        k_norm = k_norm[:, :, None, :].expand_as(attn_s)
        combo = received * k_norm
        attn_anchor = topk_mask_from_scores(attn_s, min(args.anchor_k, keep_tokens))
        attn_mapped = map_sender_mask_to_receiver(attn_anchor, fs["offsets"].to(device), fr["offsets"].to(device))
        attn_block = block_mask_from_candidate_mask(attn_mapped, args.receiver_block_size, keep_blocks)

        sender_masks = {
            "random_token": random_token_mask(attn_r.shape, r_mask, keep_tokens, sample_id + layer * 1000),
            "recent_token": recent_token_mask(attn_r.shape, r_mask, keep_tokens),
            "random_block": random_block_mask(attn_r.shape, args.receiver_block_size, keep_blocks, sample_id + 7, device),
            "recent_block": recent_block_mask(attn_r.shape, valid_len, args.receiver_block_size, keep_blocks, device),
            "uniform_block": uniform_block_mask(attn_r.shape, args.receiver_block_size, keep_blocks, device),
            "sender_attn_block": attn_block,
            "sender_k_norm": map_sender_mask_to_receiver(fixed_topk_mask(k_norm, keep_tokens), fs["offsets"].to(device), fr["offsets"].to(device)),
            "sender_received": map_sender_mask_to_receiver(fixed_topk_mask(received, keep_tokens), fs["offsets"].to(device), fr["offsets"].to(device)),
            "sender_kxreceived": map_sender_mask_to_receiver(fixed_topk_mask(combo, keep_tokens), fs["offsets"].to(device), fr["offsets"].to(device)),
            "sender_k_norm_block": block_mask_from_candidate_mask(map_sender_mask_to_receiver(fixed_topk_mask(k_norm, keep_tokens), fs["offsets"].to(device), fr["offsets"].to(device)), args.receiver_block_size, keep_blocks),
            "sender_received_block": block_mask_from_candidate_mask(map_sender_mask_to_receiver(fixed_topk_mask(received, keep_tokens), fs["offsets"].to(device), fr["offsets"].to(device)), args.receiver_block_size, keep_blocks),
            "sender_kxreceived_block": block_mask_from_candidate_mask(map_sender_mask_to_receiver(fixed_topk_mask(combo, keep_tokens), fs["offsets"].to(device), fr["offsets"].to(device)), args.receiver_block_size, keep_blocks),
        }

        for mode, mask in sender_masks.items():
            mask = mask & r_mask[:, None, :, None].bool() & r_mask[:, None, None, :].bool()
            row = evaluate_candidate_mode(mode, mask, attn_r, q_r, k_r, v_r, r_mask, args.route_k)
            mode = row.pop("mode")
            counts[mode] = counts.get(mode, 0) + 1
            for key, value in row.items():
                totals[(mode, key)] = totals.get((mode, key), 0.0) + value

    rows = []
    for mode in sorted(counts):
        row = {"layer": layer, "budget_ratio": ratio, "mode": mode}
        for (m, key), value in totals.items():
            if m == mode:
                row[key] = value / counts[mode]
        rows.append(row)
    return rows


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
    parser.add_argument("--budget_ratios", default="0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--receiver_block_size", type=int, default=32)
    parser.add_argument("--anchor_k", type=int, default=32)
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--csv", default="runs/fixed_budget_fair_compare.csv")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    fs = collect_features(sender, texts, args.max_length, device)
    fr = collect_features(receiver, texts, args.max_length, device)
    ratios = [float(x) for x in args.budget_ratios.split(",") if x.strip()]
    rows = []
    for ratio in ratios:
        print("=" * 80)
        print(f"budget_ratio={ratio}")
        for layer in parse_int_list(args.layers):
            if layer >= sender.model.config.num_hidden_layers or layer >= receiver.model.config.num_hidden_layers:
                continue
            for row in run_layer(fs, fr, layer, args, device, ratio):
                print(f"r={ratio:.1f} L{layer:02d} {row['mode']:<18} sel={row['selected_ratio']:.3f} mass={row['candidate_mass']:.4f} rec_cos={row['selective_recompute_cosine']:.4f}")
                rows.append(row)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = ["budget_ratio", "layer", "mode", "selected_tokens", "selected_ratio", "candidate_mass", "gold_topk_recall", "oracle_out_mse", "oracle_out_cosine", "selective_recompute_mse", "selective_recompute_cosine"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
