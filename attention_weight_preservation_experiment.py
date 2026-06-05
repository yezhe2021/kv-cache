import argparse
import csv
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_preserving_kv_translation_experiment import (
    LOCAL_RECEIVER_MODEL,
    LOCAL_SENDER_MODEL,
    attention_output,
    attention_probs,
    build_toy_texts,
    candidate_scores_from_sender,
    collect_qkv_features,
    load_sharegpt_json_texts,
    output_cosine,
    output_mse,
    parse_int_list,
    prepare_route_tensors,
    route_weight_cosine,
    route_weight_mse,
    split_train_eval_texts,
    topk_attention_indices,
    topk_mask_from_scores,
)


def sparse_normalize(weights, mask):
    weights = weights.masked_fill(~mask, 0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def candidate_uniform_attention(candidate_mask):
    return sparse_normalize(candidate_mask.float(), candidate_mask)


def candidate_sender_proportional_attention(attn_sender, candidate_mask):
    return sparse_normalize(attn_sender, candidate_mask)


def large_weight_preserving_attention(
    attn_sender,
    candidate_mask,
    large_k,
    residual_mode="uniform",
    saliency=None,
):
    """Preserve sender large-weight ratios, then fill remaining candidate mass.

    The large entries keep both their relative ratios and their sender-side
    total mass inside the candidate set. The remaining candidate mass is spread
    uniformly or by saliency over the non-large candidates.
    """
    large_idx = torch.topk(
        attn_sender.masked_fill(~candidate_mask, torch.finfo(attn_sender.dtype).min),
        k=min(large_k, attn_sender.shape[-1]),
        dim=-1,
    ).indices
    large_mask = torch.zeros_like(candidate_mask)
    large_mask.scatter_(-1, large_idx, True)
    large_mask = large_mask & candidate_mask
    rest_mask = candidate_mask & ~large_mask

    sender_on_candidates = sparse_normalize(attn_sender, candidate_mask)
    large_mass = sender_on_candidates.masked_fill(~large_mask, 0.0).sum(dim=-1, keepdim=True)
    large_dist = sparse_normalize(attn_sender, large_mask)

    if residual_mode == "saliency" and saliency is not None:
        residual_base = saliency[:, :, None, :].expand_as(attn_sender)
    else:
        residual_base = torch.ones_like(attn_sender)
    rest_dist = sparse_normalize(residual_base, rest_mask)
    return large_mass * large_dist + (1.0 - large_mass) * rest_dist


def attention_l1(attn_gold, attn_pred, candidate_mask, query_mask):
    diff = (attn_gold - attn_pred).abs().masked_fill(~candidate_mask, 0.0).sum(dim=-1)
    mask = query_mask[:, None, :].expand_as(diff).float()
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_attention_mode(mode_name, attn_hat, attn_gold, v_receiver, query_mask, route_k):
    out_gold = attention_output(attn_gold, v_receiver)
    out_hat = attention_output(attn_hat, v_receiver)
    gold_idx = topk_attention_indices(attn_gold, route_k)
    return {
        "mode": mode_name,
        "out_mse": float(output_mse(out_hat, out_gold, query_mask).cpu()),
        "out_cosine": float(output_cosine(out_hat, out_gold, query_mask).cpu()),
        "gold_topk_weight_mse": float(route_weight_mse(attn_gold, attn_hat, gold_idx, query_mask).cpu()),
        "gold_topk_weight_cosine": float(route_weight_cosine(attn_gold, attn_hat, gold_idx, query_mask).cpu()),
    }


@torch.no_grad()
def evaluate_layer(
    features,
    layer_idx,
    device,
    route_k,
    candidate_budget,
    block_size,
    saliency_weight,
    block_weight,
    large_k,
):
    totals = {}
    counts = {}

    def add_metrics(metrics):
        mode = metrics.pop("mode")
        counts[mode] = counts.get(mode, 0) + 1
        for key, value in metrics.items():
            totals[(mode, key)] = totals.get((mode, key), 0.0) + value

    for item in features:
        t = prepare_route_tensors(item, layer_idx, device)
        attn_gold = attention_probs(t["q_r"], t["k_r"], t["r_mask"])
        attn_sender = attention_probs(t["q_s"], t["k_s"], t["s_mask"])

        cand_scores = candidate_scores_from_sender(
            attn_sender=attn_sender,
            k_sender=t["k_s"],
            key_mask=t["r_mask"],
            block_size=block_size,
            saliency_weight=saliency_weight,
            block_weight=block_weight,
        )
        candidate_mask = topk_mask_from_scores(cand_scores, candidate_budget)
        query_mask = t["r_mask"].bool()
        key_norm = t["k_s"].norm(dim=-1)
        key_norm = key_norm / key_norm.amax(dim=-1, keepdim=True).clamp_min(1e-8)

        modes = {
            "candidate_uniform": candidate_uniform_attention(candidate_mask),
            "candidate_sender_prop": candidate_sender_proportional_attention(attn_sender, candidate_mask),
            "large_preserve_uniform": large_weight_preserving_attention(
                attn_sender,
                candidate_mask,
                large_k=large_k,
                residual_mode="uniform",
            ),
            "large_preserve_saliency": large_weight_preserving_attention(
                attn_sender,
                candidate_mask,
                large_k=large_k,
                residual_mode="saliency",
                saliency=key_norm,
            ),
            "oracle_gold_on_candidates": sparse_normalize(attn_gold, candidate_mask),
        }

        for mode_name, attn_hat in modes.items():
            metrics = evaluate_attention_mode(
                mode_name,
                attn_hat,
                attn_gold,
                t["v_r"],
                query_mask,
                route_k,
            )
            metrics["candidate_l1"] = float(attention_l1(attn_gold, attn_hat, candidate_mask, query_mask).cpu())
            add_metrics(metrics)

    rows = []
    for mode in sorted(counts):
        row = {"layer": layer_idx, "mode": mode}
        for (m, key), value in totals.items():
            if m == mode:
                row[key] = value / counts[mode]
        rows.append(row)
    return rows


def print_row(row):
    print(
        f"layer={row['layer']:02d} mode={row['mode']:<25} "
        f"out_mse={row['out_mse']:.6f} "
        f"out_cos={row['out_cosine']:.4f} "
        f"topk_w_mse={row['gold_topk_weight_mse']:.6f} "
        f"topk_w_cos={row['gold_topk_weight_cosine']:.4f} "
        f"cand_l1={row['candidate_l1']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", type=str, default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", type=str, default=LOCAL_RECEIVER_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_limit", type=int, default=1000)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,27")
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--candidate_budget", type=int, default=64)
    parser.add_argument("--large_k", type=int, default=8)
    parser.add_argument("--route_block_size", type=int, default=16)
    parser.add_argument("--route_saliency_weight", type=float, default=0.25)
    parser.add_argument("--route_block_weight", type=float, default=0.25)
    parser.add_argument("--csv", type=str, default="runs/attention_weight_preservation.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable, falling back to CPU.")

    sender_tokenizer = AutoTokenizer.from_pretrained(
        args.sender_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    sender_model = AutoModelForCausalLM.from_pretrained(
        args.sender_model,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    receiver_model = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    if sender_tokenizer.pad_token is None:
        sender_tokenizer.pad_token = sender_tokenizer.eos_token
    if receiver_tokenizer.pad_token is None:
        receiver_tokenizer.pad_token = receiver_tokenizer.eos_token

    if args.dataset_path:
        texts = load_sharegpt_json_texts(args.dataset_path, args.dataset_limit)
    else:
        texts = build_toy_texts(args.num_samples)
    _, eval_texts = split_train_eval_texts(texts, args.eval_ratio)
    print(f"Eval texts: {len(eval_texts)}")

    features = collect_qkv_features(
        sender_model,
        receiver_model,
        sender_tokenizer,
        receiver_tokenizer,
        eval_texts,
        args.max_length,
        args.batch_size,
        device,
    )
    num_layers = len(features[0]["r_qkv"].q)
    layers = list(range(num_layers)) if args.layers == "all" else parse_int_list(args.layers)

    rows = []
    for layer_idx in layers:
        print("=" * 80)
        layer_rows = evaluate_layer(
            features,
            layer_idx,
            device,
            args.route_k,
            args.candidate_budget,
            args.route_block_size,
            args.route_saliency_weight,
            args.route_block_weight,
            args.large_k,
        )
        for row in layer_rows:
            print_row(row)
        rows.extend(layer_rows)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "layer",
        "mode",
        "out_mse",
        "out_cosine",
        "gold_topk_weight_mse",
        "gold_topk_weight_cosine",
        "candidate_l1",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
