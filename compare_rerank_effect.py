import argparse
import csv
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_preserving_kv_translation_experiment import (
    LOCAL_RECEIVER_MODEL,
    LOCAL_SENDER_MODEL,
    build_toy_texts,
    candidate_level_kl,
    candidate_recall,
    candidate_scores_from_sender,
    collect_qkv_features,
    load_sharegpt_json_texts,
    parse_int_list,
    prepare_route_tensors,
    residual_route_scores,
    route_mrr,
    route_overlap,
    route_weight_cosine,
    route_weight_mse,
    split_train_eval_texts,
    topk_attention_indices,
    topk_mask_from_scores,
    attention_mass_on_route,
    attention_probs,
    train_route_scorer,
)


@torch.no_grad()
def evaluate_mode(
    features,
    layer_idx,
    num_layers,
    route_scorer,
    device,
    route_k,
    candidate_budget,
    block_size,
    saliency_weight,
    block_weight,
    delta_scale,
):
    totals = {}
    count = 0

    def add(name, value):
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

    if route_scorer is not None:
        route_scorer.eval()

    for item in features:
        t = prepare_route_tensors(item, layer_idx, device)
        attn_gold = attention_probs(t["q_r"], t["k_r"], t["r_mask"])
        attn_sender = attention_probs(t["q_s"], t["k_s"], t["s_mask"])

        gold_idx = topk_attention_indices(attn_gold, route_k)
        sender_idx = topk_attention_indices(attn_sender, route_k)

        cand_scores = candidate_scores_from_sender(
            attn_sender=attn_sender,
            k_sender=t["k_s"],
            key_mask=t["r_mask"],
            block_size=block_size,
            saliency_weight=saliency_weight,
            block_weight=block_weight,
        )
        candidate_mask = topk_mask_from_scores(cand_scores, candidate_budget)

        pred_scores = residual_route_scores(
            route_scorer=route_scorer,
            t=t,
            attn_sender=attn_sender,
            layer_idx=layer_idx,
            num_layers=num_layers,
            block_size=block_size,
            delta_scale=delta_scale,
        ).masked_fill(~candidate_mask, torch.finfo(attn_gold.dtype).min)

        route_hat = torch.topk(pred_scores, k=min(route_k, pred_scores.shape[-1]), dim=-1).indices
        attn_hat = torch.softmax(
            pred_scores.masked_fill(
                ~topk_mask_from_scores(pred_scores, route_k),
                torch.finfo(pred_scores.dtype).min,
            ),
            dim=-1,
        )

        query_mask = t["r_mask"].bool()
        add("candidate_recall", candidate_recall(candidate_mask, gold_idx, query_mask))
        add("sender_overlap", route_overlap(sender_idx, gold_idx, query_mask))
        add("route_overlap", route_overlap(route_hat, gold_idx, query_mask))
        add("route_mrr", route_mrr(route_hat, gold_idx, query_mask))
        add("route_mass", attention_mass_on_route(attn_gold, route_hat, query_mask))
        add("weight_mse", route_weight_mse(attn_gold, attn_hat, route_hat, query_mask))
        add("weight_cosine", route_weight_cosine(attn_gold, attn_hat, route_hat, query_mask))
        add("candidate_kl", candidate_level_kl(attn_gold, pred_scores, candidate_mask, query_mask))
        count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


def print_row(row):
    print(
        f"layer={row['layer']:02d} mode={row['mode']:<9} "
        f"cand_R={row['candidate_recall']:.4f} "
        f"overlap={row['route_overlap']:.4f} "
        f"mrr={row['route_mrr']:.4f} "
        f"mass={row['route_mass']:.4f} "
        f"w_mse={row['weight_mse']:.6f} "
        f"w_cos={row['weight_cosine']:.4f} "
        f"cand_KL={row['candidate_kl']:.6f}"
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
    parser.add_argument("--route_block_size", type=int, default=16)
    parser.add_argument("--route_saliency_weight", type=float, default=0.25)
    parser.add_argument("--route_block_weight", type=float, default=0.25)
    parser.add_argument("--route_scorer_epochs", type=int, default=1)
    parser.add_argument("--route_scorer_lr", type=float, default=1e-3)
    parser.add_argument("--route_scorer_hidden", type=int, default=128)
    parser.add_argument("--route_delta_scale", type=float, default=0.1)
    parser.add_argument("--route_topk_bce_alpha", type=float, default=0.1)
    parser.add_argument("--csv", type=str, default="runs/rerank_compare.csv")
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
    train_texts, eval_texts = split_train_eval_texts(texts, args.eval_ratio)
    print(f"Split texts: train={len(train_texts)}, eval={len(eval_texts)}")

    train_features = collect_qkv_features(
        sender_model,
        receiver_model,
        sender_tokenizer,
        receiver_tokenizer,
        train_texts,
        args.max_length,
        args.batch_size,
        device,
    )
    eval_features = collect_qkv_features(
        sender_model,
        receiver_model,
        sender_tokenizer,
        receiver_tokenizer,
        eval_texts,
        args.max_length,
        args.batch_size,
        device,
    )

    num_layers = len(train_features[0]["r_qkv"].q)
    layers = list(range(num_layers)) if args.layers == "all" else parse_int_list(args.layers)
    rows = []

    for layer_idx in layers:
        print("=" * 80)
        print(f"Layer {layer_idx}/{num_layers - 1}")
        no_rerank = evaluate_mode(
            eval_features,
            layer_idx,
            num_layers,
            None,
            device,
            args.route_k,
            args.candidate_budget,
            args.route_block_size,
            args.route_saliency_weight,
            args.route_block_weight,
            args.route_delta_scale,
        )
        no_row = {"layer": layer_idx, "mode": "no_rerank", **no_rerank}
        print_row(no_row)
        rows.append(no_row)

        scorer = train_route_scorer(
            train_features,
            layer_idx=layer_idx,
            num_layers=num_layers,
            candidate_budget=args.candidate_budget,
            route_k=args.route_k,
            route_epochs=args.route_scorer_epochs,
            lr=args.route_scorer_lr,
            device=device,
            block_size=args.route_block_size,
            saliency_weight=args.route_saliency_weight,
            block_weight=args.route_block_weight,
            hidden_dim=args.route_scorer_hidden,
            delta_scale=args.route_delta_scale,
            topk_bce_alpha=args.route_topk_bce_alpha,
        )
        rerank = evaluate_mode(
            eval_features,
            layer_idx,
            num_layers,
            scorer,
            device,
            args.route_k,
            args.candidate_budget,
            args.route_block_size,
            args.route_saliency_weight,
            args.route_block_weight,
            args.route_delta_scale,
        )
        re_row = {"layer": layer_idx, "mode": "rerank", **rerank}
        print_row(re_row)
        print(
            f"delta overlap={re_row['route_overlap'] - no_row['route_overlap']:+.4f} "
            f"delta weight_mse={re_row['weight_mse'] - no_row['weight_mse']:+.6f} "
            f"delta cand_KL={re_row['candidate_kl'] - no_row['candidate_kl']:+.6f}"
        )
        rows.append(re_row)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer",
            "mode",
            "candidate_recall",
            "sender_overlap",
            "route_overlap",
            "route_mrr",
            "route_mass",
            "weight_mse",
            "weight_cosine",
            "candidate_kl",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
