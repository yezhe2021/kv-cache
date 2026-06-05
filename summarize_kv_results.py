import argparse
import csv
import glob
import os
from statistics import mean

import torch


METRIC_PAIRS = [
    ("attn_kl_direct", "attn_kl_translated", "KL direct->translated"),
    ("out_mse_direct", "out_mse_translated", "MSE direct->translated"),
    ("attn_kl_zero_kv", "attn_kl_translated", "KL zero_kv->translated"),
    ("out_mse_zero_kv", "out_mse_translated", "MSE zero_kv->translated"),
]

ROUTE_METRICS = [
    "path_sender_topk_overlap",
    "path_sender_topk_mrr",
    "path_sender_attention_mass",
    "candidate_recall_at_32",
    "candidate_recall_at_64",
    "candidate_recall_at_128",
    "route_hat_overlap",
    "route_hat_mrr",
    "route_hat_attention_mass",
    "gold_route_weight_mse",
    "gold_route_weight_cosine",
    "sender_route_weight_mse",
    "sender_route_weight_cosine",
    "route_hat_weight_mse",
    "route_hat_weight_cosine",
    "route_hat_candidate_kl",
    "gold_route_direct_v_mse",
    "gold_route_direct_v_cosine",
    "gold_route_translated_v_mse",
    "gold_route_translated_v_cosine",
    "gold_route_oracle_receiver_v_mse",
    "gold_route_oracle_receiver_v_cosine",
    "sender_route_direct_v_mse",
    "sender_route_direct_v_cosine",
    "sender_route_translated_v_mse",
    "sender_route_translated_v_cosine",
    "sender_route_oracle_receiver_v_mse",
    "sender_route_oracle_receiver_v_cosine",
    "route_hat_direct_v_mse",
    "route_hat_direct_v_cosine",
    "route_hat_translated_v_mse",
    "route_hat_translated_v_cosine",
    "route_hat_oracle_receiver_v_mse",
    "route_hat_oracle_receiver_v_cosine",
    "confidence_entropy",
    "confidence_top2_margin",
]


def load_rows(ckpt_dir):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "layer_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No layer_*.pt files found in: {ckpt_dir}")

    rows = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        metrics = checkpoint.get("metrics")
        if metrics is None:
            raise KeyError(f"Checkpoint has no metrics field: {path}")

        row = {
            "path": path,
            "layer": int(metrics["layer"]),
            "attn_kl_no_reuse": float(metrics.get("attn_kl_no_reuse", 0.0)),
            "out_mse_no_reuse": float(metrics.get("out_mse_no_reuse", 0.0)),
            "attn_kl_zero_kv": float(metrics["attn_kl_zero_kv"]),
            "out_mse_zero_kv": float(metrics["out_mse_zero_kv"]),
            "attn_kl_direct": float(metrics["attn_kl_direct"]),
            "out_mse_direct": float(metrics["out_mse_direct"]),
            "attn_kl_translated": float(metrics["attn_kl_translated"]),
            "out_mse_translated": float(metrics["out_mse_translated"]),
        }
        for key in ROUTE_METRICS:
            if key in metrics:
                row[key] = float(metrics[key])
        row["kl_delta_vs_direct"] = row["attn_kl_translated"] - row["attn_kl_direct"]
        row["mse_delta_vs_direct"] = row["out_mse_translated"] - row["out_mse_direct"]
        row["kl_delta_vs_zero_kv"] = row["attn_kl_translated"] - row["attn_kl_zero_kv"]
        row["mse_delta_vs_zero_kv"] = row["out_mse_translated"] - row["out_mse_zero_kv"]
        row["better_kl_than_direct"] = row["kl_delta_vs_direct"] < 0
        row["better_mse_than_direct"] = row["mse_delta_vs_direct"] < 0
        row["better_both_than_direct"] = row["better_kl_than_direct"] and row["better_mse_than_direct"]
        rows.append(row)

    return sorted(rows, key=lambda x: x["layer"])


def print_table(rows):
    header = (
        "layer  "
        "KL_direct  KL_trans  KL_delta  "
        "MSE_direct  MSE_trans  MSE_delta  better"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        better = []
        if row["better_kl_than_direct"]:
            better.append("KL")
        if row["better_mse_than_direct"]:
            better.append("MSE")
        better_text = ",".join(better) if better else "-"
        print(
            f"{row['layer']:>5}  "
            f"{row['attn_kl_direct']:.6f}  "
            f"{row['attn_kl_translated']:.6f}  "
            f"{row['kl_delta_vs_direct']:+.6f}  "
            f"{row['out_mse_direct']:.6f}  "
            f"{row['out_mse_translated']:.6f}  "
            f"{row['mse_delta_vs_direct']:+.6f}  "
            f"{better_text}"
        )


def print_route_table(rows):
    if "path_sender_topk_overlap" not in rows[0]:
        return

    print()
    header = (
        "layer  sender_overlap  cand_R@32  cand_R@64  cand_R@128  "
        "route_overlap  weight_mse  cand_KL  gold_oracle_mse  sender_oracle_mse  hat_oracle_mse"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['layer']:>5}  "
            f"{row.get('path_sender_topk_overlap', 0.0):.6f}  "
            f"{row.get('candidate_recall_at_32', 0.0):.6f}  "
            f"{row.get('candidate_recall_at_64', 0.0):.6f}  "
            f"{row.get('candidate_recall_at_128', 0.0):.6f}  "
            f"{row.get('route_hat_overlap', 0.0):.6f}  "
            f"{row.get('route_hat_weight_mse', 0.0):.6f}  "
            f"{row.get('route_hat_candidate_kl', 0.0):.6f}  "
            f"{row.get('gold_route_oracle_receiver_v_mse', 0.0):.6f}  "
            f"{row.get('sender_route_oracle_receiver_v_mse', 0.0):.6f}  "
            f"{row.get('route_hat_oracle_receiver_v_mse', 0.0):.6f}"
        )


def print_summary(rows):
    n = len(rows)
    better_kl = [r for r in rows if r["better_kl_than_direct"]]
    better_mse = [r for r in rows if r["better_mse_than_direct"]]
    better_both = [r for r in rows if r["better_both_than_direct"]]

    print()
    print("Summary")
    print("-------")
    print(f"layers: {n}")
    print(f"better KL than direct: {len(better_kl)}/{n} {[r['layer'] for r in better_kl]}")
    print(f"better MSE than direct: {len(better_mse)}/{n} {[r['layer'] for r in better_mse]}")
    print(f"better both than direct: {len(better_both)}/{n} {[r['layer'] for r in better_both]}")
    print(f"mean KL direct: {mean(r['attn_kl_direct'] for r in rows):.6f}")
    print(f"mean KL translated: {mean(r['attn_kl_translated'] for r in rows):.6f}")
    print(f"mean KL delta: {mean(r['kl_delta_vs_direct'] for r in rows):+.6f}")
    print(f"mean MSE direct: {mean(r['out_mse_direct'] for r in rows):.6f}")
    print(f"mean MSE translated: {mean(r['out_mse_translated'] for r in rows):.6f}")
    print(f"mean MSE delta: {mean(r['mse_delta_vs_direct'] for r in rows):+.6f}")

    print()
    print("Interpretation")
    print("--------------")
    print("Negative delta means translated KV is better than direct sender KV reuse.")
    print("Positive delta means translated KV is worse than direct sender KV reuse.")
    print("no_reuse/oracle is 0 by construction and is not a deployable baseline.")


def write_csv(rows, csv_path):
    fieldnames = [
        "layer",
        "attn_kl_no_reuse",
        "attn_kl_zero_kv",
        "attn_kl_direct",
        "attn_kl_translated",
        "kl_delta_vs_direct",
        "kl_delta_vs_zero_kv",
        "out_mse_no_reuse",
        "out_mse_zero_kv",
        "out_mse_direct",
        "out_mse_translated",
        "mse_delta_vs_direct",
        "mse_delta_vs_zero_kv",
        "better_kl_than_direct",
        "better_mse_than_direct",
        "better_both_than_direct",
        "path",
    ]
    for key in ROUTE_METRICS:
        if any(key in row for row in rows) and key not in fieldnames:
            fieldnames.insert(-1, key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_dir",
        nargs="?",
        default="runs/kv_translation_ckpt_openhermes_1000_probe_layers_cuda",
        help="Directory containing layer_*.pt checkpoints.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write a CSV summary.")
    args = parser.parse_args()

    rows = load_rows(args.ckpt_dir)
    print(f"checkpoint_dir: {args.ckpt_dir}")
    print_table(rows)
    print_route_table(rows)
    print_summary(rows)

    if args.csv:
        write_csv(rows, args.csv)
        print()
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
