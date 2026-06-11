"""Run attention-output bootstrap generation across three local models."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


MODELS = {
    "qwen06": "/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B",
    "llama1b": "/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct",
    "gemma2b": "/home/yezhe/all_models/models/google/gemma-2-2b",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--layer_sets", default="12,15;8,12,15")
    parser.add_argument("--alphas", default="1.0,0.5")
    parser.add_argument("--patch_sources", default="none,translated")
    parser.add_argument("--out_dir", default="runs/three_model_output_bootstrap")
    parser.add_argument("--summary_csv", default="runs/three_model_output_bootstrap_summary.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sender_name, sender_path in MODELS.items():
        for receiver_name, receiver_path in MODELS.items():
            csv_path = out_dir / f"{sender_name}_to_{receiver_name}.csv"
            cmd = [
                sys.executable,
                "attention_output_bootstrap_generation_experiment.py",
                "--sender_model",
                sender_path,
                "--receiver_model",
                receiver_path,
                "--device",
                args.device,
                "--text_glob",
                "/home/yezhe/demo/train/**/*.txt",
                "--num_samples",
                str(args.num_samples),
                "--max_length",
                str(args.max_length),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--layer_sets",
                args.layer_sets,
                "--patch_sources",
                args.patch_sources,
                "--alphas",
                args.alphas,
                "--block_size",
                "32",
                "--budget_ratio",
                "0.5",
                "--block_score_mode",
                "anchor_count",
                "--anchor_tokens",
                "32",
                "--slots_per_block",
                "4",
                "--method",
                "multislot_headwise_norm",
                "--epochs",
                str(args.epochs),
                "--csv",
                str(csv_path),
            ]
            print(f"RUN {sender_name}->{receiver_name}")
            subprocess.run(cmd, check=True)
            with csv_path.open(newline="", encoding="utf-8") as f:
                pair_rows = list(csv.DictReader(f))
            grouped = {}
            for row in pair_rows:
                key = (row["layers"], row["patch_source"], row["alpha"])
                grouped.setdefault(key, []).append(row)
            for (layers, patch_source, alpha), vals in grouped.items():
                rows.append(
                    {
                        "sender": sender_name,
                        "receiver": receiver_name,
                        "layers": layers,
                        "patch_source": patch_source,
                        "alpha": alpha,
                        "n": len(vals),
                        "first_token_match": sum(float(x["first_token_match"]) for x in vals) / len(vals),
                        "exact_match": sum(float(x["exact_match"]) for x in vals) / len(vals),
                        "token_f1": sum(float(x["token_f1"]) for x in vals) / len(vals),
                        "sequence_similarity": sum(float(x["sequence_similarity"]) for x in vals) / len(vals),
                    }
                )

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sender",
            "receiver",
            "layers",
            "patch_source",
            "alpha",
            "n",
            "first_token_match",
            "exact_match",
            "token_f1",
            "sequence_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
