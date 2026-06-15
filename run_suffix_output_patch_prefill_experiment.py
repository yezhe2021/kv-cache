"""Run prefix-native prefill with suffix-layer attention-output patching.

For each suffix_start, layers before suffix_start run normally during receiver
prefill. Layers from suffix_start to the final receiver layer have their
attention output replaced by translated sender-guided output, then the receiver
continues the forward pass and produces native KV cache for decoding.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def num_layers(model_path):
    with open(Path(model_path) / "config.json", encoding="utf-8") as f:
        return int(json.load(f)["num_hidden_layers"])


def parse_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / max(len(rows), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver_model", default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--suffix_starts", default="8,12")
    parser.add_argument("--alphas", default="1.0,0.5")
    parser.add_argument("--csv", default="runs/suffix_output_patch_prefill.csv")
    parser.add_argument("--summary_csv", default="runs/suffix_output_patch_prefill_summary.csv")
    args = parser.parse_args()

    layer_count = num_layers(args.receiver_model)
    layer_sets = []
    for start in parse_ints(args.suffix_starts):
        if not 0 <= start < layer_count:
            raise ValueError(f"suffix_start {start} outside receiver layer range 0..{layer_count - 1}")
        layer_sets.append(",".join(str(i) for i in range(start, layer_count)))

    cmd = [
        sys.executable,
        "attention_output_bootstrap_generation_experiment.py",
        "--sender_model",
        args.sender_model,
        "--receiver_model",
        args.receiver_model,
        "--device",
        args.device,
        "--num_samples",
        str(args.num_samples),
        "--max_length",
        str(args.max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--layer_sets",
        ";".join(layer_sets),
        "--patch_sources",
        "none,translated",
        "--alphas",
        args.alphas,
        "--epochs",
        str(args.epochs),
        "--csv",
        args.csv,
    ]
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for row in rows:
        first_layer = int(row["layers"].split(",")[0])
        key = (first_layer, row["layers"], row["patch_source"], row["alpha"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (suffix_start, layers, patch_source, alpha), vals in grouped.items():
        summary_rows.append(
            {
                "suffix_start": suffix_start,
                "layers": layers,
                "patch_source": patch_source,
                "alpha": alpha,
                "n": len(vals),
                "first_token_match": mean(vals, "first_token_match"),
                "exact_match": mean(vals, "exact_match"),
                "prefix_match_tokens": mean(vals, "prefix_match_tokens"),
                "token_f1": mean(vals, "token_f1"),
                "sequence_similarity": mean(vals, "sequence_similarity"),
            }
        )

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "suffix_start",
            "layers",
            "patch_source",
            "alpha",
            "n",
            "first_token_match",
            "exact_match",
            "prefix_match_tokens",
            "token_f1",
            "sequence_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
