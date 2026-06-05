"""Output-equivalent region experiment.

Evaluates whether restricted receiver attention output matches full receiver
attention output under different region proposals. This intentionally treats
attention-matrix similarity as secondary.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", default="32")
    parser.add_argument("--budgets", default="16,32,64")
    parser.add_argument("--num_samples", default="8")
    parser.add_argument("--max_length", default="512")
    parser.add_argument("--layers", default="0,4,8,12,15")
    args = parser.parse_args()
    for budget in [x.strip() for x in args.budgets.split(",") if x.strip()]:
        out = f"runs/output_equivalent_region_k{budget}_b{args.block_size}.csv"
        cmd = [
                sys.executable,
            "evidence_recall_selective_recompute.py",
            "--device",
            "cuda",
            "--text_glob",
            "/home/yezhe/demo/train/**/*.txt",
            "--num_samples",
            args.num_samples,
            "--max_length",
            args.max_length,
            "--layers",
            args.layers,
            "--attn_k",
            budget,
            "--global_k",
            budget,
            "--saliency_k",
            budget,
            "--top_blocks",
            "4",
            "--receiver_block_size",
            args.block_size,
            "--csv",
            out,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
