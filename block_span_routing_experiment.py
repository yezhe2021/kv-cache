"""Block/span-level routing experiment entrypoint.

Runs the existing evidence recall experiment across block sizes and anchor
budgets to test sender K anchors -> receiver evidence blocks.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_sizes", default="16,32,64")
    parser.add_argument("--anchor_ks", default="16,32,64")
    parser.add_argument("--num_samples", default="8")
    parser.add_argument("--max_length", default="512")
    parser.add_argument("--layers", default="0,4,8,12,15")
    args = parser.parse_args()

    for block in [x.strip() for x in args.block_sizes.split(",") if x.strip()]:
        for anchor_k in [x.strip() for x in args.anchor_ks.split(",") if x.strip()]:
            out = f"runs/block_span_routing_b{block}_k{anchor_k}.csv"
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
                anchor_k,
                "--global_k",
                anchor_k,
                "--saliency_k",
                anchor_k,
                "--top_blocks",
                "4",
                "--receiver_block_size",
                block,
                "--csv",
                out,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
