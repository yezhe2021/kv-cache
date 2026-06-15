"""Suffix output patch experiment with an explicit train/eval text split."""

import argparse
import csv
import json
import os
from glob import glob
from pathlib import Path

import torch

from attention_output_bootstrap_generation_experiment import (
    continuation_ce,
    greedy_from_prefill,
    greedy_full_receiver,
    patched_prefill,
    seq_similarity,
    token_f1,
)
from block_memory_method_sweep import build_items, parse_method
from evidence_recall_selective_recompute import collect_features, load_bundle
from generation_effect_experiment import train_translators_return_models


def num_layers(model_path):
    with open(Path(model_path) / "config.json", encoding="utf-8") as f:
        return int(json.load(f)["num_hidden_layers"])


def parse_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / max(len(rows), 1)


def record_to_text(record):
    if isinstance(record, dict):
        if {"context", "question"}.issubset(record):
            answer = record.get("answer", "")
            return f"Context:\n{record['context']}\n\nQuestion: {record['question']}\nAnswer: {answer}".strip()
        if "conversations" in record:
            parts = []
            for turn in record["conversations"]:
                speaker = turn.get("from", "speaker")
                value = turn.get("value", "")
                parts.append(f"{speaker}: {value}")
            return "\n".join(parts).strip()
        if "data" in record:
            texts = []
            for item in record["data"]:
                for paragraph in item.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    for qa in paragraph.get("qas", []):
                        texts.append(f"Context:\n{context}\n\nQuestion: {qa.get('question', '')}".strip())
            return texts
    return str(record).strip()


def load_dataset_texts(path_or_glob, limit, max_chars):
    paths = sorted(glob(path_or_glob, recursive=True))
    if not paths:
        path = Path(path_or_glob)
        if path.is_dir():
            paths = sorted(str(p) for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".json", ".jsonl"})
        elif path.exists():
            paths = [str(path)]
    if not paths:
        raise FileNotFoundError(f"No dataset files matched: {path_or_glob}")

    texts = []
    for file_name in paths:
        path = Path(file_name)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                texts.append(text[:max_chars])
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    text = record_to_text(json.loads(line))
                    if isinstance(text, list):
                        texts.extend(x[:max_chars] for x in text if x)
                    elif text:
                        texts.append(text[:max_chars])
                    if len(texts) >= limit:
                        break
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list):
                records = data
            else:
                converted = record_to_text(data)
                records = converted if isinstance(converted, list) else [converted]
            for record in records:
                text = record_to_text(record) if not isinstance(record, str) else record
                if text:
                    texts.append(text[:max_chars])
                if len(texts) >= limit:
                    break
        if len(texts) >= limit:
            break
    if len(texts) < limit:
        raise ValueError(f"Need {limit} texts, loaded {len(texts)} from {path_or_glob}")
    return texts[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver_model", default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_path", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl")
    parser.add_argument("--num_train", type=int, default=8)
    parser.add_argument("--num_eval", type=int, default=8)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--suffix_starts", default="8,12")
    parser.add_argument("--alphas", default="1.0,0.5")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--method", default="multislot_headwise_norm")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/suffix_output_patch_prefill_split.csv")
    parser.add_argument("--summary_csv", default="runs/suffix_output_patch_prefill_split_summary.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_dataset_texts(args.dataset_path, args.num_train + args.num_eval, args.text_max_chars)
    train_texts = texts[: args.num_train]
    eval_texts = texts[args.num_train : args.num_train + args.num_eval]

    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    method = parse_method(args.method, args.slots_per_block)
    layer_count = num_layers(args.receiver_model)
    suffix_starts = parse_ints(args.suffix_starts)
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    train_s = collect_features(sender, train_texts, args.max_length, device)
    train_r = collect_features(receiver, train_texts, args.max_length, device)
    eval_s = collect_features(sender, eval_texts, args.max_length, device)
    eval_r = collect_features(receiver, eval_texts, args.max_length, device)

    enc = receiver.tokenizer(
        eval_texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids_all = enc["input_ids"].to(device)
    mask_all = enc["attention_mask"].to(device)

    rows = []
    for suffix_start in suffix_starts:
        if not 0 <= suffix_start < layer_count:
            raise ValueError(f"suffix_start {suffix_start} outside receiver layer range 0..{layer_count - 1}")
        layers = list(range(suffix_start, layer_count))
        layer_key = ",".join(str(x) for x in layers)
        per_layer = {}
        for layer in layers:
            train_items = build_items(
                train_s,
                train_r,
                layer,
                args.block_size,
                args.block_score_mode,
                args.anchor_tokens,
                args.budget_ratio,
                method,
                args.value_pool_mode,
                device,
            )
            eval_items = build_items(
                eval_s,
                eval_r,
                layer,
                args.block_size,
                args.block_score_mode,
                args.anchor_tokens,
                args.budget_ratio,
                method,
                args.value_pool_mode,
                device,
            )
            tk, tv, prep_input, denorm_pred = train_translators_return_models(
                train_items,
                method,
                args.epochs,
                args.lr,
                args.hidden,
                args.kv_loss_weight,
                args.output_loss_weight,
            )
            per_layer[layer] = {
                "items": eval_items,
                "tk": tk,
                "tv": tv,
                "prep_input": prep_input,
                "denorm_pred": denorm_pred,
                "attn_module": receiver.extractor.attn_layers[layer][1],
            }

        for sample_id in range(len(eval_texts)):
            input_ids = input_ids_all[sample_id : sample_id + 1]
            attention_mask = mask_all[sample_id : sample_id + 1]
            context_len = int(attention_mask.sum().item())
            input_ids = input_ids[:, :context_len]
            attention_mask = attention_mask[:, :context_len]
            full_new = greedy_full_receiver(receiver, input_ids, attention_mask, args.max_new_tokens)
            full_ids = full_new[0].detach().cpu().tolist()
            full_text = receiver.tokenizer.decode(full_ids, skip_special_tokens=True)
            for patch_source in ["none", "translated"]:
                alpha_values = [0.0] if patch_source == "none" else alphas
                for alpha in alpha_values:
                    prefill_out = patched_prefill(receiver, input_ids, attention_mask, per_layer, sample_id, patch_source, alpha, method)
                    boot_new = greedy_from_prefill(receiver, prefill_out, context_len, args.max_new_tokens)
                    boot_ids = boot_new[0].detach().cpu().tolist()
                    boot_text = receiver.tokenizer.decode(boot_ids, skip_special_tokens=True)
                    row = {
                        "sample_id": sample_id,
                        "split": "eval",
                        "num_train": args.num_train,
                        "num_eval": args.num_eval,
                        "suffix_start": suffix_start,
                        "layers": layer_key,
                        "patch_source": patch_source,
                        "alpha": alpha,
                        "context_tokens": context_len,
                        "max_new_tokens": args.max_new_tokens,
                        "first_token_match": int(bool(full_ids and boot_ids and full_ids[0] == boot_ids[0])),
                        "exact_match": int(full_ids == boot_ids),
                        "prefix_match_tokens": next((i for i, (a, b) in enumerate(zip(full_ids, boot_ids)) if a != b), min(len(full_ids), len(boot_ids))),
                        "token_f1": token_f1(boot_ids, full_ids),
                        "sequence_similarity": seq_similarity(boot_ids, full_ids),
                        "full_continuation_ce": float(continuation_ce(receiver, full_new, input_ids, attention_mask).cpu()),
                        "boot_continuation_ce": float(continuation_ce(receiver, boot_new, input_ids, attention_mask).cpu()),
                        "full_text": full_text.replace("\n", "\\n"),
                        "boot_text": boot_text.replace("\n", "\\n"),
                    }
                    rows.append(row)
                    print(
                        f"suffix={suffix_start} sample={sample_id} {patch_source} alpha={alpha:.2f} "
                        f"first={row['first_token_match']} exact={row['exact_match']} "
                        f"f1={row['token_f1']:.3f} sim={row['sequence_similarity']:.3f}"
                    )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "num_train",
        "num_eval",
        "suffix_start",
        "layers",
        "patch_source",
        "alpha",
        "context_tokens",
        "max_new_tokens",
        "first_token_match",
        "exact_match",
        "prefix_match_tokens",
        "token_f1",
        "sequence_similarity",
        "full_continuation_ce",
        "boot_continuation_ce",
        "full_text",
        "boot_text",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["suffix_start"], row["layers"], row["patch_source"], row["alpha"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (suffix_start, layers, patch_source, alpha), vals in grouped.items():
        summary_rows.append(
            {
                "num_train": args.num_train,
                "num_eval": args.num_eval,
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
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "num_train",
            "num_eval",
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
