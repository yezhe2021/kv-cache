import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_preserving_kv_translation_experiment import (
    LOCAL_RECEIVER_MODEL,
    LOCAL_SENDER_MODEL,
    QKVExtractor,
    TextDataset,
    align_heads,
    attention_probs,
    build_toy_texts,
    load_sharegpt_json_texts,
    parse_int_list,
    split_train_eval_texts,
)


LLAMA_3_2_1B = "/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct"


@dataclass
class ModelBundle:
    name: str
    path: str
    tokenizer: object
    model: object
    extractor: QKVExtractor


def load_text_files(pattern: str, limit: int, max_chars: int) -> List[str]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No text files matched: {pattern}")
    texts = []
    for path in paths[:limit]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars).strip()
        if text:
            texts.append(text)
    if not texts:
        raise RuntimeError(f"No non-empty text loaded from: {pattern}")
    return texts


def load_bundle(name: str, path: str, device: torch.device) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    return ModelBundle(name=name, path=path, tokenizer=tokenizer, model=model, extractor=QKVExtractor(model))


def collect_features(bundle: ModelBundle, texts: List[str], max_length: int, device: torch.device):
    rows = []
    loader = torch.utils.data.DataLoader(TextDataset(texts), batch_size=1, shuffle=False)
    for batch_texts in loader:
        batch_texts = list(batch_texts)
        enc = bundle.tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        rows.append(
            {
                "qkv": bundle.extractor.extract(input_ids, mask),
                "mask": mask.cpu(),
                "offsets": enc["offset_mapping"].cpu(),
            }
        )
    return rows


def offset_overlap_matrix(source_offsets: torch.Tensor, target_offsets: torch.Tensor) -> torch.Tensor:
    src = source_offsets[0]
    tgt = target_offsets[0]
    src_start, src_end = src[:, 0], src[:, 1]
    tgt_start, tgt_end = tgt[:, 0], tgt[:, 1]
    src_valid = src_end > src_start
    tgt_valid = tgt_end > tgt_start
    overlap = (tgt_start[:, None] < src_end[None, :]) & (src_start[None, :] < tgt_end[:, None])
    return overlap & tgt_valid[:, None] & src_valid[None, :]


def map_route_mask(source_mask: torch.Tensor, source_offsets: torch.Tensor, target_offsets: torch.Tensor):
    overlap = offset_overlap_matrix(source_offsets, target_offsets).to(source_mask.device)
    return torch.einsum("ts,bhsk,uk->bhtu", overlap.float(), source_mask.float(), overlap.float()) > 0


def map_token_mask(source_mask: torch.Tensor, source_offsets: torch.Tensor, target_offsets: torch.Tensor):
    if source_mask.ndim == 3:
        source_mask = source_mask.any(dim=1)
    overlap = offset_overlap_matrix(source_offsets, target_offsets).to(source_mask.device)
    return torch.einsum("ts,bs->bt", overlap.float(), source_mask.float()) > 0


def topk_mask(attn: torch.Tensor, k: int):
    idx = torch.topk(attn, k=min(k, attn.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(attn, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def global_received_importance(attn: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor):
    qmask = query_mask[:, None, :, None].to(attn.device).float()
    kmask = key_mask[:, None, :].to(attn.device).bool()
    received = (attn * qmask).sum(dim=(1, 2))
    return received.masked_fill(~kmask, torch.finfo(received.dtype).min)


def global_topk_mask(scores: torch.Tensor, k: int):
    idx = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def masked_route_stats(mask_a, mask_b_mapped, query_mask):
    qmask = query_mask[:, None, :, None].to(mask_a.device).bool()
    mask_a = mask_a & qmask
    mask_b_mapped = mask_b_mapped & qmask
    inter = (mask_a & mask_b_mapped).sum(dim=-1).float()
    a_count = mask_a.sum(dim=-1).float().clamp_min(1.0)
    b_count = mask_b_mapped.sum(dim=-1).float().clamp_min(1.0)
    union = (mask_a | mask_b_mapped).sum(dim=-1).float().clamp_min(1.0)
    valid = query_mask[:, None, :].expand_as(inter).to(mask_a.device).float()
    denom = valid.sum().clamp_min(1.0)
    return {
        "route_recall_a_to_b": float(((inter / a_count) * valid).sum().cpu() / denom.cpu()),
        "route_recall_b_to_a": float(((inter / b_count) * valid).sum().cpu() / denom.cpu()),
        "route_jaccard": float(((inter / union) * valid).sum().cpu() / denom.cpu()),
    }


def token_set_stats(mask_a, mask_b_mapped):
    inter = (mask_a & mask_b_mapped).sum(dim=-1).float()
    a_count = mask_a.sum(dim=-1).float().clamp_min(1.0)
    b_count = mask_b_mapped.sum(dim=-1).float().clamp_min(1.0)
    union = (mask_a | mask_b_mapped).sum(dim=-1).float().clamp_min(1.0)
    return {
        "global_recall_a_to_b": float((inter / a_count).mean().cpu()),
        "global_recall_b_to_a": float((inter / b_count).mean().cpu()),
        "global_jaccard": float((inter / union).mean().cpu()),
    }


@torch.no_grad()
def evaluate_layer(features_a, features_b, layer_idx: int, device: torch.device, route_k: int, global_k: int):
    totals = {}
    count = 0

    for fa, fb in zip(features_a, features_b):
        q_a = fa["qkv"].q[layer_idx].to(device)
        k_a = fa["qkv"].k[layer_idx].to(device)
        q_b = fb["qkv"].q[layer_idx].to(device)
        k_b = fb["qkv"].k[layer_idx].to(device)
        mask_a = fa["mask"].to(device)
        mask_b = fb["mask"].to(device)

        target_heads = max(q_a.shape[1], q_b.shape[1])
        q_a = align_heads(q_a, target_heads)
        k_a = align_heads(k_a, target_heads)
        q_b = align_heads(q_b, target_heads)
        k_b = align_heads(k_b, target_heads)

        attn_a = attention_probs(q_a, k_a, mask_a)
        attn_b = attention_probs(q_b, k_b, mask_b)

        route_a = topk_mask(attn_a, route_k)
        route_b = topk_mask(attn_b, route_k)
        route_b_in_a = map_route_mask(route_b, fb["offsets"].to(device), fa["offsets"].to(device))
        route_stats = masked_route_stats(route_a, route_b_in_a, mask_a.bool())

        imp_a = global_received_importance(attn_a, mask_a, mask_a)
        imp_b = global_received_importance(attn_b, mask_b, mask_b)
        global_a = global_topk_mask(imp_a, global_k)
        global_b = global_topk_mask(imp_b, global_k)
        global_b_in_a = map_token_mask(global_b, fb["offsets"].to(device), fa["offsets"].to(device))
        global_stats = token_set_stats(global_a, global_b_in_a)

        for stats in (route_stats, global_stats):
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
        count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


def print_row(row):
    print(
        f"L{row['layer']:02d} "
        f"route_j={row['route_jaccard']:.4f} "
        f"route_Ra2b={row['route_recall_a_to_b']:.4f} "
        f"route_Rb2a={row['route_recall_b_to_a']:.4f} "
        f"global_j={row['global_jaccard']:.4f} "
        f"global_Ra2b={row['global_recall_a_to_b']:.4f} "
        f"global_Rb2a={row['global_recall_b_to_a']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", type=str, default="qwen_sender")
    parser.add_argument("--model_a", type=str, default=LOCAL_SENDER_MODEL)
    parser.add_argument("--model_b_name", type=str, default="llama32_1b")
    parser.add_argument("--model_b", type=str, default=LLAMA_3_2_1B)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_limit", type=int, default=1000)
    parser.add_argument("--text_glob", type=str, default=None)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--eval_ratio", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layers", type=str, default="0,4,8,12,15")
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--global_k", type=int, default=32)
    parser.add_argument("--csv", type=str, default="runs/important_token_overlap.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable, falling back to CPU.")

    if args.text_glob:
        texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    elif args.dataset_path:
        texts = load_sharegpt_json_texts(args.dataset_path, args.dataset_limit)
        texts, _ = split_train_eval_texts(texts, args.eval_ratio)
        texts = texts[: args.num_samples]
    else:
        texts = build_toy_texts(args.num_samples)
    if args.eval_ratio > 0 and not args.dataset_path:
        _, texts = split_train_eval_texts(texts, args.eval_ratio)
    print(f"Texts: {len(texts)}")

    model_a = load_bundle(args.model_a_name, args.model_a, device)
    model_b = load_bundle(args.model_b_name, args.model_b, device)
    print(f"Loaded {model_a.name}: layers={model_a.model.config.num_hidden_layers}")
    print(f"Loaded {model_b.name}: layers={model_b.model.config.num_hidden_layers}")

    features_a = collect_features(model_a, texts, args.max_length, device)
    features_b = collect_features(model_b, texts, args.max_length, device)

    layers = [
        layer
        for layer in parse_int_list(args.layers)
        if layer < model_a.model.config.num_hidden_layers and layer < model_b.model.config.num_hidden_layers
    ]
    rows = []
    for layer in layers:
        row = {
            "model_a": model_a.name,
            "model_b": model_b.name,
            "layer": layer,
            **evaluate_layer(features_a, features_b, layer, device, args.route_k, args.global_k),
        }
        print_row(row)
        rows.append(row)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "model_a",
        "model_b",
        "layer",
        "route_jaccard",
        "route_recall_a_to_b",
        "route_recall_b_to_a",
        "global_jaccard",
        "global_recall_a_to_b",
        "global_recall_b_to_a",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
