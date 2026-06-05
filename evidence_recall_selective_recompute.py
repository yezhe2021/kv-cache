import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_preserving_kv_translation_experiment import (
    LOCAL_RECEIVER_MODEL,
    LOCAL_SENDER_MODEL,
    QKVExtractor,
    TextDataset,
    align_heads,
    attention_output,
    attention_probs,
    block_token_scores,
    build_toy_texts,
    load_sharegpt_json_texts,
    output_cosine,
    output_mse,
    parse_int_list,
    split_train_eval_texts,
    topk_attention_indices,
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


def map_sender_mask_to_receiver(source_mask: torch.Tensor, source_offsets: torch.Tensor, target_offsets: torch.Tensor):
    overlap = offset_overlap_matrix(source_offsets, target_offsets).to(source_mask.device)
    return torch.einsum("rs,bhsk,tk->bhrt", overlap.float(), source_mask.float(), overlap.float()) > 0


def expand_key_mask_to_receiver_blocks(candidate_mask: torch.Tensor, block_size: int):
    """If any token in a receiver block is selected, keep the whole key block."""
    bsz, heads, query_len, key_len = candidate_mask.shape
    expanded_key = torch.zeros_like(candidate_mask)
    for start in range(0, key_len, block_size):
        end = min(start + block_size, key_len)
        keep = candidate_mask[..., start:end].any(dim=-1, keepdim=True)
        expanded_key[..., start:end] = keep
    return expanded_key


def masked_mean(values: torch.Tensor, query_mask: torch.Tensor):
    mask = query_mask[:, None, :].expand_as(values).float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def sparse_normalize(weights: torch.Tensor, mask: torch.Tensor):
    weights = weights.masked_fill(~mask, 0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def topk_mask_from_scores(scores: torch.Tensor, k: int):
    idx = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def make_global_important_mask(attn_sender: torch.Tensor, s_mask: torch.Tensor, k: int):
    qmask = s_mask[:, None, :, None].to(attn_sender.device).float()
    received = (attn_sender * qmask).sum(dim=(1, 2))
    received = received.masked_fill(~s_mask.to(attn_sender.device).bool(), torch.finfo(received.dtype).min)
    idx = torch.topk(received, k=min(k, received.shape[-1]), dim=-1).indices
    token_mask = torch.zeros_like(received, dtype=torch.bool)
    token_mask.scatter_(-1, idx, True)
    return token_mask[:, None, None, :].expand_as(attn_sender)


def make_k_saliency_mask(k_sender: torch.Tensor, s_mask: torch.Tensor, k: int):
    scores = k_sender.norm(dim=-1).masked_fill(~s_mask[:, None, :].to(k_sender.device).bool(), torch.finfo(k_sender.dtype).min)
    idx = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1).indices
    token_mask = torch.zeros_like(scores, dtype=torch.bool)
    token_mask.scatter_(-1, idx, True)
    return token_mask[:, :, None, :].expand(k_sender.shape[0], k_sender.shape[1], k_sender.shape[2], k_sender.shape[2])


def make_block_saliency_mask(k_sender: torch.Tensor, s_mask: torch.Tensor, block_size: int, top_blocks: int):
    token_scores = k_sender.norm(dim=-1).masked_fill(~s_mask[:, None, :].to(k_sender.device).bool(), 0.0)
    block_scores = block_token_scores(token_scores, block_size)
    block_ids = torch.arange(k_sender.shape[-2], device=k_sender.device) // block_size
    idx = torch.topk(block_scores, k=min(top_blocks * block_size, block_scores.shape[-1]), dim=-1).indices
    rough_mask = torch.zeros_like(block_scores, dtype=torch.bool)
    rough_mask.scatter_(-1, idx, True)
    selected_blocks = torch.zeros_like(block_scores, dtype=torch.bool)
    for block_id in block_ids.unique():
        in_block = block_ids == block_id
        keep = rough_mask[..., in_block].any(dim=-1, keepdim=True)
        selected_blocks[..., in_block] = keep
    return selected_blocks[:, :, None, :].expand(k_sender.shape[0], k_sender.shape[1], k_sender.shape[2], k_sender.shape[2])


def receiver_candidate_scores(q: torch.Tensor, k: torch.Tensor, candidate_mask: torch.Tensor, r_mask: torch.Tensor):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    seq_len = q.shape[-2]
    causal = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    key_mask = r_mask[:, None, None, :].to(q.device).bool()
    scores = scores.masked_fill(causal[None, None, :, :] | ~key_mask | ~candidate_mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


@torch.no_grad()
def evaluate_candidate_mode(mode_name, candidate_mask, attn_gold, q_r, k_r, v_r, r_mask, route_k):
    query_mask = r_mask.bool()
    key_mask = r_mask[:, None, None, :].bool()
    out_gold = attention_output(attn_gold, v_r)
    candidate_mass = attn_gold.masked_fill(~candidate_mask, 0.0).sum(dim=-1)
    gold_idx = topk_attention_indices(attn_gold, route_k)
    topk_recall = candidate_mask.gather(-1, gold_idx).float().mean(dim=-1)
    selected_tokens = (candidate_mask & key_mask).sum(dim=-1).float()
    valid_tokens = key_mask.sum(dim=-1).float().clamp_min(1.0)
    selected_ratio = selected_tokens / valid_tokens

    oracle_attn = sparse_normalize(attn_gold, candidate_mask)
    oracle_out = attention_output(oracle_attn, v_r)

    recompute_attn = receiver_candidate_scores(q_r, k_r, candidate_mask, r_mask)
    recompute_out = attention_output(recompute_attn, v_r)

    return {
        "mode": mode_name,
        "selected_tokens": float(masked_mean(selected_tokens, query_mask).cpu()),
        "selected_ratio": float(masked_mean(selected_ratio, query_mask).cpu()),
        "candidate_mass": float(masked_mean(candidate_mass, query_mask).cpu()),
        "gold_topk_recall": float(masked_mean(topk_recall, query_mask).cpu()),
        "oracle_out_mse": float(output_mse(oracle_out, out_gold, query_mask).cpu()),
        "oracle_out_cosine": float(output_cosine(oracle_out, out_gold, query_mask).cpu()),
        "selective_recompute_mse": float(output_mse(recompute_out, out_gold, query_mask).cpu()),
        "selective_recompute_cosine": float(output_cosine(recompute_out, out_gold, query_mask).cpu()),
    }


def add_row(totals: Dict, counts: Dict, row: Dict):
    mode = row.pop("mode")
    counts[mode] = counts.get(mode, 0) + 1
    for key, value in row.items():
        totals[(mode, key)] = totals.get((mode, key), 0.0) + value


@torch.no_grad()
def evaluate_layer(features_s, features_r, layer_idx, device, args):
    totals, counts = {}, {}
    for fs, fr in zip(features_s, features_r):
        q_s = fs["qkv"].q[layer_idx].to(device)
        k_s = fs["qkv"].k[layer_idx].to(device)
        q_r = fr["qkv"].q[layer_idx].to(device)
        k_r = fr["qkv"].k[layer_idx].to(device)
        v_r = fr["qkv"].v[layer_idx].to(device)
        s_mask = fs["mask"].to(device)
        r_mask = fr["mask"].to(device)

        q_s = align_heads(q_s, q_r.shape[1])
        k_s = align_heads(k_s, q_r.shape[1])
        attn_sender = attention_probs(q_s, k_s, s_mask)
        attn_gold = attention_probs(q_r, k_r, r_mask)

        sender_modes = {
            "attn_topk": topk_mask_from_scores(attn_sender, args.attn_k),
            "global_important": make_global_important_mask(attn_sender, s_mask, args.global_k),
            "k_saliency": make_k_saliency_mask(k_s, s_mask, args.saliency_k),
            "block_saliency": make_block_saliency_mask(k_s, s_mask, args.block_size, args.top_blocks),
        }
        sender_modes["union"] = (
            sender_modes["attn_topk"]
            | sender_modes["global_important"]
            | sender_modes["k_saliency"]
            | sender_modes["block_saliency"]
        )

        for mode_name, sender_mask in sender_modes.items():
            receiver_mask = map_sender_mask_to_receiver(sender_mask, fs["offsets"].to(device), fr["offsets"].to(device))
            receiver_mask = receiver_mask & r_mask[:, None, :, None].bool() & r_mask[:, None, None, :].bool()
            row = evaluate_candidate_mode(mode_name, receiver_mask, attn_gold, q_r, k_r, v_r, r_mask, args.route_k)
            add_row(totals, counts, row)

            block_receiver_mask = expand_key_mask_to_receiver_blocks(receiver_mask, args.receiver_block_size)
            block_receiver_mask = block_receiver_mask & r_mask[:, None, :, None].bool() & r_mask[:, None, None, :].bool()
            row = evaluate_candidate_mode(
                f"{mode_name}_recv_block",
                block_receiver_mask,
                attn_gold,
                q_r,
                k_r,
                v_r,
                r_mask,
                args.route_k,
            )
            add_row(totals, counts, row)

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
        f"L{row['layer']:02d} {row['mode']:<17} "
        f"tokens={row['selected_tokens']:.1f} "
        f"ratio={row['selected_ratio']:.3f} "
        f"mass={row['candidate_mass']:.4f} "
        f"topk_R={row['gold_topk_recall']:.4f} "
        f"oracle_mse={row['oracle_out_mse']:.6f} "
        f"recompute_mse={row['selective_recompute_mse']:.6f} "
        f"recompute_cos={row['selective_recompute_cosine']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_name", type=str, default="qwen_sender")
    parser.add_argument("--sender_model", type=str, default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_name", type=str, default="llama32_1b")
    parser.add_argument("--receiver_model", type=str, default=LLAMA_3_2_1B)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_glob", type=str, default=None)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_limit", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layers", type=str, default="0,4,8,12,15")
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--attn_k", type=int, default=32)
    parser.add_argument("--global_k", type=int, default=32)
    parser.add_argument("--saliency_k", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--receiver_block_size", type=int, default=32)
    parser.add_argument("--top_blocks", type=int, default=4)
    parser.add_argument("--csv", type=str, default="runs/evidence_recall_selective_recompute.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable, falling back to CPU.")

    if args.text_glob:
        texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    elif args.dataset_path:
        texts = load_sharegpt_json_texts(args.dataset_path, args.dataset_limit)[: args.num_samples]
    else:
        texts = build_toy_texts(args.num_samples)
    print(f"Texts: {len(texts)}")

    sender = load_bundle(args.sender_name, args.sender_model, device)
    receiver = load_bundle(args.receiver_name, args.receiver_model, device)
    print(f"Loaded sender {sender.name}: layers={sender.model.config.num_hidden_layers}")
    print(f"Loaded receiver {receiver.name}: layers={receiver.model.config.num_hidden_layers}")

    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)

    layers = [
        layer
        for layer in parse_int_list(args.layers)
        if layer < sender.model.config.num_hidden_layers and layer < receiver.model.config.num_hidden_layers
    ]
    rows = []
    for layer in layers:
        print("=" * 80)
        for row in evaluate_layer(features_s, features_r, layer, device, args):
            row["sender"] = sender.name
            row["receiver"] = receiver.name
            print_row(row)
            rows.append(row)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sender",
        "receiver",
        "layer",
        "mode",
        "selected_tokens",
        "selected_ratio",
        "candidate_mass",
        "gold_topk_recall",
        "oracle_out_mse",
        "oracle_out_cosine",
        "selective_recompute_mse",
        "selective_recompute_cosine",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
