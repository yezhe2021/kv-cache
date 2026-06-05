import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_preserving_kv_translation_experiment import (
    LOCAL_RECEIVER_MODEL,
    LOCAL_SENDER_MODEL,
    QKVExtractor,
    TextDataset,
    align_heads,
    attention_output,
    attention_probs,
    build_toy_texts,
    candidate_scores_from_sender,
    load_sharegpt_json_texts,
    output_cosine,
    output_mse,
    parse_int_list,
    split_train_eval_texts,
    tokenize_pair,
    topk_attention_indices,
    topk_mask_from_scores,
)


LLAMA_3_2_1B = "/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct"


@dataclass
class LoadedModel:
    name: str
    path: str
    tokenizer: object
    model: object
    extractor: QKVExtractor


def parse_models(text: str) -> Dict[str, str]:
    models = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Model spec must be name=path, got: {item}")
        name, path = item.split("=", 1)
        models[name.strip()] = path.strip()
    return models


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
        raise RuntimeError(f"No non-empty text was loaded from: {pattern}")
    return texts


def load_model(name: str, path: str, device: torch.device) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    extractor = QKVExtractor(model)
    return LoadedModel(name=name, path=path, tokenizer=tokenizer, model=model, extractor=extractor)


def collect_all_model_features(models: Dict[str, LoadedModel], texts: List[str], max_length: int, device: torch.device):
    features = []
    loader = torch.utils.data.DataLoader(TextDataset(texts), batch_size=1, shuffle=False)
    for batch_texts in loader:
        batch_texts = list(batch_texts)
        item = {}
        for name, lm in models.items():
            enc = lm.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            qkv = lm.extractor.extract(input_ids, mask)
            item[name] = {
                "qkv": qkv,
                "mask": mask.cpu(),
                "offsets": enc["offset_mapping"].cpu(),
            }
        features.append(item)
    return features


def align_pair(sender_item, receiver_item, sender_layer: int, receiver_layer: int, device: torch.device):
    q_s = sender_item["qkv"].q[sender_layer].to(device)
    k_s = sender_item["qkv"].k[sender_layer].to(device)
    v_s = sender_item["qkv"].v[sender_layer].to(device)
    q_r = receiver_item["qkv"].q[receiver_layer].to(device)
    k_r = receiver_item["qkv"].k[receiver_layer].to(device)
    v_r = receiver_item["qkv"].v[receiver_layer].to(device)
    s_mask = sender_item["mask"].to(device)
    r_mask = receiver_item["mask"].to(device)

    target_heads = q_r.shape[1]
    q_s = align_heads(q_s, target_heads)
    k_s = align_heads(k_s, target_heads)
    v_s = align_heads(v_s, target_heads)

    return {
        "q_s": q_s,
        "k_s": k_s,
        "v_s": v_s,
        "q_r": q_r,
        "k_r": k_r,
        "v_r": v_r,
        "s_mask": s_mask,
        "r_mask": r_mask,
        "s_offsets": sender_item["offsets"].to(device),
        "r_offsets": receiver_item["offsets"].to(device),
    }


def offset_overlap_matrix(source_offsets: torch.Tensor, target_offsets: torch.Tensor) -> torch.Tensor:
    """Return [target_tokens, source_tokens] overlap mask using tokenizer char spans."""
    src = source_offsets[0]
    tgt = target_offsets[0]
    src_start, src_end = src[:, 0], src[:, 1]
    tgt_start, tgt_end = tgt[:, 0], tgt[:, 1]
    src_valid = src_end > src_start
    tgt_valid = tgt_end > tgt_start
    overlap = (tgt_start[:, None] < src_end[None, :]) & (src_start[None, :] < tgt_end[:, None])
    return overlap & tgt_valid[:, None] & src_valid[None, :]


def map_sender_candidates_to_receiver(
    sender_candidate_mask: torch.Tensor,
    sender_offsets: torch.Tensor,
    receiver_offsets: torch.Tensor,
) -> torch.Tensor:
    """Map [1, heads, sender_q, sender_k] candidates to receiver q/k token space."""
    overlap = offset_overlap_matrix(sender_offsets, receiver_offsets).to(sender_candidate_mask.device)
    # Query: a receiver query uses all sender query rows with overlapping char spans.
    # Key: any sender key candidate maps to all receiver keys whose char spans overlap it.
    mapped = torch.einsum(
        "rs,bhsk,tk->bhrt",
        overlap.float(),
        sender_candidate_mask.float(),
        overlap.float(),
    )
    return mapped > 0


def sparse_normalize(weights: torch.Tensor, mask: torch.Tensor):
    weights = weights.masked_fill(~mask, 0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def masked_mean(values: torch.Tensor, query_mask: torch.Tensor):
    mask = query_mask[:, None, :].expand_as(values).float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_pair_layer(
    features,
    sender_name: str,
    receiver_name: str,
    sender_layer: int,
    receiver_layer: int,
    device: torch.device,
    candidate_budget: int,
    route_k: int,
    block_size: int,
    saliency_weight: float,
    block_weight: float,
    alignment: str,
):
    totals = {}
    count = 0

    def add(key, value):
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    for feature in features:
        t = align_pair(feature[sender_name], feature[receiver_name], sender_layer, receiver_layer, device)
        attn_sender = attention_probs(t["q_s"], t["k_s"], t["s_mask"])
        attn_gold = attention_probs(t["q_r"], t["k_r"], t["r_mask"])
        out_gold = attention_output(attn_gold, t["v_r"])

        cand_scores = candidate_scores_from_sender(
            attn_sender=attn_sender,
            k_sender=t["k_s"],
            key_mask=t["s_mask"],
            block_size=block_size,
            saliency_weight=saliency_weight,
            block_weight=block_weight,
        )
        sender_candidate_mask = topk_mask_from_scores(cand_scores, candidate_budget)
        if alignment == "span":
            candidate_mask = map_sender_candidates_to_receiver(
                sender_candidate_mask=sender_candidate_mask,
                sender_offsets=t["s_offsets"],
                receiver_offsets=t["r_offsets"],
            )
            valid_key_mask = t["r_mask"][:, None, None, :].bool()
            valid_query_mask = t["r_mask"][:, None, :, None].bool()
            candidate_mask = candidate_mask & valid_key_mask & valid_query_mask
        elif alignment == "position":
            seq_len = min(sender_candidate_mask.shape[-1], attn_gold.shape[-1])
            candidate_mask = torch.zeros_like(attn_gold, dtype=torch.bool)
            candidate_mask[:, :, :seq_len, :seq_len] = sender_candidate_mask[:, :, :seq_len, :seq_len]
        else:
            raise ValueError(f"Unknown alignment: {alignment}")
        oracle_attn = sparse_normalize(attn_gold, candidate_mask)
        oracle_out = attention_output(oracle_attn, t["v_r"])

        query_mask = t["r_mask"].bool()
        gold_mass = attn_gold.masked_fill(~candidate_mask, 0.0).sum(dim=-1)
        attn_l1 = (attn_gold - oracle_attn).abs().sum(dim=-1)
        gold_idx = topk_attention_indices(attn_gold, route_k)
        topk_hits = candidate_mask.gather(-1, gold_idx).float().mean(dim=-1)

        add("oracle_mass", masked_mean(gold_mass, query_mask))
        add("oracle_attn_l1", masked_mean(attn_l1, query_mask))
        add("oracle_output_mse", output_mse(oracle_out, out_gold, query_mask))
        add("oracle_output_cosine", output_cosine(oracle_out, out_gold, query_mask))
        add("gold_topk_recall", masked_mean(topk_hits, query_mask))
        count += 1

    row = {
        "sender": sender_name,
        "receiver": receiver_name,
        "sender_layer": sender_layer,
        "receiver_layer": receiver_layer,
    }
    row.update({key: value / max(count, 1) for key, value in totals.items()})
    return row


def layer_pairs(layers: List[int], sender_num_layers: int, receiver_num_layers: int):
    pairs = []
    for layer in layers:
        if layer < sender_num_layers and layer < receiver_num_layers:
            pairs.append((layer, layer))
    return pairs


def print_row(row):
    print(
        f"{row['sender']}->{row['receiver']} "
        f"L{row['sender_layer']:02d}->L{row['receiver_layer']:02d} "
        f"mass={row['oracle_mass']:.4f} "
        f"topk_R={row['gold_topk_recall']:.4f} "
        f"attn_L1={row['oracle_attn_l1']:.4f} "
        f"out_MSE={row['oracle_output_mse']:.6f} "
        f"out_cos={row['oracle_output_cosine']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default=f"qwen_sender={LOCAL_SENDER_MODEL},qwen_receiver={LOCAL_RECEIVER_MODEL},llama32_1b={LLAMA_3_2_1B}",
        help="Comma-separated model specs: name=path,name=path",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_limit", type=int, default=1000)
    parser.add_argument(
        "--text_glob",
        type=str,
        default=None,
        help="Optional glob for plain text files, e.g. '/home/yezhe/demo/train/**/*.txt'.",
    )
    parser.add_argument(
        "--text_max_chars",
        type=int,
        default=20000,
        help="Max characters read from each text file before tokenizer truncation.",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--layers", type=str, default="0,4,8,12,15")
    parser.add_argument("--candidate_budgets", type=str, default="8,16,32,64")
    parser.add_argument("--route_k", type=int, default=8)
    parser.add_argument("--route_block_size", type=int, default=16)
    parser.add_argument("--route_saliency_weight", type=float, default=0.25)
    parser.add_argument("--route_block_weight", type=float, default=0.25)
    parser.add_argument(
        "--alignment",
        type=str,
        choices=["span", "position"],
        default="span",
        help="Map sender candidates to receiver tokens by tokenizer char-span overlap or by raw position index.",
    )
    parser.add_argument("--include_self_pairs", action="store_true")
    parser.add_argument("--csv", type=str, default="runs/cross_model_oracle_candidates.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable, falling back to CPU.")

    model_paths = parse_models(args.models)
    models = {name: load_model(name, path, device) for name, path in model_paths.items()}
    for name, lm in models.items():
        cfg = lm.model.config
        print(
            f"Loaded {name}: layers={cfg.num_hidden_layers}, "
            f"heads={cfg.num_attention_heads}, kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}"
        )

    if args.text_glob:
        texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    elif args.dataset_path:
        texts = load_sharegpt_json_texts(args.dataset_path, args.dataset_limit)
    else:
        texts = build_toy_texts(args.num_samples)
    _, eval_texts = split_train_eval_texts(texts, args.eval_ratio)
    print(f"Eval texts: {len(eval_texts)}")
    features = collect_all_model_features(models, eval_texts, args.max_length, device)

    layers = parse_int_list(args.layers)
    budgets = parse_int_list(args.candidate_budgets)
    rows = []

    for budget in budgets:
        print("=" * 80)
        print(f"candidate_budget={budget}")
        for sender_name, sender in models.items():
            for receiver_name, receiver in models.items():
                if sender_name == receiver_name and not args.include_self_pairs:
                    continue
                for sender_layer, receiver_layer in layer_pairs(
                    layers,
                    sender.model.config.num_hidden_layers,
                    receiver.model.config.num_hidden_layers,
                ):
                    row = evaluate_pair_layer(
                        features=features,
                        sender_name=sender_name,
                        receiver_name=receiver_name,
                        sender_layer=sender_layer,
                        receiver_layer=receiver_layer,
                        device=device,
                        candidate_budget=budget,
                        route_k=args.route_k,
                        block_size=args.route_block_size,
                        saliency_weight=args.route_saliency_weight,
                        block_weight=args.route_block_weight,
                        alignment=args.alignment,
                    )
                    row["candidate_budget"] = budget
                    row["alignment"] = args.alignment
                    print_row(row)
                    rows.append(row)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "candidate_budget",
        "alignment",
        "sender",
        "receiver",
        "sender_layer",
        "receiver_layer",
        "oracle_mass",
        "gold_topk_recall",
        "oracle_attn_l1",
        "oracle_output_mse",
        "oracle_output_cosine",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
