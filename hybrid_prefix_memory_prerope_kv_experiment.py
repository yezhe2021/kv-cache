"""Hybrid prefix-memory pre-RoPE KV experiment.

Sender selected memory blocks are translated as receiver pre-RoPE content KV.
Receiver keeps its own short query prompt, but the query prompt is prefilled
with the memory cache already attached. The final cache layout is:

    [translated sender memory prefix][receiver query prefix]

K position is applied only on the receiver side:

    memory positions: 0..M-1
    query positions:  M..M+Q-1
    decode positions: M+Q...
"""

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from block_memory_method_sweep import parse_method
from evidence_recall_selective_recompute import collect_features, load_bundle
from generation_effect_experiment import train_translators_return_models
from kv_cache_sharing_generation_experiment import (
    build_cache_space_items,
    greedy_full_receiver,
    infer_kv_heads,
    q_heads_to_kv_heads,
    seq_similarity,
    token_f1,
    translated_slots,
)


def format_hotpot_prompt(record):
    return (
        "Answer the question using the context. Give the short answer only.\n\n"
        f"Context:\n{record['context']}\n\n"
        f"Question:\n{record['question']}\n\n"
        "Short answer:"
    )


def format_hotpot_query(record):
    return (
        "Answer the question. Give the short answer only.\n\n"
        f"Question:\n{record['question']}\n\n"
        "Short answer:"
    )


def load_hotpot_records(path, limit):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if {"context", "question"}.issubset(row):
                records.append(row)
            if len(records) >= limit:
                break
    if len(records) < limit:
        raise ValueError(f"Need {limit} HotpotQA records, loaded {len(records)} from {path}")
    return records


def select_records_by_full_length(records, tokenizer, max_tokens, need):
    if max_tokens <= 0:
        return records[:need]
    selected = []
    for record in records:
        if token_count(tokenizer, format_hotpot_prompt(record)) <= max_tokens:
            selected.append(record)
            if len(selected) >= need:
                break
    if len(selected) < need:
        raise ValueError(f"Need {need} records with <= {max_tokens} tokens, selected {len(selected)}.")
    return selected


def encode_prompt(tokenizer, text, max_length, device):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def token_count(tokenizer, text):
    return int(tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].shape[1])


def continuation_ce(receiver, generated_ids, input_ids, attention_mask):
    full = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=1)
    logits = receiver.model(input_ids=full, attention_mask=full_mask, use_cache=False).logits
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    pred = logits[:, start:end, :].contiguous()
    labels = generated_ids.contiguous()
    return F.cross_entropy(pred.view(-1, pred.shape[-1]), labels.view(-1), reduction="mean")


def apply_receiver_rope(receiver_model, k_pre, positions):
    dummy = torch.zeros((k_pre.shape[0], positions.numel(), receiver_model.config.hidden_size), device=k_pre.device, dtype=k_pre.dtype)
    pos = positions[None, :].to(k_pre.device)
    cos, sin = receiver_model.model.rotary_emb(dummy, pos)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    _, k_rot = apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
    return k_rot


def compact_valid_slots(k, v, valid_slots):
    idx = torch.nonzero(valid_slots, as_tuple=False).flatten()
    if idx.numel() == 0:
        idx = torch.arange(k.shape[2], device=k.device)
    return k[:, :, idx, :], v[:, :, idx, :]


@torch.no_grad()
def build_memory_prefix_cache(per_layer, sample_id, receiver_model, max_memory_slots):
    q_heads, kv_heads = infer_kv_heads(receiver_model)
    cache_data = []
    memory_len = None

    for layer in range(receiver_model.config.num_hidden_layers):
        data = per_layer[layer]
        item = data["items"][sample_id]
        k_pre, v = translated_slots(item, data["tk"], data["tv"], data["prep_input"], data["denorm_pred"])
        k_pre, v = compact_valid_slots(k_pre, v, item["valid_slots"])
        if max_memory_slots > 0:
            k_pre = k_pre[:, :, :max_memory_slots, :]
            v = v[:, :, :max_memory_slots, :]
        if memory_len is None:
            memory_len = k_pre.shape[2]
        else:
            k_pre = k_pre[:, :, :memory_len, :]
            v = v[:, :, :memory_len, :]
        positions = torch.arange(k_pre.shape[2], device=k_pre.device)
        k = apply_receiver_rope(receiver_model, k_pre, positions)
        k = q_heads_to_kv_heads(k, q_heads, kv_heads).to(receiver_model.dtype).contiguous()
        v = q_heads_to_kv_heads(v, q_heads, kv_heads).to(receiver_model.dtype).contiguous()
        cache_data.append((k, v))

    return DynamicCache(ddp_cache_data=cache_data, config=receiver_model.config), int(memory_len or 0)


@torch.no_grad()
def build_native_selected_memory_prefix_cache(per_layer, sample_id, receiver_model, max_memory_slots):
    q_heads, kv_heads = infer_kv_heads(receiver_model)
    cache_data = []
    memory_len = None

    for layer in range(receiver_model.config.num_hidden_layers):
        item = per_layer[layer]["items"][sample_id]
        k_pre, v = compact_valid_slots(item["target_k_slots"], item["target_v_slots"], item["valid_slots"])
        if max_memory_slots > 0:
            k_pre = k_pre[:, :, :max_memory_slots, :]
            v = v[:, :, :max_memory_slots, :]
        if memory_len is None:
            memory_len = k_pre.shape[2]
        else:
            k_pre = k_pre[:, :, :memory_len, :]
            v = v[:, :, :memory_len, :]
        positions = torch.arange(k_pre.shape[2], device=k_pre.device)
        k = apply_receiver_rope(receiver_model, k_pre, positions)
        k = q_heads_to_kv_heads(k, q_heads, kv_heads).to(receiver_model.dtype).contiguous()
        v = q_heads_to_kv_heads(v, q_heads, kv_heads).to(receiver_model.dtype).contiguous()
        cache_data.append((k, v))

    return DynamicCache(ddp_cache_data=cache_data, config=receiver_model.config), int(memory_len or 0)


@torch.no_grad()
def prefill_query_with_memory_cache(receiver, query_ids, memory_cache, memory_len):
    device = query_ids.device
    query_len = query_ids.shape[1]
    attention_mask = torch.ones((1, memory_len + query_len), dtype=torch.long, device=device)
    position_ids = torch.arange(memory_len, memory_len + query_len, device=device).unsqueeze(0)
    return receiver.model(
        input_ids=query_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=memory_cache,
        use_cache=True,
    )


@torch.no_grad()
def greedy_from_conditioned_prefill(receiver, prefill_out, memory_len, query_len, max_new_tokens):
    generated = []
    cur = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(cur)
    cache = prefill_out.past_key_values
    device = cur.device

    while len(generated) < max_new_tokens:
        step = len(generated)
        cache_len = cache.get_seq_length()
        attention_mask = torch.ones((1, cache_len + 1), dtype=torch.long, device=device)
        position_ids = torch.tensor([[memory_len + query_len + step - 1]], dtype=torch.long, device=device)
        out = receiver.model(
            input_ids=cur,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_id)
        cur = next_id
        cache = out.past_key_values
    return torch.cat(generated, dim=1)


def answer_contains(text, answer):
    return int(answer.strip().lower() in text.strip().lower())


def selected_answer_recall(record, full_prompt, item, max_memory_slots):
    answer = str(record.get("answer", "")).strip().lower()
    text = full_prompt.lower()
    if not answer or answer not in text:
        return 0
    valid = item["valid_slots"]
    slot_starts = item["slot_starts"]
    chosen = torch.nonzero(valid, as_tuple=False).flatten()
    if max_memory_slots > 0:
        chosen = chosen[:max_memory_slots]
    if chosen.numel() == 0:
        return 0
    block_ids = sorted({int(slot_starts[i].item()) for i in chosen})
    # block_id is represented by its token start. Decode-free approximation:
    # if any selected block token span maps back to text containing the answer.
    offsets = item.get("r_offsets")
    if offsets is None:
        return 0
    for start in block_ids:
        end = min(start + int(item.get("block_size", 32)), offsets.shape[1])
        char_spans = offsets[0, start:end]
        valid_spans = char_spans[char_spans[:, 1] > char_spans[:, 0]]
        if valid_spans.numel() == 0:
            continue
        char_start = int(valid_spans[:, 0].min().item())
        char_end = int(valid_spans[:, 1].max().item())
        if answer in text[char_start:char_end]:
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver_model", default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct")
    parser.add_argument("--dataset_path", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_train", type=int, default=4)
    parser.add_argument("--num_eval", type=int, default=8)
    parser.add_argument("--record_scan_limit", type=int, default=512)
    parser.add_argument("--max_full_tokens_for_records", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--query_max_length", type=int, default=96)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_memory_slots", type=int, default=96)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--method", default="multislot_headwise_norm")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/hybrid_prefix_memory_prerope_kv_hotpotqa.csv")
    parser.add_argument("--summary_csv", default="runs/hybrid_prefix_memory_prerope_kv_hotpotqa_summary.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    method = parse_method(args.method, args.slots_per_block)

    need_records = args.num_train + args.num_eval
    scan_limit = max(args.record_scan_limit, need_records)
    records = load_hotpot_records(args.dataset_path, scan_limit)
    records = select_records_by_full_length(records, receiver.tokenizer, args.max_full_tokens_for_records, need_records)
    train_records = records[: args.num_train]
    eval_records = records[args.num_train : args.num_train + args.num_eval]
    train_full = [format_hotpot_prompt(r) for r in train_records]
    eval_full = [format_hotpot_prompt(r) for r in eval_records]
    eval_query = [format_hotpot_query(r) for r in eval_records]

    train_s = collect_features(sender, train_full, args.max_length, device)
    train_r = collect_features(receiver, train_full, args.max_length, device)
    eval_s = collect_features(sender, eval_full, args.max_length, device)
    eval_r = collect_features(receiver, eval_full, args.max_length, device)

    per_layer = {}
    for layer in range(receiver.model.config.num_hidden_layers):
        train_items = build_cache_space_items(
            train_s,
            train_r,
            None,
            None,
            layer,
            args.block_size,
            args.block_score_mode,
            args.anchor_tokens,
            args.budget_ratio,
            method,
            args.value_pool_mode,
            device,
            "pre_rope",
        )
        eval_items = build_cache_space_items(
            eval_s,
            eval_r,
            None,
            None,
            layer,
            args.block_size,
            args.block_score_mode,
            args.anchor_tokens,
            args.budget_ratio,
            method,
            args.value_pool_mode,
            device,
            "pre_rope",
        )
        for item, feature_row in zip(eval_items, eval_r):
            item["r_offsets"] = feature_row["offsets"].to(device)
            item["block_size"] = args.block_size
        tk, tv, prep_input, denorm_pred = train_translators_return_models(
            train_items,
            method,
            args.epochs,
            args.lr,
            args.hidden,
            args.kv_loss_weight,
            args.output_loss_weight,
        )
        per_layer[layer] = {"items": eval_items, "tk": tk, "tv": tv, "prep_input": prep_input, "denorm_pred": denorm_pred}
        print(f"trained layer {layer:02d}")

    rows = []
    for sample_id, record in enumerate(eval_records):
        full_untruncated_tokens = token_count(receiver.tokenizer, eval_full[sample_id])
        query_untruncated_tokens = token_count(receiver.tokenizer, eval_query[sample_id])
        full_ids, full_mask = encode_prompt(receiver.tokenizer, eval_full[sample_id], args.max_length, device)
        query_ids, query_mask = encode_prompt(receiver.tokenizer, eval_query[sample_id], args.query_max_length, device)
        full_new = greedy_full_receiver(receiver, full_ids, full_mask, args.max_new_tokens)
        query_new = greedy_full_receiver(receiver, query_ids, query_mask, args.max_new_tokens)

        memory_cache, memory_len = build_memory_prefix_cache(per_layer, sample_id, receiver.model, args.max_memory_slots)
        hybrid_prefill = prefill_query_with_memory_cache(receiver, query_ids, memory_cache, memory_len)
        hybrid_new = greedy_from_conditioned_prefill(receiver, hybrid_prefill, memory_len, query_ids.shape[1], args.max_new_tokens)
        native_cache, native_memory_len = build_native_selected_memory_prefix_cache(per_layer, sample_id, receiver.model, args.max_memory_slots)
        native_prefill = prefill_query_with_memory_cache(receiver, query_ids, native_cache, native_memory_len)
        native_new = greedy_from_conditioned_prefill(receiver, native_prefill, native_memory_len, query_ids.shape[1], args.max_new_tokens)

        full_ids_list = full_new[0].detach().cpu().tolist()
        full_text = receiver.tokenizer.decode(full_ids_list, skip_special_tokens=True)
        full_answer_contains = answer_contains(full_text, record.get("answer", ""))
        full_context_truncated = int(full_untruncated_tokens > args.max_length)
        query_truncated = int(query_untruncated_tokens > args.query_max_length)
        answer_recall = selected_answer_recall(record, eval_full[sample_id], per_layer[0]["items"][sample_id], args.max_memory_slots)
        variants = {
            "query_only": query_new,
            "native_receiver_selected_memory_prefix": native_new,
            "query_plus_sender_memory_conditioned_prerope": hybrid_new,
        }
        for name, pred in variants.items():
            pred_ids = pred[0].detach().cpu().tolist()
            pred_text = receiver.tokenizer.decode(pred_ids, skip_special_tokens=True)
            row = {
                "sample_id": sample_id,
                "variant": name,
                "memory_slots": memory_len,
                "native_memory_slots": native_memory_len,
                "selected_answer_recall": answer_recall,
                "query_tokens": int(query_mask.sum().item()),
                "full_tokens": int(full_mask.sum().item()),
                "full_untruncated_tokens": full_untruncated_tokens,
                "query_untruncated_tokens": query_untruncated_tokens,
                "full_context_truncated": full_context_truncated,
                "query_truncated": query_truncated,
                "max_new_tokens": args.max_new_tokens,
                "first_token_match": int(bool(full_ids_list and pred_ids and full_ids_list[0] == pred_ids[0])),
                "exact_match": int(full_ids_list == pred_ids),
                "prefix_match_tokens": next((i for i, (a, b) in enumerate(zip(full_ids_list, pred_ids)) if a != b), min(len(full_ids_list), len(pred_ids))),
                "token_f1": token_f1(pred_ids, full_ids_list),
                "sequence_similarity": seq_similarity(pred_ids, full_ids_list),
                "answer_contains": answer_contains(pred_text, record.get("answer", "")),
                "full_answer_contains": full_answer_contains,
                "full_baseline_failed": int(not bool(full_answer_contains)),
                "full_continuation_ce": float(continuation_ce(receiver, full_new, full_ids, full_mask).cpu()),
                "variant_continuation_ce": float(continuation_ce(receiver, pred, query_ids, query_mask).cpu()),
                "answer": record.get("answer", "").replace("\n", "\\n"),
                "full_text": full_text.replace("\n", "\\n"),
                "variant_text": pred_text.replace("\n", "\\n"),
            }
            rows.append(row)
            print(
                f"sample={sample_id} {name} first={row['first_token_match']} exact={row['exact_match']} "
                f"f1={row['token_f1']:.3f} sim={row['sequence_similarity']:.3f} "
                f"answer={row['answer_contains']} full_answer={row['full_answer_contains']}"
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
        "variant",
        "memory_slots",
        "native_memory_slots",
        "selected_answer_recall",
        "query_tokens",
        "full_tokens",
        "full_untruncated_tokens",
        "query_untruncated_tokens",
        "full_context_truncated",
        "query_truncated",
        "max_new_tokens",
        "first_token_match",
        "exact_match",
        "prefix_match_tokens",
        "token_f1",
        "sequence_similarity",
        "answer_contains",
        "full_answer_contains",
        "full_baseline_failed",
        "full_continuation_ce",
        "variant_continuation_ce",
        "answer",
        "full_text",
        "variant_text",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summary_rows = []
    for variant, vals in grouped.items():
        full_correct = [x for x in vals if float(x["full_answer_contains"]) > 0]
        summary_rows.append(
            {
                "variant": variant,
                "n": len(vals),
                "n_full_correct": len(full_correct),
                "full_baseline_answer_rate": sum(float(x["full_answer_contains"]) for x in vals) / len(vals),
                "full_context_truncation_rate": sum(float(x["full_context_truncated"]) for x in vals) / len(vals),
                "selected_answer_recall": sum(float(x["selected_answer_recall"]) for x in vals) / len(vals),
                "first_token_match": sum(float(x["first_token_match"]) for x in vals) / len(vals),
                "exact_match": sum(float(x["exact_match"]) for x in vals) / len(vals),
                "prefix_match_tokens": sum(float(x["prefix_match_tokens"]) for x in vals) / len(vals),
                "token_f1": sum(float(x["token_f1"]) for x in vals) / len(vals),
                "sequence_similarity": sum(float(x["sequence_similarity"]) for x in vals) / len(vals),
                "answer_contains": sum(float(x["answer_contains"]) for x in vals) / len(vals),
                "answer_contains_when_full_correct": (
                    sum(float(x["answer_contains"]) for x in full_correct) / len(full_correct)
                    if full_correct
                    else 0.0
                ),
            }
        )
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "n",
            "n_full_correct",
            "full_baseline_answer_rate",
            "full_context_truncation_rate",
            "selected_answer_recall",
            "first_token_match",
            "exact_match",
            "prefix_match_tokens",
            "token_f1",
            "sequence_similarity",
            "answer_contains",
            "answer_contains_when_full_correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
