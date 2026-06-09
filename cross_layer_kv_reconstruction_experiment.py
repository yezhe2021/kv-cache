"""Receiver cross-layer KV reconstruction experiment.

Keep only a few receiver anchor-layer KV caches, reconstruct the missing
receiver layers with simple cross-layer rules or learned adapters, then decode
from the reconstructed full-layer cache and compare against receiver full-cache
decode.
"""

import argparse
import csv
import os
from difflib import SequenceMatcher

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from attention_preserving_kv_translation_experiment import parse_int_list
from evidence_recall_selective_recompute import LLAMA_3_2_1B, load_bundle, load_text_files


def token_f1(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    used = set()
    overlap = 0
    for x in a:
        for i, y in enumerate(b):
            if i not in used and x == y:
                used.add(i)
                overlap += 1
                break
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def seq_similarity(a, b):
    return SequenceMatcher(a=[int(x) for x in a], b=[int(x) for x in b]).ratio()


def continuation_ce(receiver, generated_ids, input_ids, attention_mask):
    full = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=1)
    logits = receiver.model(input_ids=full, attention_mask=full_mask, use_cache=False).logits
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    pred = logits[:, start:end, :].contiguous()
    labels = generated_ids.contiguous()
    return F.cross_entropy(pred.view(-1, pred.shape[-1]), labels.view(-1), reduction="mean")


def encode_prompt(tokenizer, text, max_length, device):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def prefix_cache(receiver, input_ids, attention_mask):
    out = receiver.model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True)
    return out.past_key_values


@torch.no_grad()
def greedy_from_cache(receiver, input_ids, cache, max_new_tokens, context_len):
    device = input_ids.device
    generated = []
    cur = input_ids[:, -1:]
    for step in range(max_new_tokens):
        cache_len = cache.get_seq_length()
        attention_mask = torch.ones((1, cache_len + 1), dtype=torch.long, device=device)
        position_ids = torch.tensor([[context_len - 1 + step]], dtype=torch.long, device=device)
        out = receiver.model(
            input_ids=cur,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur)
        cache = out.past_key_values
    return torch.cat(generated, dim=1)


def clone_cache_data(cache):
    return [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in cache.layers]


def nearest_anchor(layer, anchors):
    return min(anchors, key=lambda x: (abs(x - layer), x))


def interp_anchors(layer, anchors):
    lower = [x for x in anchors if x < layer]
    upper = [x for x in anchors if x > layer]
    if not lower or not upper:
        a = nearest_anchor(layer, anchors)
        return a, a, 0.0
    lo = max(lower)
    hi = min(upper)
    w = (layer - lo) / max(hi - lo, 1)
    return lo, hi, w


class KVAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, k, v):
        return self.k(k), self.v(v)


def train_adapters(cache_rows, anchors, epochs, lr):
    num_layers = len(cache_rows[0])
    dim = cache_rows[0][0][0].shape[-1]
    device = cache_rows[0][0][0].device
    adapters = {}
    for layer in range(num_layers):
        if layer in anchors:
            continue
        src = nearest_anchor(layer, anchors)
        model = KVAdapter(dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(epochs):
            total = None
            for row in cache_rows:
                src_k, src_v = row[src]
                tgt_k, tgt_v = row[layer]
                pred_k, pred_v = model(src_k.float(), src_v.float())
                loss = F.mse_loss(pred_k, tgt_k.float()) + F.mse_loss(pred_v, tgt_v.float())
                total = loss if total is None else total + loss
            total = total / len(cache_rows)
            opt.zero_grad()
            total.backward()
            opt.step()
        adapters[layer] = (src, model.eval())
    return adapters


@torch.no_grad()
def reconstruct_cache(cache_data, anchors, method, adapters, receiver_model):
    out = []
    num_layers = len(cache_data)
    for layer in range(num_layers):
        if layer in anchors:
            k, v = cache_data[layer]
        elif method == "copy_nearest":
            k, v = cache_data[nearest_anchor(layer, anchors)]
        elif method == "interp":
            lo, hi, w = interp_anchors(layer, anchors)
            k = (1.0 - w) * cache_data[lo][0].float() + w * cache_data[hi][0].float()
            v = (1.0 - w) * cache_data[lo][1].float() + w * cache_data[hi][1].float()
        elif method == "adapter":
            src, adapter = adapters[layer]
            k, v = adapter(cache_data[src][0].float(), cache_data[src][1].float())
        else:
            raise ValueError(f"Unknown method: {method}")
        out.append((k.to(receiver_model.dtype).contiguous(), v.to(receiver_model.dtype).contiguous()))
    return DynamicCache(ddp_cache_data=out, config=receiver_model.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--anchor_layers", default="8,12,15")
    parser.add_argument("--methods", default="copy_nearest,interp,adapter")
    parser.add_argument("--adapter_epochs", type=int, default=100)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--csv", default="runs/cross_layer_kv_reconstruction.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    receiver = load_bundle("receiver", args.receiver_model, device)
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    num_layers = receiver.model.config.num_hidden_layers
    anchors = sorted({x for x in parse_int_list(args.anchor_layers) if 0 <= x < num_layers})
    if not anchors:
        raise ValueError("No valid anchor layers for this receiver.")
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    prompts = []
    cache_rows = []
    refs = []
    for text in texts:
        input_ids, attention_mask = encode_prompt(receiver.tokenizer, text, args.max_length, device)
        context_len = int(attention_mask.sum().item())
        input_ids = input_ids[:, :context_len]
        attention_mask = attention_mask[:, :context_len]
        cache = prefix_cache(receiver, input_ids, attention_mask)
        cache_data = clone_cache_data(cache)
        ref = greedy_from_cache(receiver, input_ids, DynamicCache(ddp_cache_data=cache_data, config=receiver.model.config), args.max_new_tokens, context_len)
        prompts.append((input_ids, attention_mask, context_len))
        cache_rows.append(cache_data)
        refs.append(ref)

    adapters = train_adapters(cache_rows, anchors, args.adapter_epochs, args.adapter_lr) if "adapter" in methods else {}
    rows = []
    for sample_id, (input_ids, attention_mask, context_len) in enumerate(prompts):
        ref_ids = refs[sample_id][0].detach().cpu().tolist()
        ref_text = receiver.tokenizer.decode(ref_ids, skip_special_tokens=True)
        ref_ce = float(continuation_ce(receiver, refs[sample_id], input_ids, attention_mask).cpu())
        for method in methods:
            rec_cache = reconstruct_cache(cache_rows[sample_id], anchors, method, adapters, receiver.model)
            pred = greedy_from_cache(receiver, input_ids, rec_cache, args.max_new_tokens, context_len)
            pred_ids = pred[0].detach().cpu().tolist()
            pred_text = receiver.tokenizer.decode(pred_ids, skip_special_tokens=True)
            row = {
                "sample_id": sample_id,
                "anchor_layers": ",".join(str(x) for x in anchors),
                "method": method,
                "context_tokens": context_len,
                "cache_tokens": cache_rows[sample_id][0][0].shape[-2],
                "max_new_tokens": args.max_new_tokens,
                "first_token_match": int(bool(ref_ids and pred_ids and ref_ids[0] == pred_ids[0])),
                "exact_match": int(ref_ids == pred_ids),
                "prefix_match_tokens": next((i for i, (a, b) in enumerate(zip(ref_ids, pred_ids)) if a != b), min(len(ref_ids), len(pred_ids))),
                "token_f1": token_f1(pred_ids, ref_ids),
                "sequence_similarity": seq_similarity(pred_ids, ref_ids),
                "reference_continuation_ce": ref_ce,
                "reconstructed_continuation_ce": float(continuation_ce(receiver, pred, input_ids, attention_mask).cpu()),
                "reference_text": ref_text.replace("\n", "\\n"),
                "reconstructed_text": pred_text.replace("\n", "\\n"),
            }
            rows.append(row)
            print(
                f"sample={sample_id} method={method} first={row['first_token_match']} "
                f"exact={row['exact_match']} f1={row['token_f1']:.3f} sim={row['sequence_similarity']:.3f}"
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
        "anchor_layers",
        "method",
        "context_tokens",
        "cache_tokens",
        "max_new_tokens",
        "first_token_match",
        "exact_match",
        "prefix_match_tokens",
        "token_f1",
        "sequence_similarity",
        "reference_continuation_ce",
        "reconstructed_continuation_ce",
        "reference_text",
        "reconstructed_text",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
