"""Sweep stronger block-memory translation methods.

This script assumes block selection is already handled by sparse routing
anchors. It compares methods for representing and translating selected blocks:

* one mean slot per block vs multiple top-token slots per block
* shared linear vs MLP vs head-wise linear translators
* optional feature standardization
* optional sender routing prior as attention bias

The target is receiver full attention output. The oracle is receiver
token-level selective recompute inside the same selected blocks.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_preserving_kv_translation_experiment import (
    LOCAL_SENDER_MODEL,
    align_heads,
    attention_output,
    attention_probs,
    output_cosine,
    output_mse,
    parse_int_list,
)
from block_translated_memory_recovery import (
    receiver_block_sender_masks,
    score_blocks,
    select_blocks,
    sender_token_weights,
    sender_value_weights,
    selected_block_candidate_mask,
    sparse_normalize,
)
from evidence_recall_selective_recompute import (
    LLAMA_3_2_1B,
    collect_features,
    load_bundle,
    load_text_files,
)


class SharedLinearTranslator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class MLPTranslator(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        z = self.in_norm(x)
        return self.net(z) + self.residual(x)


class HeadWiseLinearTranslator(nn.Module):
    def __init__(self, heads, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.einsum("bhsd,hdo->bhso", x, self.weight) + self.bias[None, :, None, :]


@dataclass
class MethodConfig:
    name: str
    slot_mode: str
    slots_per_block: int
    translator: str
    normalize: bool
    prior_alpha: float


def parse_method(name, default_slots):
    presets = {
        "baseline_1slot_linear": MethodConfig(name, "mean", 1, "linear", False, 0.0),
        "multislot_linear": MethodConfig(name, "top_tokens", default_slots, "linear", False, 0.0),
        "multislot_mlp": MethodConfig(name, "top_tokens", default_slots, "mlp", False, 0.0),
        "multislot_mlp_norm": MethodConfig(name, "top_tokens", default_slots, "mlp", True, 0.0),
        "multislot_headwise_norm": MethodConfig(name, "top_tokens", default_slots, "headwise", True, 0.0),
        "multislot_mlp_norm_prior": MethodConfig(name, "top_tokens", default_slots, "mlp", True, 1.0),
    }
    if name not in presets:
        raise ValueError(f"Unknown method {name}. Available: {sorted(presets)}")
    return presets[name]


def make_translator(kind, heads, in_dim, out_dim, hidden):
    if kind == "linear":
        return SharedLinearTranslator(in_dim, out_dim)
    if kind == "mlp":
        return MLPTranslator(in_dim, out_dim, hidden=hidden)
    if kind == "headwise":
        return HeadWiseLinearTranslator(heads, in_dim, out_dim)
    raise ValueError(f"Unknown translator: {kind}")


def masked_stats(tensors, mask_key, value_key):
    values = []
    for item in tensors:
        valid = item[mask_key][None, None, :, None]
        x = item[value_key].masked_select(valid).view(-1, item[value_key].shape[-1])
        if x.numel() > 0:
            values.append(x)
    x = torch.cat(values, dim=0)
    return x.mean(dim=0), x.std(dim=0).clamp_min(1e-5)


def normalize(x, mean, std):
    return (x - mean.view(*([1] * (x.ndim - 1)), -1)) / std.view(*([1] * (x.ndim - 1)), -1)


def denormalize(x, mean, std):
    return x * std.view(*([1] * (x.ndim - 1)), -1) + mean.view(*([1] * (x.ndim - 1)), -1)


def pool_tokens(x, weights, token_mask):
    weights = weights.masked_fill(~token_mask[None, None, :], 0.0)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (x * weights[..., None]).sum(dim=-2) / denom


def build_slots_for_block(
    k_s,
    v_s,
    routing_weights,
    value_weights,
    block_sender_mask,
    valid_block,
    config,
):
    heads = k_s.shape[1]
    device = k_s.device
    k_slots, v_slots, slot_scores = [], [], []
    if not bool(valid_block.item()) or not bool(block_sender_mask.any().item()):
        zero_k = torch.zeros(1, heads, k_s.shape[-1], device=device, dtype=k_s.dtype)
        zero_v = torch.zeros(1, heads, v_s.shape[-1], device=device, dtype=v_s.dtype)
        for _ in range(config.slots_per_block):
            k_slots.append(zero_k)
            v_slots.append(zero_v)
            slot_scores.append(torch.tensor(0.0, device=device))
        return k_slots, v_slots, slot_scores

    if config.slot_mode == "mean":
        k_slots.append(pool_tokens(k_s, routing_weights, block_sender_mask))
        v_slots.append(pool_tokens(v_s, value_weights, block_sender_mask))
        score = routing_weights.masked_fill(~block_sender_mask[None, None, :], 0.0).sum(dim=-1).mean()
        slot_scores.append(score)
        return k_slots, v_slots, slot_scores

    token_score = routing_weights.mean(dim=1).squeeze(0)
    masked_score = token_score.masked_fill(~block_sender_mask, float("-inf"))
    valid_tokens = int(block_sender_mask.sum().item())
    keep = min(config.slots_per_block, valid_tokens)
    idx = torch.topk(masked_score, k=keep).indices if keep > 0 else torch.empty(0, dtype=torch.long, device=device)
    for slot_id in range(config.slots_per_block):
        if slot_id < keep:
            pos = idx[slot_id]
            k_slots.append(k_s[:, :, pos, :])
            v_slots.append(v_s[:, :, pos, :])
            slot_scores.append(token_score[pos].clamp_min(0.0))
        else:
            k_slots.append(torch.zeros(1, heads, k_s.shape[-1], device=device, dtype=k_s.dtype))
            v_slots.append(torch.zeros(1, heads, v_s.shape[-1], device=device, dtype=v_s.dtype))
            slot_scores.append(torch.tensor(0.0, device=device))
    return k_slots, v_slots, slot_scores


def build_receiver_slot_targets(k_r, v_r, valid_blocks, block_starts, block_ends, slots_per_block):
    k_targets, v_targets = [], []
    for block_id in range(valid_blocks.shape[0]):
        start = int(block_starts[block_id].item())
        end = int(block_ends[block_id].item())
        if bool(valid_blocks[block_id].item()) and end > start:
            idx = torch.linspace(start, end - 1, steps=min(slots_per_block, end - start), device=k_r.device).round().long()
            for slot_id in range(slots_per_block):
                if slot_id < idx.numel():
                    pos = idx[slot_id]
                    k_targets.append(k_r[:, :, pos, :])
                    v_targets.append(v_r[:, :, pos, :])
                else:
                    k_targets.append(torch.zeros_like(k_r[:, :, 0, :]))
                    v_targets.append(torch.zeros_like(v_r[:, :, 0, :]))
        else:
            for _ in range(slots_per_block):
                k_targets.append(torch.zeros_like(k_r[:, :, 0, :]))
                v_targets.append(torch.zeros_like(v_r[:, :, 0, :]))
    return torch.stack(k_targets, dim=2), torch.stack(v_targets, dim=2)


def slot_attention(q_r, k_slots, v_slots, valid_slots, slot_starts, slot_scores, receiver_mask, prior_alpha):
    d = q_r.shape[-1]
    scores = torch.matmul(q_r, k_slots.transpose(-1, -2)) / math.sqrt(d)
    query_pos = torch.arange(q_r.shape[-2], device=q_r.device)
    causal = slot_starts[None, :] <= query_pos[:, None]
    mask = valid_slots[None, None, None, :] & causal[None, None, :, :]
    if prior_alpha:
        prior = slot_scores.clamp_min(1e-8)
        prior = prior / prior.max().clamp_min(1e-8)
        scores = scores + prior_alpha * prior.log()[None, None, None, :]
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v_slots)


def build_items(features_s, features_r, layer, block_size, block_score_mode, anchor_tokens, budget_ratio, config, value_pool_mode, device):
    items = []
    for fs, fr in zip(features_s, features_r):
        q_s = align_heads(fs["qkv"].q[layer].to(device), fr["qkv"].q[layer].shape[1])
        k_s = align_heads(fs["qkv"].k[layer].to(device), fr["qkv"].q[layer].shape[1])
        v_s = align_heads(fs["qkv"].v[layer].to(device), fr["qkv"].q[layer].shape[1])
        q_r = fr["qkv"].q[layer].to(device)
        k_r = fr["qkv"].k[layer].to(device)
        v_r = fr["qkv"].v[layer].to(device)
        s_mask = fs["mask"].to(device)
        r_mask = fr["mask"].to(device)

        attn_s = attention_probs(q_s, k_s, s_mask)
        attn_r = attention_probs(q_r, k_r, r_mask)
        full_out = attention_output(attn_r, v_r)
        routing_weights = sender_token_weights(k_s, attn_s, s_mask, "kxreceived")
        value_weights = sender_value_weights(v_s, attn_s, s_mask, value_pool_mode)
        block_sender_masks, block_starts, block_ends, valid_from_sender = receiver_block_sender_masks(
            fs["offsets"].to(device),
            fr["offsets"].to(device),
            r_mask,
            block_size,
            device,
        )
        valid_recv = torch.tensor([end > start for start, end in zip(block_starts.tolist(), block_ends.tolist())], dtype=torch.bool, device=device)
        candidate_blocks = valid_from_sender & valid_recv
        block_scores = score_blocks(
            routing_weights,
            block_sender_masks,
            candidate_blocks,
            mode=block_score_mode,
            anchor_tokens=anchor_tokens,
        )
        valid_blocks = select_blocks(block_scores, candidate_blocks, keep_blocks=None, budget_ratio=budget_ratio)

        k_slots, v_slots, slot_scores, slot_starts, valid_slots = [], [], [], [], []
        for block_id in range(valid_blocks.shape[0]):
            bk, bv, bs = build_slots_for_block(
                k_s,
                v_s,
                routing_weights,
                value_weights,
                block_sender_masks[block_id],
                valid_blocks[block_id],
                config,
            )
            k_slots.extend(bk)
            v_slots.extend(bv)
            slot_scores.extend(bs)
            for _ in range(config.slots_per_block):
                slot_starts.append(block_starts[block_id])
                valid_slots.append(valid_blocks[block_id])

        sender_k_slots = torch.stack(k_slots, dim=2)
        sender_v_slots = torch.stack(v_slots, dim=2)
        slot_scores = torch.stack(slot_scores).to(device)
        slot_starts = torch.stack(slot_starts).to(device)
        valid_slots = torch.stack(valid_slots).to(device)
        target_k_slots, target_v_slots = build_receiver_slot_targets(k_r, v_r, valid_blocks, block_starts, block_ends, config.slots_per_block)

        candidate_mask = selected_block_candidate_mask(attn_r.shape, valid_blocks, block_starts, block_ends, r_mask, device)
        selective_oracle_out = attention_output(sparse_normalize(attn_r, candidate_mask), v_r)

        items.append(
            {
                "sender_k_slots": sender_k_slots,
                "sender_v_slots": sender_v_slots,
                "target_k_slots": target_k_slots,
                "target_v_slots": target_v_slots,
                "q_r": q_r,
                "r_mask": r_mask,
                "full_out": full_out,
                "selective_oracle_out": selective_oracle_out,
                "valid_slots": valid_slots,
                "slot_starts": slot_starts,
                "slot_scores": slot_scores,
                "selected_ratio": float(valid_blocks.float().mean().cpu()),
            }
        )
    return items


def train_and_eval(items, config, epochs, lr, hidden, kv_loss_weight, output_loss_weight):
    sample = items[0]
    heads = sample["sender_k_slots"].shape[1]
    sender_dim = sample["sender_k_slots"].shape[-1]
    receiver_dim = sample["target_k_slots"].shape[-1]
    device = sample["sender_k_slots"].device

    tk = make_translator(config.translator, heads, sender_dim, receiver_dim, hidden).to(device)
    tv = make_translator(config.translator, heads, sender_dim, receiver_dim, hidden).to(device)
    opt = torch.optim.AdamW(list(tk.parameters()) + list(tv.parameters()), lr=lr)

    stats = {}
    if config.normalize:
        for key in ["sender_k_slots", "sender_v_slots", "target_k_slots", "target_v_slots"]:
            mean, std = masked_stats(items, "valid_slots", key)
            stats[key] = (mean, std)

    def prep_input(item, key):
        x = item[key]
        if config.normalize:
            mean, std = stats[key]
            return normalize(x, mean, std)
        return x

    def prep_target(item, key):
        x = item[key]
        if config.normalize:
            mean, std = stats[key]
            return normalize(x, mean, std)
        return x

    def denorm_pred(x, key):
        if config.normalize:
            mean, std = stats[key]
            return denormalize(x, mean, std)
        return x

    for _ in range(epochs):
        for item in items:
            pred_k_norm = tk(prep_input(item, "sender_k_slots"))
            pred_v_norm = tv(prep_input(item, "sender_v_slots"))
            pred_k = denorm_pred(pred_k_norm, "target_k_slots")
            pred_v = denorm_pred(pred_v_norm, "target_v_slots")
            valid = item["valid_slots"][None, None, :, None]
            losses = []
            if kv_loss_weight > 0:
                target_k = prep_target(item, "target_k_slots") if config.normalize else item["target_k_slots"]
                target_v = prep_target(item, "target_v_slots") if config.normalize else item["target_v_slots"]
                losses.append(kv_loss_weight * F.mse_loss(pred_k_norm.masked_select(valid), target_k.masked_select(valid)))
                losses.append(kv_loss_weight * F.mse_loss(pred_v_norm.masked_select(valid), target_v.masked_select(valid)))
            if output_loss_weight > 0:
                out = slot_attention(
                    item["q_r"],
                    pred_k,
                    pred_v,
                    item["valid_slots"],
                    item["slot_starts"],
                    item["slot_scores"],
                    item["r_mask"],
                    config.prior_alpha,
                )
                qmask = item["r_mask"].bool()[:, None, :, None].expand_as(out)
                losses.append(output_loss_weight * F.mse_loss(out.masked_select(qmask), item["full_out"].masked_select(qmask)))
            loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            opt.step()

    totals = {"translated_mse": 0.0, "translated_cosine": 0.0, "selective_oracle_cosine": 0.0, "selected_ratio": 0.0}
    with torch.no_grad():
        for item in items:
            pred_k = denorm_pred(tk(prep_input(item, "sender_k_slots")), "target_k_slots")
            pred_v = denorm_pred(tv(prep_input(item, "sender_v_slots")), "target_v_slots")
            out = slot_attention(
                item["q_r"],
                pred_k,
                pred_v,
                item["valid_slots"],
                item["slot_starts"],
                item["slot_scores"],
                item["r_mask"],
                config.prior_alpha,
            )
            qmask = item["r_mask"].bool()
            totals["translated_mse"] += float(output_mse(out, item["full_out"], qmask).cpu())
            totals["translated_cosine"] += float(output_cosine(out, item["full_out"], qmask).cpu())
            totals["selective_oracle_cosine"] += float(output_cosine(item["selective_oracle_out"], item["full_out"], qmask).cpu())
            totals["selected_ratio"] += item["selected_ratio"]
    return {k: v / len(items) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layers", default="8,12,15")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--methods", default="baseline_1slot_linear,multislot_linear,multislot_mlp,multislot_mlp_norm,multislot_headwise_norm,multislot_mlp_norm_prior")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/block_memory_method_sweep.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)

    rows = []
    methods = [parse_method(name.strip(), args.slots_per_block) for name in args.methods.split(",") if name.strip()]
    for layer in parse_int_list(args.layers):
        if layer >= sender.model.config.num_hidden_layers or layer >= receiver.model.config.num_hidden_layers:
            continue
        for method in methods:
            items = build_items(
                features_s,
                features_r,
                layer,
                args.block_size,
                args.block_score_mode,
                args.anchor_tokens,
                args.budget_ratio,
                method,
                args.value_pool_mode,
                device,
            )
            metrics = train_and_eval(
                items,
                method,
                args.epochs,
                args.lr,
                args.hidden,
                args.kv_loss_weight,
                args.output_loss_weight,
            )
            row = {
                "layer": layer,
                "method": method.name,
                "slot_mode": method.slot_mode,
                "slots_per_block": method.slots_per_block,
                "translator": method.translator,
                "normalize": method.normalize,
                "prior_alpha": method.prior_alpha,
                "block_score_mode": args.block_score_mode,
                "anchor_tokens": args.anchor_tokens,
                "budget_ratio": args.budget_ratio,
                **metrics,
            }
            rows.append(row)
            print(
                f"L{layer:02d} {method.name:<26} "
                f"ratio={row['selected_ratio']:.3f} "
                f"trans_cos={row['translated_cosine']:.4f} "
                f"oracle_cos={row['selective_oracle_cosine']:.4f} "
                f"mse={row['translated_mse']:.6f}"
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "layer",
        "method",
        "slot_mode",
        "slots_per_block",
        "translator",
        "normalize",
        "prior_alpha",
        "block_score_mode",
        "anchor_tokens",
        "budget_ratio",
        "selected_ratio",
        "translated_mse",
        "translated_cosine",
        "selective_oracle_cosine",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
