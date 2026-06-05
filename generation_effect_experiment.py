"""Generation-proxy evaluation for translated block memory methods.

For each method and layer, train translated block-memory adapters, replace the
receiver attention output at that layer during a full forward pass, and compare
next-token behavior against the unmodified receiver model.

Metrics:
* full_ce: normal receiver next-token cross entropy
* patched_ce: receiver CE with one attention layer replaced by translated output
* ce_delta: patched_ce - full_ce
* logit_kl: KL(full logits || patched logits)
* top1_match: next-token argmax agreement with full receiver
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from block_memory_method_sweep import (
    build_items,
    denormalize,
    make_translator,
    masked_stats,
    normalize,
    parse_method,
    slot_attention,
)
from evidence_recall_selective_recompute import (
    LLAMA_3_2_1B,
    collect_features,
    load_bundle,
    load_text_files,
)


def masked_ce(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].bool()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return (loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp_min(1.0)


def logit_kl_and_match(full_logits, patched_logits, attention_mask):
    full = full_logits[:, :-1, :].float()
    patched = patched_logits[:, :-1, :].float()
    mask = attention_mask[:, 1:].bool()
    full_logp = F.log_softmax(full, dim=-1)
    patched_logp = F.log_softmax(patched, dim=-1)
    full_p = full_logp.exp()
    kl = (full_p * (full_logp - patched_logp)).sum(dim=-1)
    kl = (kl * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    match = full.argmax(dim=-1).eq(patched.argmax(dim=-1))
    match = (match.float() * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return float(kl.cpu()), float(match.cpu())


def train_translators_return_models(items, method, epochs, lr, hidden, kv_loss_weight, output_loss_weight):
    sample = items[0]
    heads = sample["sender_k_slots"].shape[1]
    sender_dim = sample["sender_k_slots"].shape[-1]
    receiver_dim = sample["target_k_slots"].shape[-1]
    device = sample["sender_k_slots"].device
    tk = make_translator(method.translator, heads, sender_dim, receiver_dim, hidden).to(device)
    tv = make_translator(method.translator, heads, sender_dim, receiver_dim, hidden).to(device)
    opt = torch.optim.AdamW(list(tk.parameters()) + list(tv.parameters()), lr=lr)

    stats = {}
    if method.normalize:
        for key in ["sender_k_slots", "sender_v_slots", "target_k_slots", "target_v_slots"]:
            stats[key] = masked_stats(items, "valid_slots", key)

    def prep_input(item, key):
        x = item[key]
        if method.normalize:
            mean, std = stats[key]
            return normalize(x, mean, std)
        return x

    def prep_target(item, key):
        x = item[key]
        if method.normalize:
            mean, std = stats[key]
            return normalize(x, mean, std)
        return x

    def denorm_pred(x, key):
        if method.normalize:
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
                losses.append(kv_loss_weight * F.mse_loss(pred_k_norm.masked_select(valid), prep_target(item, "target_k_slots").masked_select(valid)))
                losses.append(kv_loss_weight * F.mse_loss(pred_v_norm.masked_select(valid), prep_target(item, "target_v_slots").masked_select(valid)))
            if output_loss_weight > 0:
                out = slot_attention(item["q_r"], pred_k, pred_v, item["valid_slots"], item["slot_starts"], item["slot_scores"], item["r_mask"], method.prior_alpha)
                qmask = item["r_mask"].bool()[:, None, :, None].expand_as(out)
                losses.append(output_loss_weight * F.mse_loss(out.masked_select(qmask), item["full_out"].masked_select(qmask)))
            loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return tk, tv, prep_input, denorm_pred


@torch.no_grad()
def translated_attention_hidden(item, tk, tv, prep_input, denorm_pred, method, attn_module):
    pred_k = denorm_pred(tk(prep_input(item, "sender_k_slots")), "target_k_slots")
    pred_v = denorm_pred(tv(prep_input(item, "sender_v_slots")), "target_v_slots")
    out_heads = slot_attention(item["q_r"], pred_k, pred_v, item["valid_slots"], item["slot_starts"], item["slot_scores"], item["r_mask"], method.prior_alpha)
    hidden = out_heads.transpose(1, 2).contiguous().view(out_heads.shape[0], out_heads.shape[2], -1)
    return attn_module.o_proj(hidden.to(attn_module.o_proj.weight.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--layers", default="8,12,15")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--methods", default="baseline_1slot_linear,multislot_mlp,multislot_headwise_norm")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/generation_effect_experiment.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)
    enc = receiver.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids_all = enc["input_ids"].to(device)
    mask_all = enc["attention_mask"].to(device)

    rows = []
    methods = [parse_method(name.strip(), args.slots_per_block) for name in args.methods.split(",") if name.strip()]
    for layer in parse_int_list(args.layers):
        attn_module = receiver.extractor.attn_layers[layer][1]
        for method in methods:
            items = build_items(features_s, features_r, layer, args.block_size, args.block_score_mode, args.anchor_tokens, args.budget_ratio, method, args.value_pool_mode, device)
            tk, tv, prep_input, denorm_pred = train_translators_return_models(items, method, args.epochs, args.lr, args.hidden, args.kv_loss_weight, args.output_loss_weight)
            totals = {"full_ce": 0.0, "patched_ce": 0.0, "logit_kl": 0.0, "top1_match": 0.0}
            for i, item in enumerate(items):
                input_ids = input_ids_all[i : i + 1]
                attention_mask = mask_all[i : i + 1]
                full_logits = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                replacement = translated_attention_hidden(item, tk, tv, prep_input, denorm_pred, method, attn_module)

                def hook(_module, _inputs, output):
                    repl = replacement.to(output[0].dtype if isinstance(output, tuple) else output.dtype)
                    if isinstance(output, tuple):
                        return (repl,) + output[1:]
                    return repl

                handle = attn_module.register_forward_hook(hook)
                patched_logits = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                handle.remove()
                full_ce = masked_ce(full_logits, input_ids, attention_mask)
                patched_ce = masked_ce(patched_logits, input_ids, attention_mask)
                kl, match = logit_kl_and_match(full_logits, patched_logits, attention_mask)
                totals["full_ce"] += float(full_ce.cpu())
                totals["patched_ce"] += float(patched_ce.cpu())
                totals["logit_kl"] += kl
                totals["top1_match"] += match
            n = len(items)
            row = {
                "layer": layer,
                "method": method.name,
                "full_ce": totals["full_ce"] / n,
                "patched_ce": totals["patched_ce"] / n,
                "ce_delta": totals["patched_ce"] / n - totals["full_ce"] / n,
                "logit_kl": totals["logit_kl"] / n,
                "top1_match": totals["top1_match"] / n,
            }
            rows.append(row)
            print(
                f"L{layer:02d} {method.name:<24} "
                f"full_ce={row['full_ce']:.4f} patched_ce={row['patched_ce']:.4f} "
                f"delta={row['ce_delta']:+.4f} kl={row['logit_kl']:.4f} top1={row['top1_match']:.3f}"
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = ["layer", "method", "full_ce", "patched_ce", "ce_delta", "logit_kl", "top1_match"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
