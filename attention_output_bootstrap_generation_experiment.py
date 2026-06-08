"""Bootstrap generation by patching attention output during receiver prefill.

Instead of packing translated memory directly as receiver KV cache, this
experiment patches one or more receiver attention outputs during full-context
prefill. Receiver later layers then run normally on the patched hidden
trajectory and produce native past_key_values. Greedy decoding uses that native
cache and is compared with normal full receiver generation.
"""

import argparse
import csv
import os
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

from attention_preserving_kv_translation_experiment import LOCAL_SENDER_MODEL, parse_int_list
from block_memory_method_sweep import build_items, parse_method
from evidence_recall_selective_recompute import LLAMA_3_2_1B, collect_features, load_bundle, load_text_files
from generation_effect_experiment import train_translators_return_models, translated_attention_hidden


@torch.no_grad()
def oracle_attention_hidden(item, attn_module):
    out_heads = item["selective_oracle_out"]
    hidden = out_heads.transpose(1, 2).contiguous().view(out_heads.shape[0], out_heads.shape[2], -1)
    return attn_module.o_proj(hidden.to(attn_module.o_proj.weight.dtype))


def token_f1(a, b):
    a_list = [int(x) for x in a]
    b_list = [int(x) for x in b]
    if not a_list and not b_list:
        return 1.0
    if not a_list or not b_list:
        return 0.0
    used = set()
    overlap = 0
    for x in a_list:
        for i, y in enumerate(b_list):
            if i not in used and x == y:
                used.add(i)
                overlap += 1
                break
    precision = overlap / len(a_list)
    recall = overlap / len(b_list)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def seq_similarity(a, b):
    return SequenceMatcher(a=[int(x) for x in a], b=[int(x) for x in b]).ratio()


@torch.no_grad()
def greedy_full_receiver(receiver, input_ids, attention_mask, max_new_tokens):
    out = receiver.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=receiver.tokenizer.eos_token_id,
    )
    return out[:, input_ids.shape[1] :]


@torch.no_grad()
def greedy_from_prefill(receiver, prefill_out, context_len, max_new_tokens):
    generated = []
    next_id = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_id)
    cache = prefill_out.past_key_values
    cur = next_id
    device = cur.device
    while len(generated) < max_new_tokens:
        cache_len = cache.get_seq_length()
        attention_mask = torch.ones((1, cache_len + 1), dtype=torch.long, device=device)
        position_ids = torch.tensor([[context_len + len(generated) - 1]], dtype=torch.long, device=device)
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


def continuation_ce(receiver, generated_ids, input_ids, attention_mask):
    full = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=1)
    logits = receiver.model(input_ids=full, attention_mask=full_mask, use_cache=False).logits
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    pred = logits[:, start:end, :].contiguous()
    labels = generated_ids.contiguous()
    return F.cross_entropy(pred.view(-1, pred.shape[-1]), labels.view(-1), reduction="mean")


@torch.no_grad()
def patched_prefill(receiver, input_ids, attention_mask, per_layer, sample_id, patch_source, alpha, method):
    handles = []
    replacements = {}
    if patch_source != "none":
        for layer, data in per_layer.items():
            if patch_source == "translated":
                replacements[layer] = translated_attention_hidden(
                    data["items"][sample_id],
                    data["tk"],
                    data["tv"],
                    data["prep_input"],
                    data["denorm_pred"],
                    method,
                    data["attn_module"],
                )
            elif patch_source == "selective_oracle":
                replacements[layer] = oracle_attention_hidden(data["items"][sample_id], data["attn_module"])
            else:
                raise ValueError(f"Unknown patch_source: {patch_source}")

            def make_hook(layer_id):
                def hook(_module, _inputs, output):
                    original = output[0] if isinstance(output, tuple) else output
                    replacement = replacements[layer_id].to(original.dtype)
                    patched = alpha * replacement + (1.0 - alpha) * original
                    if isinstance(output, tuple):
                        return (patched,) + output[1:]
                    return patched

                return hook

            handles.append(data["attn_module"].register_forward_hook(make_hook(layer)))

    out = receiver.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    for handle in handles:
        handle.remove()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_model", default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", default=LLAMA_3_2_1B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_glob", default="/home/yezhe/demo/train/**/*.txt")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--text_max_chars", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--layer_sets", default="12,15;8,12,15")
    parser.add_argument("--patch_sources", default="none,selective_oracle,translated")
    parser.add_argument("--alphas", default="1.0,0.5")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.5)
    parser.add_argument("--block_score_mode", default="anchor_count")
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--slots_per_block", type=int, default=4)
    parser.add_argument("--method", default="multislot_headwise_norm")
    parser.add_argument("--value_pool_mode", default="uniform")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kv_loss_weight", type=float, default=0.1)
    parser.add_argument("--output_loss_weight", type=float, default=1.0)
    parser.add_argument("--csv", default="runs/attention_output_bootstrap_generation.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = load_text_files(args.text_glob, args.num_samples, args.text_max_chars)
    sender = load_bundle("sender", args.sender_model, device)
    receiver = load_bundle("receiver", args.receiver_model, device)
    features_s = collect_features(sender, texts, args.max_length, device)
    features_r = collect_features(receiver, texts, args.max_length, device)
    method = parse_method(args.method, args.slots_per_block)
    layer_sets = [parse_int_list(chunk) for chunk in args.layer_sets.split(";") if chunk.strip()]
    patch_sources = [x.strip() for x in args.patch_sources.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    enc = receiver.tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    input_ids_all = enc["input_ids"].to(device)
    mask_all = enc["attention_mask"].to(device)

    rows = []
    for layers in layer_sets:
        layer_key = ",".join(str(x) for x in layers)
        per_layer = {}
        for layer in layers:
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
            tk, tv, prep_input, denorm_pred = train_translators_return_models(
                items,
                method,
                args.epochs,
                args.lr,
                args.hidden,
                args.kv_loss_weight,
                args.output_loss_weight,
            )
            per_layer[layer] = {
                "items": items,
                "tk": tk,
                "tv": tv,
                "prep_input": prep_input,
                "denorm_pred": denorm_pred,
                "attn_module": receiver.extractor.attn_layers[layer][1],
            }

        for sample_id in range(len(texts)):
            input_ids = input_ids_all[sample_id : sample_id + 1]
            attention_mask = mask_all[sample_id : sample_id + 1]
            context_len = int(attention_mask.sum().item())
            input_ids = input_ids[:, :context_len]
            attention_mask = attention_mask[:, :context_len]
            full_new = greedy_full_receiver(receiver, input_ids, attention_mask, args.max_new_tokens)
            full_ids = full_new[0].detach().cpu().tolist()
            full_text = receiver.tokenizer.decode(full_ids, skip_special_tokens=True)
            for patch_source in patch_sources:
                alpha_values = [0.0] if patch_source == "none" else alphas
                for alpha in alpha_values:
                    prefill_out = patched_prefill(receiver, input_ids, attention_mask, per_layer, sample_id, patch_source, alpha, method)
                    boot_new = greedy_from_prefill(receiver, prefill_out, context_len, args.max_new_tokens)
                    boot_ids = boot_new[0].detach().cpu().tolist()
                    boot_text = receiver.tokenizer.decode(boot_ids, skip_special_tokens=True)
                    row = {
                        "sample_id": sample_id,
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
                        f"L[{layer_key}] sample={sample_id} {patch_source} alpha={alpha:.2f} "
                        f"first={row['first_token_match']} exact={row['exact_match']} "
                        f"f1={row['token_f1']:.3f} sim={row['sequence_similarity']:.3f}"
                    )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id",
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
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
