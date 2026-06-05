"""
Attention-Preserving KV Translation: minimal experiment.

默认使用 /home/yezhe/all_models 中的本地模型目录，不会主动下载模型：
    /home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B
    /home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B

请先把 Hugging Face 模型文件放进对应目录，再运行：
    python3 attention_preserving_kv_translation_experiment.py \
        --device cpu \
        --num_samples 128 \
        --max_length 128 \
        --epochs 5

如果你已经安装 CUDA 版 torch，可以把 --device cpu 改成 --device cuda。
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOCAL_SENDER_MODEL = "/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B"
LOCAL_RECEIVER_MODEL = "/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B"


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def build_toy_texts(num_samples: int) -> List[str]:
    # 这里先用固定的小样本文本做最小实验，目的是验证 KV 映射逻辑是否跑通。
    # 如果要做正式实验，可以把这里替换成真实语料读取逻辑。
    base = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Large language models use attention mechanisms to process context.",
        "A transformer stores key and value caches during autoregressive decoding.",
        "Machine learning models often learn similar latent structures across datasets.",
        "The river bank was covered with grass after the heavy rain.",
        "He went to the bank to deposit money before the meeting.",
        "Retrieval augmented generation combines search results with language generation.",
        "Mathematical reasoning requires models to track intermediate variables carefully.",
        "Python is widely used for deep learning research and engineering.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    return [base[i % len(base)] + f" Sample id: {i}." for i in range(num_samples)]


def format_sharegpt_conversation(example: dict) -> str:
    """Convert one ShareGPT/OpenHermes item into a single plain text sequence."""
    role_names = {
        "system": "System",
        "human": "User",
        "user": "User",
        "gpt": "Assistant",
        "assistant": "Assistant",
    }
    parts = []
    for turn in example.get("conversations", []):
        value = str(turn.get("value", "")).strip()
        if not value:
            continue
        role = role_names.get(str(turn.get("from", "")).lower(), "Turn")
        parts.append(f"{role}: {value}")
    return "\n\n".join(parts)


def load_sharegpt_json_texts(dataset_path: str, limit: int) -> List[str]:
    """Read the first `limit` examples from a large JSON array without loading it all."""
    decoder = json.JSONDecoder()
    texts = []
    buffer = ""

    with open(dataset_path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                raise ValueError(f"Dataset is empty or not a JSON array: {dataset_path}")
            if ch.isspace():
                continue
            if ch != "[":
                raise ValueError(f"Expected a JSON array starting with '[': {dataset_path}")
            break

        while len(texts) < limit:
            chunk = f.read(1024 * 1024)
            if not chunk and not buffer.strip():
                break
            buffer += chunk

            while len(texts) < limit:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith("]"):
                    return texts

                try:
                    example, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break

                text = format_sharegpt_conversation(example)
                if text:
                    texts.append(text)
                buffer = buffer[end:]

            if not chunk:
                break

    return texts


def split_train_eval_texts(texts: List[str], eval_ratio: float):
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("--eval_ratio must be in [0.0, 1.0).")
    if len(texts) == 0:
        raise ValueError("No texts to split.")
    if len(texts) == 1 or eval_ratio == 0.0:
        return texts, texts

    eval_size = max(1, int(round(len(texts) * eval_ratio)))
    eval_size = min(eval_size, len(texts) - 1)
    split_idx = len(texts) - eval_size
    return texts[:split_idx], texts[split_idx:]


@dataclass
class QKVCache:
    # 每个 list 的长度等于模型 attention 层数。
    # 每个 tensor 形状约为 [batch, heads, seq_len, head_dim]。
    q: List[torch.Tensor]
    k: List[torch.Tensor]
    v: List[torch.Tensor]


class QKVExtractor:
    """Extract Q/K/V from common Hugging Face decoder-only models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = getattr(model, "config", None)
        self.handles = []
        self.q_raw = []
        self.k_raw = []
        self.v_raw = []
        self.attn_layers = self._find_attention_layers()

        if len(self.attn_layers) == 0:
            raise RuntimeError("没有找到 attention layers，请检查模型结构并修改 _find_attention_layers。")

    def _find_attention_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            # Qwen/LLaMA 类 decoder-only 模型的 attention 模块通常都有这三个投影层。
            # 如果换成别的模型结构，这里可能需要适配对应的模块命名。
            if all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "v_proj"]):
                layers.append((name, module))
        return layers

    def _clear(self):
        self.q_raw = []
        self.k_raw = []
        self.v_raw = []

    def _register_hooks(self):
        self.handles = []

        for _, attn in self.attn_layers:
            # forward hook 会在模型前向传播时截获 q/k/v 投影层的输出。
            # detach() 表示这里只收集特征，不让梯度回传到原始大模型。
            self.handles.append(
                attn.q_proj.register_forward_hook(
                    lambda m, inp, out: self.q_raw.append(out.detach())
                )
            )
            self.handles.append(
                attn.k_proj.register_forward_hook(
                    lambda m, inp, out: self.k_raw.append(out.detach())
                )
            )
            self.handles.append(
                attn.v_proj.register_forward_hook(
                    lambda m, inp, out: self.v_raw.append(out.detach())
                )
            )

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @staticmethod
    def _reshape_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # HF 模型投影层输出通常是 [batch, seq_len, hidden]。
        # 注意力计算需要拆成多头格式 [batch, heads, seq_len, head_dim]。
        b, s, hidden = x.shape
        head_dim = hidden // num_heads
        return x.view(b, s, num_heads, head_dim).transpose(1, 2).contiguous()

    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QKVCache:
        self._clear()
        self._register_hooks()

        with torch.no_grad():
            # use_cache=False 是为了让模型完整走 attention 前向，方便 hook 收集每层 Q/K/V。
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        self._remove_hooks()

        q_list, k_list, v_list = [], [], []

        for i, (_, attn) in enumerate(self.attn_layers):
            # Qwen3Attention 不在 attention module 上暴露 num_heads，
            # 但会在 model.config 里保存 num_attention_heads / num_key_value_heads。
            num_q_heads = (
                getattr(attn, "num_heads", None)
                or getattr(attn, "num_attention_heads", None)
                or getattr(self.config, "num_attention_heads", None)
            )

            num_kv_heads = (
                getattr(attn, "num_key_value_heads", None)
                or getattr(self.config, "num_key_value_heads", None)
                or num_q_heads
            )

            if num_q_heads is None:
                raise RuntimeError("无法读取 num_heads，请手动适配 attention module。")

            q = self._reshape_to_heads(self.q_raw[i], num_q_heads)
            k = self._reshape_to_heads(self.k_raw[i], num_kv_heads)
            v = self._reshape_to_heads(self.v_raw[i], num_kv_heads)

            if k.shape[1] != q.shape[1]:
                # GQA/MQA 模型里 KV head 数可能少于 Q head 数。
                # repeat 到相同 head 数后，后面的注意力计算可以统一处理。
                repeat_factor = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            # 放到 CPU 可以降低显存占用；训练每层 translator 时再搬回 device。
            q_list.append(q.float().cpu())
            k_list.append(k.float().cpu())
            v_list.append(v.float().cpu())

        return QKVCache(q=q_list, k=k_list, v=v_list)


class LinearKVTranslator(nn.Module):
    # 最简单的映射：每个 token/head 的 K 或 V 向量独立经过一个线性层。
    def __init__(self, sender_dim: int, receiver_dim: int):
        super().__init__()
        self.proj = nn.Linear(sender_dim, receiver_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LowRankKVTranslator(nn.Module):
    # 低秩映射：先降维再升维，参数更少，但表达能力弱一些。
    def __init__(self, sender_dim: int, receiver_dim: int, rank: int = 32):
        super().__init__()
        rank = min(rank, sender_dim, receiver_dim)
        self.down = nn.Linear(sender_dim, rank, bias=False)
        self.up = nn.Linear(rank, receiver_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def causal_mask(seq_len: int, device: torch.device):
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1,
    ).bool()


def attention_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    # 用 receiver 的 Q 和某组 K 重新计算注意力图。
    # 训练目标就是让 sender K 映射后的注意力图接近 receiver 原始注意力图。
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)

    s = q.shape[-2]
    scores = scores.masked_fill(
        causal_mask(s, scores.device)[None, None, :, :],
        torch.finfo(scores.dtype).min,
    )

    if attention_mask is not None:
        key_mask = attention_mask[:, None, None, :].to(
            dtype=torch.bool,
            device=scores.device,
        )
        scores = scores.masked_fill(
            ~key_mask,
            torch.finfo(scores.dtype).min,
        )

    return F.softmax(scores, dim=-1)


def attention_kl_loss(
    gold_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # KL(gold || pred)：gold 是 receiver 原始注意力分布，pred 是替代 K 得到的注意力分布。
    gold = gold_probs.clamp_min(eps)
    pred = pred_probs.clamp_min(eps)
    return (gold * (gold.log() - pred.log())).sum(dim=-1).mean()


def attention_output(attn: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(attn, v)


def align_heads(x: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Repeat or trim heads so sender/receiver route metrics are comparable."""
    cur_heads = x.shape[1]
    if cur_heads == target_heads:
        return x
    if cur_heads < target_heads and target_heads % cur_heads == 0:
        return x.repeat_interleave(target_heads // cur_heads, dim=1)
    if cur_heads > target_heads:
        return x[:, :target_heads]
    raise ValueError(f"Cannot align {cur_heads} heads to {target_heads} heads.")


def masked_metric_mean(values: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    mask = query_mask[:, None, :].expand_as(values).to(values.device)
    denom = mask.float().sum().clamp_min(1.0)
    return (values * mask.float()).sum() / denom


def topk_attention_indices(attn: torch.Tensor, route_k: int) -> torch.Tensor:
    k = min(route_k, attn.shape[-1])
    return torch.topk(attn, k=k, dim=-1).indices


def route_overlap(pred_idx: torch.Tensor, gold_idx: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    pred_mask = torch.zeros(
        *pred_idx.shape[:-1],
        gold_idx.shape[-2],
        device=pred_idx.device,
        dtype=torch.bool,
    )
    pred_mask.scatter_(-1, pred_idx, True)
    hits = pred_mask.gather(-1, gold_idx).float().sum(dim=-1)
    overlap = hits / gold_idx.shape[-1]
    return masked_metric_mean(overlap, query_mask)


def route_mrr(pred_idx: torch.Tensor, gold_idx: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    eq = pred_idx.unsqueeze(-1).eq(gold_idx.unsqueeze(-2))
    ranks = torch.arange(
        1,
        pred_idx.shape[-1] + 1,
        device=pred_idx.device,
        dtype=torch.float32,
    ).view(*([1] * (eq.ndim - 2)), -1, 1)
    reciprocal = torch.where(eq, 1.0 / ranks, torch.zeros_like(ranks))
    best = reciprocal.max(dim=-2).values.mean(dim=-1)
    return masked_metric_mean(best, query_mask)


def attention_mass_on_route(attn_gold: torch.Tensor, route_idx: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    mass = attn_gold.gather(-1, route_idx).sum(dim=-1)
    return masked_metric_mean(mass, query_mask)


def route_weight_mse(
    attn_gold: torch.Tensor,
    attn_pred: torch.Tensor,
    route_idx: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    gold_w = attn_gold.gather(-1, route_idx)
    pred_w = attn_pred.gather(-1, route_idx)
    gold_w = gold_w / gold_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pred_w = pred_w / pred_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mse = (gold_w - pred_w).pow(2).mean(dim=-1)
    return masked_metric_mean(mse, query_mask)


def route_weight_cosine(
    attn_gold: torch.Tensor,
    attn_pred: torch.Tensor,
    route_idx: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    gold_w = attn_gold.gather(-1, route_idx)
    pred_w = attn_pred.gather(-1, route_idx)
    gold_w = gold_w / gold_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pred_w = pred_w / pred_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = F.cosine_similarity(gold_w, pred_w, dim=-1)
    return masked_metric_mean(cos, query_mask)


def candidate_level_kl(
    attn_gold: torch.Tensor,
    pred_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    gold_on_candidates = attn_gold.masked_fill(~candidate_mask, 0.0)
    gold_dist = gold_on_candidates / gold_on_candidates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_pred = F.log_softmax(
        pred_scores.masked_fill(~candidate_mask, torch.finfo(pred_scores.dtype).min),
        dim=-1,
    )
    kl = F.kl_div(log_pred, gold_dist, reduction="none").sum(dim=-1)
    return masked_metric_mean(kl, query_mask)


def topk_mask_from_scores(scores: torch.Tensor, budget: int) -> torch.Tensor:
    k = min(budget, scores.shape[-1])
    idx = torch.topk(scores, k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def normalize_token_scores(scores: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    mask = key_mask[:, None, :].to(scores.device).bool()
    safe = scores.masked_fill(~mask, 0.0)
    denom = safe.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return safe / denom


def block_token_scores(token_scores: torch.Tensor, block_size: int) -> torch.Tensor:
    bsz, heads, seq_len = token_scores.shape
    out = torch.zeros_like(token_scores)
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        block_mean = token_scores[:, :, start:end].mean(dim=-1, keepdim=True)
        out[:, :, start:end] = block_mean
    return out


def candidate_scores_from_sender(
    attn_sender: torch.Tensor,
    k_sender: torch.Tensor,
    key_mask: torch.Tensor,
    block_size: int,
    saliency_weight: float,
    block_weight: float,
) -> torch.Tensor:
    token_saliency = normalize_token_scores(k_sender.norm(dim=-1), key_mask)
    block_saliency = block_token_scores(token_saliency, block_size)
    scores = (
        attn_sender
        + saliency_weight * token_saliency[:, :, None, :]
        + block_weight * block_saliency[:, :, None, :]
    )
    seq_len = scores.shape[-1]
    invalid = causal_mask(seq_len, scores.device)[None, None, :, :]
    key_mask_bool = key_mask[:, None, None, :].to(scores.device).bool()
    scores = scores.masked_fill(invalid | ~key_mask_bool, torch.finfo(scores.dtype).min)
    return scores


def candidate_recall(
    candidate_mask: torch.Tensor,
    gold_idx: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    hits = candidate_mask.gather(-1, gold_idx).float().sum(dim=-1)
    recall = hits / gold_idx.shape[-1]
    return masked_metric_mean(recall, query_mask)


def masked_softmax_from_indices(scores: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, route_idx, True)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return F.softmax(masked_scores, dim=-1)


def normalized_attention_on_route(attn: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
    route_mass = attn.gather(-1, route_idx).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    route_weights = attn.gather(-1, route_idx) / route_mass
    sparse = torch.zeros_like(attn)
    sparse.scatter_add_(-1, route_idx, route_weights)
    return sparse


def output_cosine(out_pred: torch.Tensor, out_gold: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(out_pred, out_gold, dim=-1)
    return masked_metric_mean(cos, query_mask)


def output_mse(out_pred: torch.Tensor, out_gold: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    mse = (out_pred - out_gold).pow(2).mean(dim=-1)
    return masked_metric_mean(mse, query_mask)


def sender_rank_features(attn_sender: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(attn_sender, dim=-1, descending=True)
    ranks = torch.zeros_like(attn_sender)
    rank_values = torch.arange(
        attn_sender.shape[-1],
        device=attn_sender.device,
        dtype=attn_sender.dtype,
    ).view(*([1] * (attn_sender.ndim - 1)), -1)
    rank_values = rank_values.expand_as(sorted_idx)
    ranks.scatter_(-1, sorted_idx, rank_values)
    return ranks / max(attn_sender.shape[-1] - 1, 1)


class RouteScorer(nn.Module):
    """Residual route scorer: final_score = log(sender_attn_score) + delta_score."""

    def __init__(self, q_dim: int, key_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim + key_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        q: torch.Tensor,
        key_sketch: torch.Tensor,
        sender_attn_score: torch.Tensor,
        sender_rank: torch.Tensor,
        key_norm: torch.Tensor,
        block_saliency: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        bsz, heads, seq_len, q_dim = q.shape
        key_dim = key_sketch.shape[-1]
        q_pair = q[:, :, :, None, :].expand(bsz, heads, seq_len, seq_len, q_dim)
        k_pair = key_sketch[:, :, None, :, :].expand(bsz, heads, seq_len, seq_len, key_dim)
        pos = (
            torch.arange(seq_len, device=q.device)[None, :]
            - torch.arange(seq_len, device=q.device)[:, None]
        ).float()
        pos = (pos / max(seq_len - 1, 1))[None, None, :, :, None].expand(bsz, heads, seq_len, seq_len, 1)
        layer_feat = torch.full_like(pos, float(layer_idx) / max(num_layers - 1, 1))
        head_ids = torch.arange(heads, device=q.device).float() / max(heads - 1, 1)
        head_feat = head_ids[None, :, None, None, None].expand(bsz, heads, seq_len, seq_len, 1)
        sender_attn_feat = sender_attn_score[..., None]
        sender_rank_feat = sender_rank[..., None]
        key_norm_feat = key_norm[:, :, None, :, None].expand(bsz, heads, seq_len, seq_len, 1)
        block_saliency_feat = block_saliency[:, :, None, :, None].expand(bsz, heads, seq_len, seq_len, 1)
        x = torch.cat(
            [
                q_pair,
                k_pair,
                sender_attn_feat,
                sender_rank_feat,
                key_norm_feat,
                block_saliency_feat,
                pos,
                layer_feat,
                head_feat,
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


def tokenize_pair(
    tokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def collect_qkv_features(
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
):
    # 先一次性抽取全部样本的 Q/K/V，再训练每一层的小 translator。
    # 这样训练 translator 时不需要反复跑两个大模型，速度更稳定。
    sender_extractor = QKVExtractor(sender_model)
    receiver_extractor = QKVExtractor(receiver_model)

    all_features = []

    loader = DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
    )

    for batch_texts in tqdm(loader, desc="Extracting QKV"):
        batch_texts = list(batch_texts)

        s_input_ids, s_mask = tokenize_pair(
            sender_tokenizer,
            batch_texts,
            max_length,
            device,
        )

        r_input_ids, r_mask = tokenize_pair(
            receiver_tokenizer,
            batch_texts,
            max_length,
            device,
        )

        s_qkv = sender_extractor.extract(s_input_ids, s_mask)
        r_qkv = receiver_extractor.extract(r_input_ids, r_mask)

        all_features.append(
            {
                "s_qkv": s_qkv,
                "r_qkv": r_qkv,
                "s_mask": s_mask.cpu(),
                "r_mask": r_mask.cpu(),
            }
        )

    return all_features


def train_one_layer(
    features,
    layer_idx: int,
    translator_type: str,
    rank: int,
    epochs: int,
    lr: float,
    device: torch.device,
    lambda_attn: float,
    lambda_out: float,
):
    sample = features[0]

    # translator 的输入维度来自 sender 的 head_dim，输出维度来自 receiver 的 head_dim。
    sender_dim = sample["s_qkv"].k[layer_idx].shape[-1]
    receiver_dim = sample["r_qkv"].k[layer_idx].shape[-1]

    if translator_type == "linear":
        f_k = LinearKVTranslator(sender_dim, receiver_dim).to(device)
        f_v = LinearKVTranslator(sender_dim, receiver_dim).to(device)
    elif translator_type == "lowrank":
        f_k = LowRankKVTranslator(sender_dim, receiver_dim, rank=rank).to(device)
        f_v = LowRankKVTranslator(sender_dim, receiver_dim, rank=rank).to(device)
    else:
        raise ValueError(f"Unknown translator_type: {translator_type}")

    optimizer = torch.optim.AdamW(
        list(f_k.parameters()) + list(f_v.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        total_kl = 0.0
        total_mse = 0.0
        count = 0

        for item in features:
            # 本实验只学习 sender K/V -> receiver K/V。
            # Q 固定使用 receiver 的 Q，因为目标是复原 receiver 侧的注意力行为。
            q_r = item["r_qkv"].q[layer_idx].to(device)
            k_r = item["r_qkv"].k[layer_idx].to(device)
            v_r = item["r_qkv"].v[layer_idx].to(device)

            k_s = item["s_qkv"].k[layer_idx].to(device)
            v_s = item["s_qkv"].v[layer_idx].to(device)

            r_mask = item["r_mask"].to(device)

            # 不同 tokenizer 可能让同一文本得到不同长度。
            # 这里裁到共同长度，保证 Q/K/V 可以做矩阵乘法。
            min_s = min(q_r.shape[-2], k_s.shape[-2], k_r.shape[-2])

            q_r = q_r[:, :, :min_s, :]
            k_r = k_r[:, :, :min_s, :]
            v_r = v_r[:, :, :min_s, :]

            k_s = k_s[:, :, :min_s, :]
            v_s = v_s[:, :, :min_s, :]

            r_mask = r_mask[:, :min_s]

            k_hat = f_k(k_s)
            v_hat = f_v(v_s)

            with torch.no_grad():
                # gold 是 receiver 自己的注意力图和输出，是训练要逼近的目标。
                attn_gold = attention_probs(q_r, k_r, r_mask)
                out_gold = attention_output(attn_gold, v_r)

            # pred 是 receiver Q + 映射后的 sender K/V 得到的注意力图和输出。
            attn_pred = attention_probs(q_r, k_hat, r_mask)
            out_pred = attention_output(attn_pred, v_hat)

            loss_kl = attention_kl_loss(attn_gold, attn_pred)
            loss_mse = F.mse_loss(out_pred, out_gold)

            loss = lambda_attn * loss_kl + lambda_out * loss_mse

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(f_k.parameters()) + list(f_v.parameters()),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            total_kl += loss_kl.item()
            total_mse += loss_mse.item()
            count += 1

        print(
            f"Layer {layer_idx:02d} | Epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / count:.6f} | "
            f"attn_KL={total_kl / count:.6f} | "
            f"out_MSE={total_mse / count:.6f}"
        )

    return f_k, f_v


@torch.no_grad()
def evaluate_one_layer(
    features,
    layer_idx: int,
    f_k: nn.Module,
    f_v: nn.Module,
    device: torch.device,
):
    """
    评估四种情况：

    1. oracle / no_reuse:
       不复用 sender KV。
       receiver 正常使用自己的 K/V。
       这是最强上界，因此 KL=0，MSE=0。

    2. zero_kv:
       receiver 的 K/V 缺失，而且没有任何补偿。
       用全零 K/V 占位。
       注意：这不是正常的 no_reuse，而是 missing-KV-without-compensation baseline。

    3. direct:
       直接复用 sender KV。
       不训练映射，只通过 adapt_dim 做裁剪/补零。

    4. translated:
       使用训练好的 f_k / f_v，把 sender KV 映射到 receiver KV 空间。
    """
    f_k.eval()
    f_v.eval()

    total_kl_oracle = 0.0
    total_mse_oracle = 0.0

    total_kl_zero = 0.0
    total_mse_zero = 0.0

    total_kl_direct = 0.0
    total_mse_direct = 0.0

    total_kl_translated = 0.0
    total_mse_translated = 0.0

    count = 0

    for item in features:
        q_r = item["r_qkv"].q[layer_idx].to(device)
        k_r = item["r_qkv"].k[layer_idx].to(device)
        v_r = item["r_qkv"].v[layer_idx].to(device)

        k_s = item["s_qkv"].k[layer_idx].to(device)
        v_s = item["s_qkv"].v[layer_idx].to(device)

        r_mask = item["r_mask"].to(device)

        # 评估阶段和训练阶段保持同样的长度对齐方式。
        min_s = min(q_r.shape[-2], k_s.shape[-2], k_r.shape[-2])

        q_r = q_r[:, :, :min_s, :]
        k_r = k_r[:, :, :min_s, :]
        v_r = v_r[:, :, :min_s, :]

        k_s = k_s[:, :, :min_s, :]
        v_s = v_s[:, :, :min_s, :]

        r_mask = r_mask[:, :min_s]

        # ============================================================
        # Oracle / no_reuse:
        # receiver 不复用 sender KV，正常使用自己的 K/V。
        # 这就是大模型原始行为，是本实验最强上界。
        # ============================================================
        attn_gold = attention_probs(q_r, k_r, r_mask)
        out_gold = attention_output(attn_gold, v_r)

        attn_oracle = attn_gold
        out_oracle = out_gold

        # ============================================================
        # Baseline 0: zero_kv
        # receiver 的 KV 缺失，没有任何补偿，用全零 K/V 占位。
        # 注意这不是正常不复用，而是“KV 缺失且无补偿”。
        # ============================================================
        k_zero = torch.zeros_like(k_r)
        v_zero = torch.zeros_like(v_r)

        attn_zero = attention_probs(q_r, k_zero, r_mask)
        out_zero = attention_output(attn_zero, v_zero)

        # ============================================================
        # Baseline 1: direct
        # 直接复用 sender KV，不训练映射。
        # 只通过裁剪/补零把 sender head_dim 调整到 receiver head_dim。
        # ============================================================
        k_direct = adapt_dim(k_s, k_r.shape[-1])
        v_direct = adapt_dim(v_s, v_r.shape[-1])

        attn_direct = attention_probs(q_r, k_direct, r_mask)
        out_direct = attention_output(attn_direct, v_direct)

        # ============================================================
        # Method: translated
        # 使用训练好的 f_k / f_v，将 sender KV 翻译成 receiver 风格 KV。
        # ============================================================
        k_translated = f_k(k_s)
        v_translated = f_v(v_s)

        attn_translated = attention_probs(q_r, k_translated, r_mask)
        out_translated = attention_output(attn_translated, v_translated)

        # oracle / no_reuse：理论上严格为 0。
        total_kl_oracle += attention_kl_loss(attn_gold, attn_oracle).item()
        total_mse_oracle += F.mse_loss(out_oracle, out_gold).item()

        # zero_kv
        total_kl_zero += attention_kl_loss(attn_gold, attn_zero).item()
        total_mse_zero += F.mse_loss(out_zero, out_gold).item()

        # direct
        total_kl_direct += attention_kl_loss(attn_gold, attn_direct).item()
        total_mse_direct += F.mse_loss(out_direct, out_gold).item()

        # translated
        total_kl_translated += attention_kl_loss(attn_gold, attn_translated).item()
        total_mse_translated += F.mse_loss(out_translated, out_gold).item()

        count += 1

    return {
        "layer": layer_idx,

        # no_reuse = receiver 使用自己的 KV，不复用 sender KV
        "attn_kl_no_reuse": total_kl_oracle / count,
        "out_mse_no_reuse": total_mse_oracle / count,

        # zero_kv = KV 缺失且无补偿
        "attn_kl_zero_kv": total_kl_zero / count,
        "out_mse_zero_kv": total_mse_zero / count,

        # direct sender KV
        "attn_kl_direct": total_kl_direct / count,
        "out_mse_direct": total_mse_direct / count,

        # translated sender KV
        "attn_kl_translated": total_kl_translated / count,
        "out_mse_translated": total_mse_translated / count,
    }


def adapt_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    # 这是一个朴素基线，不学习任何参数；只为 direct 指标提供可比较的 K/V。
    cur_dim = x.shape[-1]

    if cur_dim == target_dim:
        return x

    if cur_dim > target_dim:
        return x[..., :target_dim]

    pad = torch.zeros(
        *x.shape[:-1],
        target_dim - cur_dim,
        device=x.device,
        dtype=x.dtype,
    )

    return torch.cat([x, pad], dim=-1)


def prepare_route_tensors(item: Dict, layer_idx: int, device: torch.device):
    q_r = item["r_qkv"].q[layer_idx].to(device)
    k_r = item["r_qkv"].k[layer_idx].to(device)
    v_r = item["r_qkv"].v[layer_idx].to(device)

    q_s = item["s_qkv"].q[layer_idx].to(device)
    k_s = item["s_qkv"].k[layer_idx].to(device)
    v_s = item["s_qkv"].v[layer_idx].to(device)

    r_mask = item["r_mask"].to(device)
    s_mask = item["s_mask"].to(device)

    target_heads = q_r.shape[1]
    q_s = align_heads(q_s, target_heads)
    k_s = align_heads(k_s, target_heads)
    v_s = align_heads(v_s, target_heads)

    seq_len = min(
        q_r.shape[-2],
        k_r.shape[-2],
        v_r.shape[-2],
        q_s.shape[-2],
        k_s.shape[-2],
        v_s.shape[-2],
    )

    return {
        "q_r": q_r[:, :, :seq_len, :],
        "k_r": k_r[:, :, :seq_len, :],
        "v_r": v_r[:, :, :seq_len, :],
        "q_s": q_s[:, :, :seq_len, :],
        "k_s": k_s[:, :, :seq_len, :],
        "v_s": v_s[:, :, :seq_len, :],
        "r_mask": r_mask[:, :seq_len],
        "s_mask": s_mask[:, :seq_len],
    }


def route_scores_dot(q_r: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
    key = adapt_dim(k_s, q_r.shape[-1])
    return torch.matmul(q_r, key.transpose(-1, -2)) / math.sqrt(q_r.shape[-1])


def route_aux_features(t: Dict, attn_sender: torch.Tensor, block_size: int):
    key_norm = normalize_token_scores(t["k_s"].norm(dim=-1), t["r_mask"])
    block_saliency = block_token_scores(key_norm, block_size)
    sender_rank = sender_rank_features(attn_sender)
    key_sketch = adapt_dim(t["k_s"], t["q_r"].shape[-1])
    return key_sketch, key_norm, block_saliency, sender_rank


def residual_route_scores(
    route_scorer: Optional[RouteScorer],
    t: Dict,
    attn_sender: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    block_size: int,
    delta_scale: float,
) -> torch.Tensor:
    base = torch.log(attn_sender.clamp_min(1e-8))
    if route_scorer is None:
        return base

    key_sketch, key_norm, block_saliency, sender_rank = route_aux_features(
        t=t,
        attn_sender=attn_sender,
        block_size=block_size,
    )
    delta = route_scorer(
        t["q_r"],
        key_sketch,
        attn_sender,
        sender_rank,
        key_norm,
        block_saliency,
        layer_idx,
        num_layers,
    )
    return base + delta_scale * delta


def add_content_metrics(
    totals: Dict[str, float],
    prefix: str,
    attn_route: torch.Tensor,
    out_gold: torch.Tensor,
    t: Dict,
    f_v: Optional[nn.Module],
):
    def add(name: str, value: torch.Tensor):
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

    v_direct = adapt_dim(t["v_s"], t["v_r"].shape[-1])
    out_direct = attention_output(attn_route, v_direct)
    add(f"{prefix}_direct_v_mse", output_mse(out_direct, out_gold, t["r_mask"].bool()))
    add(f"{prefix}_direct_v_cosine", output_cosine(out_direct, out_gold, t["r_mask"].bool()))

    if f_v is not None:
        v_translated = f_v(t["v_s"])
        out_translated = attention_output(attn_route, v_translated)
        add(f"{prefix}_translated_v_mse", output_mse(out_translated, out_gold, t["r_mask"].bool()))
        add(f"{prefix}_translated_v_cosine", output_cosine(out_translated, out_gold, t["r_mask"].bool()))

    out_oracle_content = attention_output(attn_route, t["v_r"])
    add(f"{prefix}_oracle_receiver_v_mse", output_mse(out_oracle_content, out_gold, t["r_mask"].bool()))
    add(f"{prefix}_oracle_receiver_v_cosine", output_cosine(out_oracle_content, out_gold, t["r_mask"].bool()))


def train_route_scorer(
    features,
    layer_idx: int,
    num_layers: int,
    candidate_budget: int,
    route_k: int,
    route_epochs: int,
    lr: float,
    device: torch.device,
    block_size: int,
    saliency_weight: float,
    block_weight: float,
    hidden_dim: int,
    delta_scale: float,
    topk_bce_alpha: float,
) -> Optional[RouteScorer]:
    if route_epochs <= 0:
        return None

    sample = prepare_route_tensors(features[0], layer_idx, device)
    scorer = RouteScorer(
        q_dim=sample["q_r"].shape[-1],
        key_dim=sample["q_r"].shape[-1],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)

    for epoch in range(route_epochs):
        total_loss = 0.0
        count = 0
        for item in features:
            t = prepare_route_tensors(item, layer_idx, device)
            with torch.no_grad():
                attn_gold = attention_probs(t["q_r"], t["k_r"], t["r_mask"])
                attn_sender = attention_probs(t["q_s"], t["k_s"], t["s_mask"])
                cand_scores = candidate_scores_from_sender(
                    attn_sender=attn_sender,
                    k_sender=t["k_s"],
                    key_mask=t["r_mask"],
                    block_size=block_size,
                    saliency_weight=saliency_weight,
                    block_weight=block_weight,
                )
                candidate_mask = topk_mask_from_scores(cand_scores, candidate_budget)
                gold_on_candidates = attn_gold.masked_fill(~candidate_mask, 0.0)
                gold_dist = gold_on_candidates / gold_on_candidates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                gold_idx = topk_attention_indices(attn_gold, route_k=route_k)
                gold_topk_mask = torch.zeros_like(attn_gold)
                gold_topk_mask.scatter_(-1, gold_idx, 1.0)

            pred_scores = residual_route_scores(
                route_scorer=scorer,
                t=t,
                attn_sender=attn_sender,
                layer_idx=layer_idx,
                num_layers=num_layers,
                block_size=block_size,
                delta_scale=delta_scale,
            )
            pred_scores = pred_scores.masked_fill(~candidate_mask, torch.finfo(pred_scores.dtype).min)
            log_pred = F.log_softmax(pred_scores, dim=-1)
            loss_per_query = F.kl_div(log_pred, gold_dist, reduction="none").sum(dim=-1)
            loss_kl = masked_metric_mean(loss_per_query, t["r_mask"].bool())

            bce_per_token = F.binary_cross_entropy_with_logits(
                pred_scores.masked_fill(~candidate_mask, 0.0),
                gold_topk_mask,
                reduction="none",
            )
            bce_per_token = bce_per_token * candidate_mask.float()
            bce_per_query = bce_per_token.sum(dim=-1) / candidate_mask.float().sum(dim=-1).clamp_min(1.0)
            loss_bce = masked_metric_mean(bce_per_query, t["r_mask"].bool())
            loss = loss_kl + topk_bce_alpha * loss_bce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        print(
            f"Layer {layer_idx:02d} | RouteScorer epoch {epoch + 1}/{route_epochs} | "
            f"loss={total_loss / max(count, 1):.6f}"
        )

    return scorer


@torch.no_grad()
def evaluate_route_recovery(
    features,
    layer_idx: int,
    num_layers: int,
    f_v: nn.Module,
    route_scorer: Optional[RouteScorer],
    device: torch.device,
    route_k: int,
    candidate_budgets: List[int],
    rerank_candidate_budget: int,
    block_size: int,
    saliency_weight: float,
    block_weight: float,
    delta_scale: float,
) -> Dict[str, float]:
    if route_scorer is not None:
        route_scorer.eval()
    if f_v is not None:
        f_v.eval()

    totals: Dict[str, float] = {}
    count = 0

    def add(name: str, value: torch.Tensor):
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

    for item in features:
        t = prepare_route_tensors(item, layer_idx, device)
        attn_gold = attention_probs(t["q_r"], t["k_r"], t["r_mask"])
        attn_sender = attention_probs(t["q_s"], t["k_s"], t["s_mask"])

        gold_idx = topk_attention_indices(attn_gold, route_k)
        sender_idx = topk_attention_indices(attn_sender, route_k)

        add("path_sender_topk_overlap", route_overlap(sender_idx, gold_idx, t["r_mask"].bool()))
        add("path_sender_topk_mrr", route_mrr(sender_idx, gold_idx, t["r_mask"].bool()))
        add("path_sender_attention_mass", attention_mass_on_route(attn_gold, sender_idx, t["r_mask"].bool()))

        cand_scores = candidate_scores_from_sender(
            attn_sender=attn_sender,
            k_sender=t["k_s"],
            key_mask=t["r_mask"],
            block_size=block_size,
            saliency_weight=saliency_weight,
            block_weight=block_weight,
        )
        for budget in candidate_budgets:
            cand_mask = topk_mask_from_scores(cand_scores, budget)
            add(f"candidate_recall_at_{budget}", candidate_recall(cand_mask, gold_idx, t["r_mask"].bool()))

        candidate_mask = topk_mask_from_scores(cand_scores, rerank_candidate_budget)
        rerank_scores = residual_route_scores(
            route_scorer=route_scorer,
            t=t,
            attn_sender=attn_sender,
            layer_idx=layer_idx,
            num_layers=num_layers,
            block_size=block_size,
            delta_scale=delta_scale,
        )
        rerank_scores = rerank_scores.masked_fill(~candidate_mask, torch.finfo(rerank_scores.dtype).min)
        route_hat = torch.topk(rerank_scores, k=min(route_k, rerank_scores.shape[-1]), dim=-1).indices
        attn_hat = masked_softmax_from_indices(rerank_scores, route_hat)

        out_gold = attention_output(attn_gold, t["v_r"])
        add("route_hat_overlap", route_overlap(route_hat, gold_idx, t["r_mask"].bool()))
        add("route_hat_mrr", route_mrr(route_hat, gold_idx, t["r_mask"].bool()))
        add("route_hat_attention_mass", attention_mass_on_route(attn_gold, route_hat, t["r_mask"].bool()))
        add("route_hat_weight_mse", route_weight_mse(attn_gold, attn_hat, route_hat, t["r_mask"].bool()))
        add("route_hat_weight_cosine", route_weight_cosine(attn_gold, attn_hat, route_hat, t["r_mask"].bool()))
        add("route_hat_candidate_kl", candidate_level_kl(attn_gold, rerank_scores, candidate_mask, t["r_mask"].bool()))

        attn_gold_route = normalized_attention_on_route(attn_gold, gold_idx)
        attn_sender_route = normalized_attention_on_route(attn_sender, sender_idx)
        add("sender_route_weight_mse", route_weight_mse(attn_gold, attn_sender_route, sender_idx, t["r_mask"].bool()))
        add("sender_route_weight_cosine", route_weight_cosine(attn_gold, attn_sender_route, sender_idx, t["r_mask"].bool()))
        add("gold_route_weight_mse", route_weight_mse(attn_gold, attn_gold_route, gold_idx, t["r_mask"].bool()))
        add("gold_route_weight_cosine", route_weight_cosine(attn_gold, attn_gold_route, gold_idx, t["r_mask"].bool()))
        add_content_metrics(totals, "gold_route", attn_gold_route, out_gold, t, f_v)
        add_content_metrics(totals, "sender_route", attn_sender_route, out_gold, t, f_v)
        add_content_metrics(totals, "route_hat", attn_hat, out_gold, t, f_v)

        entropy = -(attn_hat.clamp_min(1e-8) * attn_hat.clamp_min(1e-8).log()).sum(dim=-1)
        top2 = torch.topk(rerank_scores, k=min(2, rerank_scores.shape[-1]), dim=-1).values
        margin = top2[..., 0] - top2[..., -1]
        margin = torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin))
        add("confidence_entropy", masked_metric_mean(entropy, t["r_mask"].bool()))
        add("confidence_top2_margin", masked_metric_mean(margin, t["r_mask"].bool()))

        count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


def require_local_model(path: str, label: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} 模型目录不存在: {path}")

    if not os.path.exists(os.path.join(path, "config.json")):
        raise FileNotFoundError(
            f"{label} 模型目录缺少 config.json: {path}\n"
            "请先把完整模型文件放入该目录。脚本已设置 local_files_only=True，不会联网下载模型。"
        )


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sender_model", type=str, default=LOCAL_SENDER_MODEL)
    parser.add_argument("--receiver_model", type=str, default=LOCAL_RECEIVER_MODEL)

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional ShareGPT/OpenHermes JSON array file. If set, texts are loaded from this dataset.",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=1000,
        help="Number of dataset examples to load when --dataset_path is set.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of loaded texts reserved for evaluation. Use 0 to evaluate on the training texts.",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--translator",
        type=str,
        choices=["linear", "lowrank"],
        default="linear",
    )

    parser.add_argument("--rank", type=int, default=32)

    parser.add_argument("--lambda_attn", type=float, default=1.0)
    parser.add_argument("--lambda_out", type=float, default=1.0)

    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="all 或逗号分隔，例如 0,8,16,24",
    )

    parser.add_argument("--save_dir", type=str, default="runs/kv_translation_ckpt")
    parser.add_argument(
        "--route_k",
        type=int,
        default=8,
        help="Top-k receiver attention path size used for path recovery metrics.",
    )
    parser.add_argument(
        "--candidate_budgets",
        type=str,
        default="32,64,128",
        help="Comma-separated candidate set sizes for Recall@K path coverage.",
    )
    parser.add_argument(
        "--rerank_candidate_budget",
        type=int,
        default=64,
        help="Candidate set size used by dot/MLP route reranking.",
    )
    parser.add_argument(
        "--route_block_size",
        type=int,
        default=16,
        help="Block size for block-level sender saliency sketch.",
    )
    parser.add_argument("--route_saliency_weight", type=float, default=0.25)
    parser.add_argument("--route_block_weight", type=float, default=0.25)
    parser.add_argument(
        "--route_scorer_epochs",
        type=int,
        default=0,
        help="If >0, train a lightweight MLP RouteScorer over candidate paths.",
    )
    parser.add_argument("--route_scorer_lr", type=float, default=1e-3)
    parser.add_argument("--route_scorer_hidden", type=int, default=128)
    parser.add_argument(
        "--route_topk_bce_alpha",
        type=float,
        default=0.1,
        help="Weight for candidate-set BCE loss that classifies receiver gold top-k route tokens.",
    )
    parser.add_argument(
        "--route_delta_scale",
        type=float,
        default=0.1,
        help="Scale for residual delta in final_score = log(sender_attn_score) + scale * delta_score.",
    )

    args = parser.parse_args()

    requested_device = args.device

    device = torch.device(
        requested_device
        if requested_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    if requested_device == "cuda" and device.type != "cuda":
        print("WARNING: 当前 torch 不支持 CUDA，已回退到 CPU。")

    require_local_model(args.sender_model, "sender")
    require_local_model(args.receiver_model, "receiver")

    os.makedirs(args.save_dir, exist_ok=True)

    # local_files_only=True 保证只从本地路径加载，不会自动访问 Hugging Face。
    print("Loading sender model...")

    sender_tokenizer = AutoTokenizer.from_pretrained(
        args.sender_model,
        trust_remote_code=True,
        local_files_only=True,
    )

    sender_model = AutoModelForCausalLM.from_pretrained(
        args.sender_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    print("Loading receiver model...")

    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model,
        trust_remote_code=True,
        local_files_only=True,
    )

    receiver_model = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    if sender_tokenizer.pad_token is None:
        sender_tokenizer.pad_token = sender_tokenizer.eos_token

    if receiver_tokenizer.pad_token is None:
        receiver_tokenizer.pad_token = receiver_tokenizer.eos_token

    if args.dataset_path:
        texts = load_sharegpt_json_texts(args.dataset_path, args.dataset_limit)
        if not texts:
            raise RuntimeError(f"No usable conversations were loaded from dataset: {args.dataset_path}")
        print(f"Loaded {len(texts)} examples from dataset: {args.dataset_path}")
    else:
        texts = build_toy_texts(args.num_samples)

    train_texts, eval_texts = split_train_eval_texts(texts, args.eval_ratio)
    print(
        f"Split texts: train={len(train_texts)}, eval={len(eval_texts)}, "
        f"eval_ratio={args.eval_ratio}"
    )

    # features 中保存了两个模型所有层的 Q/K/V；后面会逐层训练映射器。
    print("Collecting train QKV features...")
    train_features = collect_qkv_features(
        sender_model=sender_model,
        receiver_model=receiver_model,
        sender_tokenizer=sender_tokenizer,
        receiver_tokenizer=receiver_tokenizer,
        texts=train_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    print("Collecting eval QKV features...")
    eval_features = collect_qkv_features(
        sender_model=sender_model,
        receiver_model=receiver_model,
        sender_tokenizer=sender_tokenizer,
        receiver_tokenizer=receiver_tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    num_layers = len(train_features[0]["r_qkv"].q)

    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = parse_int_list(args.layers)

    candidate_budgets = parse_int_list(args.candidate_budgets)

    results = []

    for layer_idx in layer_indices:
        # 每一层独立训练一组 f_k/f_v，并保存成单独 checkpoint。
        print("=" * 80)
        print(f"Training layer {layer_idx}/{num_layers - 1}")

        f_k, f_v = train_one_layer(
            features=train_features,
            layer_idx=layer_idx,
            translator_type=args.translator,
            rank=args.rank,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            lambda_attn=args.lambda_attn,
            lambda_out=args.lambda_out,
        )

        metrics = evaluate_one_layer(
            features=eval_features,
            layer_idx=layer_idx,
            f_k=f_k,
            f_v=f_v,
            device=device,
        )

        route_scorer = train_route_scorer(
            features=train_features,
            layer_idx=layer_idx,
            num_layers=num_layers,
            candidate_budget=args.rerank_candidate_budget,
            route_k=args.route_k,
            route_epochs=args.route_scorer_epochs,
            lr=args.route_scorer_lr,
            device=device,
            block_size=args.route_block_size,
            saliency_weight=args.route_saliency_weight,
            block_weight=args.route_block_weight,
            hidden_dim=args.route_scorer_hidden,
            delta_scale=args.route_delta_scale,
            topk_bce_alpha=args.route_topk_bce_alpha,
        )

        route_metrics = evaluate_route_recovery(
            features=eval_features,
            layer_idx=layer_idx,
            num_layers=num_layers,
            f_v=f_v,
            route_scorer=route_scorer,
            device=device,
            route_k=args.route_k,
            candidate_budgets=candidate_budgets,
            rerank_candidate_budget=args.rerank_candidate_budget,
            block_size=args.route_block_size,
            saliency_weight=args.route_saliency_weight,
            block_weight=args.route_block_weight,
            delta_scale=args.route_delta_scale,
        )
        metrics.update(route_metrics)

        results.append(metrics)

        print("Evaluation:", metrics)

        torch.save(
            {
                "layer": layer_idx,
                "f_k": f_k.state_dict(),
                "f_v": f_v.state_dict(),
                "route_scorer": route_scorer.state_dict() if route_scorer is not None else None,
                "metrics": metrics,
                "args": vars(args),
            },
            os.path.join(args.save_dir, f"layer_{layer_idx:02d}.pt"),
        )

    print("=" * 80)
    print("Final results")

    for r in results:
        print(
            f"Layer {r['layer']:02d} | "
            f"KL no_reuse/oracle={r['attn_kl_no_reuse']:.6f} | "
            f"zero_kv={r['attn_kl_zero_kv']:.6f} | "
            f"direct={r['attn_kl_direct']:.6f} | "
            f"translated={r['attn_kl_translated']:.6f} || "
            f"MSE no_reuse/oracle={r['out_mse_no_reuse']:.6f} | "
            f"zero_kv={r['out_mse_zero_kv']:.6f} | "
            f"direct={r['out_mse_direct']:.6f} | "
            f"translated={r['out_mse_translated']:.6f} || "
            f"path sender_overlap={r['path_sender_topk_overlap']:.6f} | "
            f"cand_R@{args.rerank_candidate_budget}={r.get(f'candidate_recall_at_{args.rerank_candidate_budget}', 0.0):.6f} | "
            f"route_hat_overlap={r['route_hat_overlap']:.6f} | "
            f"route_hat_oracle_content_MSE={r['route_hat_oracle_receiver_v_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
