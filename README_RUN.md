# Attention-Preserving KV Translation

Remote project path:

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复
```

## Layout

```text
.
├── attention_preserving_kv_translation_experiment.py  # main experiment
├── summarize_kv_results.py                            # summarize layer_*.pt metrics
├── requirements.txt
├── README_RUN.md
└── runs/                                              # experiment outputs/checkpoints
```

Current run folders are stored under `runs/`.

## Models

The experiment script uses local model paths and will not download models:

```text
/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B
/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B
```

## Run

Smoke or CPU run:

```bash
python3 attention_preserving_kv_translation_experiment.py \
  --device cpu \
  --num_samples 128 \
  --max_length 128 \
  --epochs 5
```

CUDA run, if CUDA torch is installed:

```bash
python3 attention_preserving_kv_translation_experiment.py \
  --device cuda \
  --num_samples 128 \
  --max_length 128 \
  --epochs 5
```

Summarize the latest multi-layer CUDA probe:

```bash
python3 summarize_kv_results.py
```

Summarize a specific run:

```bash
python3 summarize_kv_results.py runs/kv_translation_ckpt
```

## Route Recovery MVP

The main script now reports receiver-side attention route recovery metrics in addition to the original KV translation baseline.

Implemented MVP pieces:

- sender/receiver top-k attention route extraction
- sender attention + K-norm saliency + block saliency candidate sketch
- candidate route coverage metrics: `Recall@32`, `Recall@64`, `Recall@128`
- residual candidate reranking in logit space: `score_final = log(sender_attn_score) + route_delta_scale * delta_score`
- optional MLP `RouteScorer` learns only the receiver-side correction term with candidate-level KL plus top-k BCE training
- routed content comparison split by `gold_route`, `sender_route`, and `route_hat`
- each route reports direct sender V, translated sender V, and oracle receiver V content metrics
- weight-level route metrics: top-k weight MSE, top-k weight cosine, and candidate-level KL
- simple recovery quality features: route entropy and top-2 score margin

Example path-recovery smoke run:

```bash
python3 attention_preserving_kv_translation_experiment.py \
  --device cuda \
  --layers 0,4,8,12,16,20,24,27 \
  --num_samples 32 \
  --max_length 128 \
  --epochs 1 \
  --route_k 8 \
  --candidate_budgets 32,64,128 \
  --rerank_candidate_budget 64 \
  --save_dir runs/route_recovery_smoke
```

Enable learned candidate reranking:

```bash
python3 attention_preserving_kv_translation_experiment.py \
  --device cuda \
  --layers 0,4,8,12,16,20,24,27 \
  --num_samples 128 \
  --max_length 128 \
  --epochs 3 \
  --route_scorer_epochs 3 \
  --route_delta_scale 0.1 \
  --route_topk_bce_alpha 0.1 \
  --save_dir runs/route_recovery_mlp
```

For residual reranking, sweep small correction scales first:

```bash
--route_delta_scale 0.05
--route_delta_scale 0.1
--route_delta_scale 0.2
```
