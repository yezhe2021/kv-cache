# Experiment Results Summary: Routing Prior / Evidence Region

Remote project:

```bash
/home/yezhe/恢复注意力图代码/注意力图恢复
```

Run environment:

```text
Python: /home/yezhe/data/miniconda3/envs/attnkv/bin/python
GPU: Tesla V100-PCIE-32GB
num_samples: 8
max_length: 512
layers: 0,4,8,12,15
```

## 1. Overall Conclusion

The five rerun experiments support the current direction:

```text
Do not try to recover the full cross-model attention matrix.
Use sender K / attention as a routing prior to select receiver evidence blocks,
then let receiver recompute its own attention output inside the selected region.
```

The strongest evidence comes from fixed-budget comparison. Under the same token/block budget, sender-guided block methods consistently outperform random, recent, and uniform baselines. This indicates that sender K/attention contains transferable routing information, not just a larger candidate set.

## 2. Fixed-Budget Fair Compare

Script:

```bash
fixed_budget_fair_compare.py
```

Result file:

```bash
runs/fixed_budget_fair_compare_rerun_20260530.csv
```

Purpose:

Compare sender-guided routing methods against random/recent/uniform baselines under the same retention budget.

Important modes:

```text
random_block
uniform_block
sender_attn_block
sender_k_norm_block
sender_kxreceived_block
```

Key results:

```text
budget=0.2, layer=8
random_block          mass=0.1882  cos=0.7072
uniform_block         mass=0.5329  cos=0.7613
sender_attn_block     mass=0.6014  cos=0.8536
sender_k_norm_block   mass=0.4299  cos=0.7846
sender_kxrecv_block   mass=0.5966  cos=0.8019

budget=0.2, layer=12
random_block          mass=0.1884  cos=0.7216
uniform_block         mass=0.5856  cos=0.8437
sender_attn_block     mass=0.5839  cos=0.8758
sender_k_norm_block   mass=0.3972  cos=0.7918
sender_kxrecv_block   mass=0.6595  cos=0.8474

budget=0.4, layer=12
random_block          mass=0.3757  cos=0.8093
uniform_block         mass=0.6469  cos=0.9121
sender_attn_block     mass=0.7892  cos=0.9390
sender_k_norm_block   mass=0.6073  cos=0.8696
sender_kxrecv_block   mass=0.7891  cos=0.9027

budget=0.4, layer=15
random_block          mass=0.3745  cos=0.8630
uniform_block         mass=0.6294  cos=0.9298
sender_attn_block     mass=0.7612  cos=0.9428
sender_k_norm_block   mass=0.4834  cos=0.8751
sender_kxrecv_block   mass=0.7865  cos=0.9171
```

Conclusion:

Sender-guided block routing is clearly better than random and recent baselines. It also usually improves candidate mass over uniform block selection. This is the main evidence that sender routing prior is real.

## 3. Rule K-Routing Prior

Script:

```bash
rule_k_routing_prior.py
```

Result file:

```bash
runs/rule_k_routing_prior_rerun_20260530.csv
```

Purpose:

Evaluate which rule-based K-derived signals are useful routing priors.

Compared modes:

```text
k_norm
received_mass
k_norm_x_received
k_outlier
local_k_variance
```

Key results:

```text
layer=8
best mass: k_outlier
ratio=0.703  mass=0.8430  cos=0.9522

layer=12
best mass: k_norm
ratio=0.717  mass=0.8431  cos=0.9499

layer=15
best mass: k_outlier
ratio=0.744  mass=0.8559  cos=0.9578
```

More compact high-mass options:

```text
layer=12 received_mass
ratio=0.509  mass=0.8051  cos=0.9394

layer=15 received_mass
ratio=0.583  mass=0.8307  cos=0.9537
```

Conclusion:

K norm and K outlier are strong but often select many tokens. Received attention mass is more compact in higher layers. A promising scorer should combine K strength and received attention mass under a fixed budget.

## 4. Learned Block Routing Predictor

Script:

```bash
train_block_routing_predictor.py
```

Result file:

```bash
runs/train_block_routing_predictor_rerun_20260530.csv
```

Purpose:

Train a lightweight predictor that maps sender block features to receiver oracle blocks.

Current features:

```text
K norm mean/max
K variance
received attention mean/max
```

Results:

```text
L0   ratio=0.373  mass=0.5732  cos=0.9152
L4   ratio=0.373  mass=0.5221  cos=0.8967
L8   ratio=0.372  mass=0.3918  cos=0.8215
L12  ratio=0.369  mass=0.3842  cos=0.7986
L15  ratio=0.371  mass=0.4102  cos=0.9098
```

Conclusion:

The predictor runs, but it is weaker than rule-based methods. It should be treated as an MVP baseline, not the current main method.

Potential improvements:

```text
head-level features
multi-layer sender features
relative position features
block-level K statistics
pairwise/ranking loss
fixed-budget top-block objective
output-equivalent block mask as training target
```

## 5. Block / Span Routing Sweep

Script:

```bash
block_span_routing_experiment.py
```

Result files:

```bash
runs/block_span_routing_b16_k16.csv
runs/block_span_routing_b16_k32.csv
runs/block_span_routing_b32_k16.csv
runs/block_span_routing_b32_k32.csv
runs/block_span_routing_b64_k16.csv
runs/block_span_routing_b64_k32.csv
```

Purpose:

Sweep receiver block size and anchor budget to test block/span evidence region quality.

Summary for `union_recv_block`, averaged over layers 8, 12, and 15:

```text
b16 k16  ratio=0.440  mass=0.7690  cos=0.9419
b16 k32  ratio=0.563  mass=0.8259  cos=0.9509
b32 k16  ratio=0.532  mass=0.8095  cos=0.9485
b32 k32  ratio=0.660  mass=0.8470  cos=0.9532
b64 k16  ratio=0.622  mass=0.8319  cos=0.9512
b64 k32  ratio=0.717  mass=0.8551  cos=0.9538
```

Conclusion:

`block_size=32, anchor_k=16` is a good tradeoff:

```text
avg ratio=0.532
avg mass=0.8095
avg cos=0.9485
```

Larger blocks or larger anchor budgets slightly improve mass/cos, but they increase retained ratio much more.

## 6. Output-Equivalent Region

Script:

```bash
output_equivalent_region.py
```

Result files:

```bash
runs/output_equivalent_region_k16_b32.csv
runs/output_equivalent_region_k32_b32.csv
runs/output_equivalent_region_k64_b32.csv
```

Purpose:

Evaluate whether restricted evidence regions can recover receiver attention output, rather than matching the full attention matrix.

Summary for `union_recv_block`, averaged over layers 8, 12, and 15:

```text
k16 b32  ratio=0.532  mass=0.8095  cos=0.9485
k32 b32  ratio=0.660  mass=0.8470  cos=0.9532
k64 b32  ratio=0.725  mass=0.8565  cos=0.9539
```

Layer details:

```text
k16 b32
L8   ratio=0.510  mass=0.7885  cos=0.9449
L12  ratio=0.506  mass=0.8025  cos=0.9449
L15  ratio=0.580  mass=0.8374  cos=0.9559

k32 b32
L8   ratio=0.636  mass=0.8405  cos=0.9522
L12  ratio=0.676  mass=0.8471  cos=0.9500
L15  ratio=0.670  mass=0.8533  cos=0.9576

k64 b32
L8   ratio=0.708  mass=0.8553  cos=0.9536
L12  ratio=0.736  mass=0.8573  cos=0.9504
L15  ratio=0.731  mass=0.8569  cos=0.9578
```

Conclusion:

`k=16, block=32` already reaches output cosine around 0.95. Increasing to `k=32` gives a small gain. Increasing to `k=64` gives almost no cosine gain but keeps substantially more tokens.

## 7. Recommended Next Steps

1. Treat `fixed_budget_fair_compare.py` as the main evidence experiment.

2. Expand fixed-budget evaluation:

```text
more samples
more layers
more model pairs
budgets from 10% to 60%
```

3. Focus on compact scorer design:

```text
K norm
K outlier
received mass
K norm x received mass
multi-scale block scores
fixed-budget block selection
```

4. Use output recovery as the main target:

```text
selected_ratio
candidate_mass
gold_topk_recall
selective_recompute_mse
selective_recompute_cosine
```

5. Do not use token-level route overlap as the primary objective for heterogeneous models. It is useful as diagnosis, but current results show that exact token-level route matching is weak for Qwen -> Llama.

