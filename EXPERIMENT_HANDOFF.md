# Handoff: Cross-Model Routing Prior / Block Memory Translation

本文档总结当前项目目标、已完成实验、最新结论、关键脚本、结果文件和下一步工作。

远程项目目录：

```bash
/home/yezhe/恢复注意力图代码/注意力图恢复
```

推荐 Python 环境：

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python
```

主要数据源：

```bash
/home/yezhe/demo/train/**/*.txt
```

当前数据源约有 `10,749` 个 `.txt` 文件。多数实验使用快速验证配置：

```text
num_samples = 8 / 16 / 32 / 64
max_length = 512
```

## 1. 当前任务表述

早期问题表述是：

```text
异构模型之间是否可以恢复 / 复用 sender 的 token-level attention route 或完整 attention matrix？
```

目前这个表述已经被实验否定。更准确的目标是：

```text
sender 已经完成 prefill，拥有 K_s / V_s / attention；
从 sender 的 K_s / attention 中提取可迁移 routing prior；
用该 prior 选择 receiver 需要的 evidence block/span；
sender 将候选区域和 translated memory 传给 receiver；
receiver 尽量避免完整计算历史 K/V，
用 translated memory 近似恢复自己的 attention output / logits。
```

当前路线分成两层：

```text
1. 路由层：
   sender K / attention -> selected receiver blocks。

2. memory translation 层：
   selected sender K/V block memory -> receiver-readable K/V memory。
```

## 2. 已确认的核心结论

### 2.1 token-level route 不可靠

相关脚本：

```bash
compare_model_important_tokens.py
cross_model_oracle_candidates.py
compare_rerank_effect.py
```

结论：

```text
Qwen -> Llama 的 token-level per-query route overlap 很低；
Qwen -> Qwen 明显更高；
跨 tokenizer 模型必须使用 span/offset alignment，不能用简单 position alignment；
global important tokens 比 per-query token route 稳定。
```

因此不要继续把最终目标设为：

```text
sender token i -> receiver token j
sender attention weight -> receiver attention weight
完整 attention matrix recovery
```

### 2.2 block/span evidence region 是可靠方向

相关脚本：

```bash
evidence_recall_selective_recompute.py
block_span_routing_experiment.py
output_equivalent_region.py
fixed_budget_fair_compare.py
rule_k_routing_prior.py
```

固定预算实验说明：

```text
在相同 token/block 保留率下，
sender-guided block routing 明显优于 random / recent / uniform baseline。
```

这证明 sender 的 K / attention 中存在可迁移 routing prior，不只是因为选了更多 token。

### 2.3 当前块选择已经基本可靠

最新 block translated memory 实验使用：

```text
block_score_mode = anchor_count
anchor_tokens = 64
block_size = 32
budget_ratio = 0.5
```

注意：早期版本曾用 block 内平均分数选块，这是不正确/不充分的，因为它会稀释 K / attention 的稀疏峰值。现在已改为：

```text
1. 全局按 routing score 选 top anchor tokens；
2. 统计 anchor tokens 落入哪些 blocks；
3. 用 anchor_count 选择重要 blocks。
```

最新 64 样本结果：

```bash
runs/block_translated_memory_recovery_b32_r50_sparse_anchor_n64_v3.csv
```

关键结果：

```text
selected_ratio ≈ 0.438
selective_oracle_cosine ≈ 0.96
```

解释：

```text
只保留约 44% 的 block；
如果在这些 block 内使用 receiver token-level K/V 做 selective recompute，
attention output 与 full receiver output 的 cosine 约为 0.96。
```

所以当前主要瓶颈已经不是“块有没有选对”，而是：

```text
sender block memory 如何翻译成 receiver-readable memory。
```

## 3. 重要指标定义

### selected_ratio

保留的 block/token 比例。最新 sparse-anchor block 实验中大约是：

```text
selected_ratio ≈ 0.438
```

虽然命令中设置 `budget_ratio=0.5`，但统计时包含固定长度中的无效/padding block，因此实际显示约 0.438。

### selective_oracle_cosine

真正的 oracle 上限。

定义：

```text
在同样选中的 receiver blocks 内，
使用 receiver 自己的 token-level K/V 和 gold attention 做 selective recompute，
再与 full receiver attention output 比较 cosine。
```

它衡量的是：

```text
候选 evidence blocks 本身能支持多高的恢复上限。
```

### mean_block_cosine

不是 oracle。

定义：

```text
把 receiver block K/V 做 mean pooling 后再参与 attention。
```

它只是 block memory 压缩 baseline。

### translated_cosine / trans_cos

定义：

```text
translated memory 算出的 attention output
与 receiver full attention output 的 cosine similarity。
```

也就是比较：

```text
softmax(Q_r K_r^T) V_r
```

和：

```text
softmax(Q_r T_k(K_s)^T) T_v(V_s)
```

在当前 block memory 实验中，`trans_cos` 是单层 attention output 恢复质量，不等同于最终生成质量。

### generation proxy metrics

在生成代理实验中，我们把某层 attention output 替换进 receiver forward，然后比较：

```text
full_ce:
    原始 receiver next-token cross entropy。

patched_ce:
    替换 attention output 后的 next-token cross entropy。

ce_delta:
    patched_ce - full_ce，越接近 0 越好。

logit_kl:
    full logits 与 patched logits 的 KL。

top1_match:
    patched logits argmax 是否与 full logits argmax 一致。
```

## 4. 当前主要脚本

### 4.1 原始 KV translation

```bash
attention_preserving_kv_translation_experiment.py
summarize_kv_results.py
```

作用：

```text
训练 sender KV -> receiver KV 的线性/低秩映射；
评估 translated KV 对 attention / output 的恢复。
```

这条线是早期方案。当前重点已转到 block evidence region 和 block memory translation。

### 4.2 token-level route 诊断

```bash
compare_model_important_tokens.py
cross_model_oracle_candidates.py
compare_rerank_effect.py
attention_weight_preservation_experiment.py
```

作用：

```text
诊断 token-level route overlap；
比较 span alignment / position alignment；
验证直接复用 attention weight 的局限。
```

### 4.3 evidence region / selective recompute

```bash
evidence_recall_selective_recompute.py
block_span_routing_experiment.py
output_equivalent_region.py
fixed_budget_fair_compare.py
rule_k_routing_prior.py
```

作用：

```text
验证 sender K/attention 是否能召回 receiver evidence region；
在 selected region 内让 receiver 用自己的 K/V 重算 output；
固定预算下比较 sender-guided 与 random/recent/uniform。
```

### 4.4 learned block routing predictor

```bash
train_block_routing_predictor.py
```

当前结论：

```text
MVP 能跑通，但弱于规则式方法；
不能作为主方法，只能作为 learned routing baseline。
```

### 4.5 block-level translated memory recovery

```bash
block_translated_memory_recovery.py
```

作用：

```text
固定使用 sparse-anchor 选块；
把 selected sender K/V block memory 映射到 receiver-readable memory；
receiver Q attend translated memory；
恢复 receiver attention output。
```

关键修正：

```text
K memory 用 routing_pool_mode 池化；
V memory 用 value_pool_mode 池化；
block 选择使用 anchor_count，而不是 block 内平均；
selective_oracle_* 才是真 oracle；
mean_block_* 只是压缩 baseline。
```

### 4.6 block memory method sweep

```bash
block_memory_method_sweep.py
```

作用：

```text
在块选择已经固定的前提下，
比较不同 block memory 表示和 translator。
```

已实现方法：

```text
baseline_1slot_linear:
    每块 1 个 mean slot + shared linear translator。

multislot_linear:
    每块多个 top-token slots + shared linear translator。

multislot_mlp:
    多 slot + MLP translator。

multislot_mlp_norm:
    多 slot + MLP translator + 标准化。

multislot_headwise_norm:
    多 slot + head-wise linear translator + 标准化。

multislot_mlp_norm_prior:
    多 slot + MLP + 标准化 + sender routing prior attention bias。
```

### 4.7 generation proxy experiments

```bash
generation_effect_experiment.py
generation_effect_multilayer_experiment.py
generation_effect_self_consistent_oracle.py
```

作用：

```text
把 translated / oracle attention output patch 进 receiver forward；
比较 next-token CE、logit KL、top1 match；
评估单层、多层、全层替换对生成相关 logits 的影响。
```

## 5. 最新实验结果

### 5.1 sparse-anchor block translated memory, 64 samples

命令核心配置：

```text
num_samples = 64
max_length = 512
layers = 8,12,15
block_size = 32
budget_ratio = 0.5
block_score_mode = anchor_count
anchor_tokens = 64
pool_modes = received,kxreceived
value_pool_mode = uniform
```

结果文件：

```bash
runs/block_translated_memory_recovery_b32_r50_sparse_anchor_n64_v3.csv
```

结果：

```text
L08 received:
trans_cos=0.7660  selective_oracle_cos=0.9631  mean_block_cos=0.6745

L08 kxreceived:
trans_cos=0.7645  selective_oracle_cos=0.9636  mean_block_cos=0.6757

L12 received:
trans_cos=0.7156  selective_oracle_cos=0.9692  mean_block_cos=0.6386

L12 kxreceived:
trans_cos=0.7117  selective_oracle_cos=0.9591  mean_block_cos=0.6357

L15 received:
trans_cos=0.7166  selective_oracle_cos=0.9612  mean_block_cos=0.6920

L15 kxreceived:
trans_cos=0.7130  selective_oracle_cos=0.9619  mean_block_cos=0.6922
```

结论：

```text
块选择上限高；
translated memory 明显优于 mean block baseline；
但与 oracle 仍有差距，translation 仍是瓶颈。
```

### 5.2 block memory method sweep, 64 samples

结果文件：

```bash
runs/block_memory_method_sweep_b32_r50_n64.csv
```

关键结果：

```text
L08 baseline_1slot_linear      trans_cos=0.6851
L08 multislot_headwise_norm    trans_cos=0.8151

L12 baseline_1slot_linear      trans_cos=0.6330
L12 multislot_headwise_norm    trans_cos=0.7664

L15 baseline_1slot_linear      trans_cos=0.6411
L15 multislot_headwise_norm    trans_cos=0.8040
```

结论：

```text
多 slot + head-wise normalized translator 是当前最好的 memory translation 方案；
相比 1-slot linear baseline 有稳定大幅提升。
```

### 5.3 单层 generation proxy, 32 samples

结果文件：

```bash
runs/generation_effect_b32_r50_n32.csv
```

关键结果：

```text
L08 multislot_headwise_norm:
ce_delta=+0.0343  top1=0.847

L12 multislot_headwise_norm:
ce_delta=+0.0206  top1=0.898

L15 multislot_headwise_norm:
ce_delta=+0.0235  top1=0.908
```

结论：

```text
单层替换对 next-token logits 的破坏较小；
L12 / L15 效果比 L8 更稳。
```

### 5.4 多层 translated memory patch, 16 samples

结果文件：

```bash
runs/generation_effect_multilayer_with_oracle_b32_r50_n16.csv
```

结果：

```text
L[12,15] selective_oracle:
ce_delta=+0.0581  top1=0.883

L[12,15] multislot_headwise_norm:
ce_delta=+0.0727  top1=0.859

L[8,12,15] selective_oracle:
ce_delta=+0.1130  top1=0.829

L[8,12,15] multislot_headwise_norm:
ce_delta=+0.1460  top1=0.779
```

结论：

```text
两层中高层替换还可控；
加入 L8 后明显变差；
早层更敏感，不适合简单一起替换。
```

### 5.5 全层替换

结果文件：

```bash
runs/generation_effect_all_layers_with_oracle_b32_r50_n8.csv
```

结果：

```text
All layers selective_oracle:
ce_delta=+0.5914  top1=0.719

All layers multislot_headwise_norm:
ce_delta=+2.2602  top1=0.362
```

结论：

```text
当前方法不能直接全层替换；
即使没有 translation 误差，selective oracle 全层也明显退化；
说明全层只保留 50% blocks 会造成强误差累积。
```

### 5.6 self-consistent oracle + residual blend

结果文件：

```bash
runs/generation_effect_self_consistent_oracle_b32_r50_n16.csv
```

该实验比较：

```text
static:
    replacement 基于原始 receiver hidden trajectory 预计算。

self_consistent:
    在 forward hook 中用当前 hidden_states 重算 Q/K/V，
    再在 selected blocks 内做 selective attention。

alpha:
    residual blend:
    patched = alpha * replacement + (1 - alpha) * original_attention_output
```

关键结果：

```text
L[12,15] self_consistent alpha=1.00:
ce_delta=+0.0545  top1=0.878

L[12,15] self_consistent alpha=0.50:
ce_delta=+0.0069  top1=0.946

L[8,12,15] self_consistent alpha=1.00:
ce_delta=+0.1040  top1=0.823

L[8,12,15] self_consistent alpha=0.50:
ce_delta=+0.0103  top1=0.924

All layers self_consistent alpha=1.00:
ce_delta=+3.1036  top1=0.202

All layers self_consistent alpha=0.50:
ce_delta=+0.4592  top1=0.648
```

结论：

```text
逐层自洽对 2-3 层 patch 有小幅帮助；
residual blend 非常有效；
全层 self-consistent hard replacement 会严重偏离轨迹；
当前多层方案应使用 residual replacement，而不是 hard replacement。
```

## 6. 当前总判断

### 已经比较确定

```text
1. token-level route recovery 不适合 Qwen -> Llama 异构模型。

2. sender K / attention 可以作为 block evidence routing prior。

3. sparse anchor block selection 比 block mean score 更合理。

4. 当前 50% block budget 下，selected blocks 的单层 oracle 上限很高。

5. memory translation 是当前核心瓶颈。

6. 多 slot block memory 明显优于 1-slot block memory。

7. head-wise normalized translator 是当前最强的 translated memory 方法。

8. 单层 patch 生成代理指标可接受。

9. 两层中高层 patch 可控；三层开始变差；全层 hard replacement 不可行。

10. residual blend 是多层替换中最重要的稳定技巧。
```

### 不能再说的过时结论

```text
1. “mean-pooled receiver block K/V 是 oracle”是错误的。
   正确 oracle 是 selected blocks 内 receiver token-level selective recompute。

2. “block 内平均分数选块足够”是不充分的。
   当前应使用 sparse anchor / anchor_count 选块。

3. “单层 trans_cos 高就能保证生成质量”不成立。
   必须看 generation proxy 或真实 generation。

4. “全层替换可行”不成立。
   当前全层 hard replacement 明显退化。
```

## 7. 推荐下一步

### 7.1 translated memory 加 residual blend

已经在 oracle 上证明：

```text
patched = alpha * replacement + (1 - alpha) * original_attention_output
```

对多层稳定性极其重要。

下一步应把这个机制加到 translated memory 多层实验里，测试：

```text
alpha = 0.25, 0.5, 0.75, 1.0
layers = 12,15 / 8,12,15
method = multislot_headwise_norm
```

### 7.2 分层 budget

早层更敏感，不应和高层使用同样 budget。

建议：

```text
L0-L7: 不替换，或只替换很少，或 higher budget；
L8-L11: 谨慎替换；
L12-L15: 优先替换。
```

### 7.3 recent native KV + old translated memory

保留 receiver 最近窗口原生 K/V：

```text
recent window: receiver native K/V
old context: sender translated block memory
```

这可能比纯 translated memory 稳定很多。

### 7.4 更强 translator

继续加强：

```text
multi-slot memory
head-wise translator
nonlinear / low-rank + residual translator
per-layer adapter
output-equivalent loss
logit / CE proxy loss
```

### 7.5 真实生成评估

当前 generation proxy 还是 next-token logits 级别。

最终需要：

```text
greedy generation 对比
token overlap / edit distance
task-level QA / summarization metric
perplexity on larger sample
latency / memory saving
```

## 8. 常用命令

### sparse-anchor block translated memory, 64 samples

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复

/home/yezhe/data/miniconda3/envs/attnkv/bin/python block_translated_memory_recovery.py \
  --device cuda \
  --text_glob '/home/yezhe/demo/train/**/*.txt' \
  --num_samples 64 \
  --max_length 512 \
  --layers 8,12,15 \
  --block_size 32 \
  --budget_ratio 0.5 \
  --block_score_mode anchor_count \
  --anchor_tokens 64 \
  --pool_modes received,kxreceived \
  --value_pool_mode uniform \
  --epochs 30 \
  --lr 0.001 \
  --kv_loss_weight 0.1 \
  --output_loss_weight 1.0 \
  --csv runs/block_translated_memory_recovery_b32_r50_sparse_anchor_n64_v3.csv
```

### block memory method sweep, 64 samples

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复

/home/yezhe/data/miniconda3/envs/attnkv/bin/python block_memory_method_sweep.py \
  --device cuda \
  --text_glob '/home/yezhe/demo/train/**/*.txt' \
  --num_samples 64 \
  --max_length 512 \
  --layers 8,12,15 \
  --block_size 32 \
  --budget_ratio 0.5 \
  --block_score_mode anchor_count \
  --anchor_tokens 64 \
  --slots_per_block 4 \
  --methods baseline_1slot_linear,multislot_linear,multislot_mlp,multislot_mlp_norm,multislot_headwise_norm,multislot_mlp_norm_prior \
  --value_pool_mode uniform \
  --epochs 20 \
  --lr 0.001 \
  --kv_loss_weight 0.1 \
  --output_loss_weight 1.0 \
  --csv runs/block_memory_method_sweep_b32_r50_n64.csv
```

### single-layer generation proxy, 32 samples

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复

/home/yezhe/data/miniconda3/envs/attnkv/bin/python generation_effect_experiment.py \
  --device cuda \
  --text_glob '/home/yezhe/demo/train/**/*.txt' \
  --num_samples 32 \
  --max_length 512 \
  --layers 8,12,15 \
  --block_size 32 \
  --budget_ratio 0.5 \
  --block_score_mode anchor_count \
  --anchor_tokens 64 \
  --slots_per_block 4 \
  --methods baseline_1slot_linear,multislot_mlp,multislot_headwise_norm \
  --value_pool_mode uniform \
  --epochs 10 \
  --lr 0.001 \
  --kv_loss_weight 0.1 \
  --output_loss_weight 1.0 \
  --csv runs/generation_effect_b32_r50_n32.csv
```

### multi-layer with oracle

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复

/home/yezhe/data/miniconda3/envs/attnkv/bin/python generation_effect_multilayer_experiment.py \
  --device cuda \
  --text_glob '/home/yezhe/demo/train/**/*.txt' \
  --num_samples 16 \
  --max_length 512 \
  --layer_sets '12,15;8,12,15' \
  --block_size 32 \
  --budget_ratio 0.5 \
  --block_score_mode anchor_count \
  --anchor_tokens 64 \
  --slots_per_block 4 \
  --methods baseline_1slot_linear,multislot_headwise_norm \
  --include_selective_oracle \
  --value_pool_mode uniform \
  --epochs 10 \
  --lr 0.001 \
  --kv_loss_weight 0.1 \
  --output_loss_weight 1.0 \
  --csv runs/generation_effect_multilayer_with_oracle_b32_r50_n16.csv
```

### self-consistent oracle + residual blend

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复

/home/yezhe/data/miniconda3/envs/attnkv/bin/python generation_effect_self_consistent_oracle.py \
  --device cuda \
  --text_glob '/home/yezhe/demo/train/**/*.txt' \
  --num_samples 16 \
  --max_length 512 \
  --layer_sets '12,15;8,12,15;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' \
  --block_size 32 \
  --budget_ratio 0.5 \
  --block_score_mode anchor_count \
  --anchor_tokens 64 \
  --alphas 1.0,0.75,0.5 \
  --modes static,self_consistent \
  --csv runs/generation_effect_self_consistent_oracle_b32_r50_n16.csv
```

