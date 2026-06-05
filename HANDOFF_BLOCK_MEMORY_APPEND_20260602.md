
## 8. 最新补充：block-level translated memory recovery 实验

脚本：

```bash
block_translated_memory_recovery.py
```

这个实验是在前面 evidence region / selective recompute 路线上的进一步版本。前面的主要设定是：

```text
sender 负责选 evidence region；
receiver 在候选区域内用自己的历史 K/V 重新计算 attention output。
```

最新实验测试更激进的设定：

```text
sender 已经完成 prefill，持有 K_s / V_s / attention；
sender 从 K_s / attention 中选出重要 block/span；
sender 将 block-level K/V memory 通过轻量 T_k / T_v 映射到 receiver 可读空间；
receiver 不完整计算历史 K/V，只用自己的 Q attend translated block memory；
最终近似 receiver full attention output。
```

### 8.1 当前实现方式

核心流程：

```text
1. 对齐 sender / receiver 的 token span。
2. 以 receiver block 为单位，找到与该 block 字符 span 重叠的 sender tokens。
3. 对 sender K_s 做 block pooling，得到 sender block K memory。
4. 对 sender V_s 做 block pooling，得到 sender block V memory。
5. 选择 top blocks 作为候选 block memory。
6. 训练轻量线性映射：
   T_k: sender block K -> receiver-readable block K
   T_v: sender block V -> receiver-readable block V
7. receiver 使用自己的 Q_r attend translated K/V memory，得到 translated output。
8. 用 receiver full attention output 作为训练和评估目标。
```

注意，最新版已经修正两个关键点：

```text
K memory 用 routing_pool_mode 池化；
V memory 用 value_pool_mode 池化。
```

也就是说，K 负责路由，应使用 routing prior 权重；V 负责内容，应使用内容保真权重。当前主实验用的是：

```text
routing_pool_mode = uniform / received / kxreceived
value_pool_mode = uniform
```

这样避免把 V 的内容表示也被 routing saliency 过度扭曲。

### 8.2 oracle 指标修正

早期版本中 `oracle_block_cosine` 实际只是 receiver K/V 的 mean-pooled block memory 结果，不是真正 oracle。因此最新版已改名并新增指标：

```text
mean_block_cosine:
    receiver mean-pooled block K/V memory baseline。
    这不是 oracle，只是 block memory 压缩基线。

selective_oracle_cosine:
    真正 oracle。
    在同样选中的 receiver blocks 内，使用 receiver token-level K/V 和 gold attention 做 selective recompute。
    它表示候选 block 区域本身的理论上限。
```

这个区分很重要。现在可以判断瓶颈来自两部分：

```text
selective_oracle_cosine 高 -> 说明选中的 evidence region 本身足够好；
translated_cosine 低于 selective_oracle_cosine -> 说明瓶颈在 sender memory -> receiver-readable memory 的翻译质量。
```

### 8.3 运行命令

```bash
cd /home/yezhe/恢复注意力图代码/注意力图恢复

/home/yezhe/data/miniconda3/envs/attnkv/bin/python block_translated_memory_recovery.py \
  --device cuda \
  --text_glob '/home/yezhe/demo/train/**/*.txt' \
  --num_samples 8 \
  --max_length 512 \
  --layers 8,12,15 \
  --block_size 32 \
  --budget_ratio 0.5 \
  --pool_modes uniform,received,kxreceived \
  --value_pool_mode uniform \
  --epochs 30 \
  --lr 0.001 \
  --kv_loss_weight 0.1 \
  --output_loss_weight 1.0 \
  --csv runs/block_translated_memory_recovery_b32_r50_n8_v2.csv
```

结果文件：

```bash
runs/block_translated_memory_recovery_b32_r50_n8_v2.csv
```

### 8.4 当前结果

配置：

```text
num_samples = 8
max_length = 512
layers = 8,12,15
block_size = 32
budget_ratio = 0.5
value_pool_mode = uniform
```

结果：

```text
L08 uniform:
selected_ratio=0.438
translated_cosine=0.7403
selective_oracle_cosine=0.8861
mean_block_cosine=0.6356

L08 received:
selected_ratio=0.438
translated_cosine=0.7391
selective_oracle_cosine=0.9375
mean_block_cosine=0.6595

L08 kxreceived:
selected_ratio=0.438
translated_cosine=0.7410
selective_oracle_cosine=0.9375
mean_block_cosine=0.6595

L12 uniform:
selected_ratio=0.438
translated_cosine=0.7054
selective_oracle_cosine=0.9078
mean_block_cosine=0.6247

L12 received:
selected_ratio=0.438
translated_cosine=0.7016
selective_oracle_cosine=0.9498
mean_block_cosine=0.6299

L12 kxreceived:
selected_ratio=0.438
translated_cosine=0.7016
selective_oracle_cosine=0.9498
mean_block_cosine=0.6299

L15 uniform:
selected_ratio=0.438
translated_cosine=0.7319
selective_oracle_cosine=0.9210
mean_block_cosine=0.6761

L15 received:
selected_ratio=0.438
translated_cosine=0.7405
selective_oracle_cosine=0.9769
mean_block_cosine=0.6928

L15 kxreceived:
selected_ratio=0.438
translated_cosine=0.7407
selective_oracle_cosine=0.9769
mean_block_cosine=0.6928
```

### 8.5 当前判断

这个实验说明：

```text
1. block-level evidence region 的 oracle 上限很高，尤其 received / kxreceived 路由下，
   selective_oracle_cosine 可达到约 0.94~0.98。

2. translated block memory 已经能达到约 0.70~0.74 的 output cosine，
   并且明显高于 mean-pooled receiver block memory baseline 的部分结果。

3. 当前主要瓶颈不再是 evidence region 是否能覆盖 receiver 需要的信息，
   而是 sender block memory 如何更好地翻译成 receiver-readable memory。
```

因此下一步不应该只继续扩大 block 或 token ratio，而应该重点改进 memory translation：

```text
更强的 T_k / T_v：低秩 + 非线性、per-layer adapter、head-specific adapter；
更合理的 V pooling：uniform、V norm、received、或内容保真 learned pooling；
训练目标更偏 output recovery，而不是单纯拟合 receiver mean-pooled K/V；
引入 block 内多 slot memory，而不是每个 block 只压成一个 memory；
保留 recent receiver native K/V，与 translated old memory 混合。
```
