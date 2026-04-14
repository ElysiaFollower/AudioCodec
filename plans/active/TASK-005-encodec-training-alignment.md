Owner: ely
Status: active
Last reviewed: 2026-04-14

# TASK-005 Encodec Training Alignment

## Goal

在保留当前 `SEANet + EMA RVQ` 主干的前提下，优先对齐 `Encodec / Audiocraft` 的训练目标与训练机制，解决“单样本 overfit 可行，但全数据训练重建仍有明显电流音和数字沙砾感”的问题。

## Background

- 当前 `encodec-inspired` 路线已经证明：
  - 单样本 overfit 可以接近原声
  - 全数据训练结果仍明显劣于参考 `Encodec`
- 已完成的分析表明，主要缺口更像在训练 recipe，而不是基础前向/反向链路：
  - 缺少 `MS-STFT discriminator`
  - 缺少 `feature matching`
  - 缺少梯度级 `Balancer`
  - 当前主 loss 与参考 run 不一致

## Hypothesis

当前大问题不是“模型不会表达”，而是“训练目标没有足够好地约束真实听感”。因此，优先补齐感知相关训练机制，比继续调整纯重建 loss 更有希望改善全数据训练音质。

## In Scope

- 在当前训练框架中引入对抗训练所需的最小基础设施
- 优先实现 `MS-STFT discriminator`
- 实现 generator 侧的 adversarial loss 与 feature matching loss
- 为多损失训练准备一个最小可用的 balancing 机制
- 通过短程 overfit 与小规模全数据训练验证是否改善听感

## Out Of Scope

- 一次性完全复刻 `Audiocraft CompressionSolver`
- 分布式训练 / 多卡训练
- 熵编码、LM 压缩或流式推理
- 课程报告最终写作

## Execution Strategy

### Phase 1: Training Objective Alignment

先补齐最关键的训练目标链路：

1. `MS-STFT discriminator`
2. generator adversarial loss
3. feature matching loss
4. 训练脚本中的双优化器流程

Current status:

- `src/audiocodec/adversarial.py` 已新增最小 `MS-STFT discriminator`
- `train.py` 已支持可选 generator/discriminator 双优化器训练
- 新实验配置：[configs/ablation-adversarial-msstft.json](/Users/ely/workspace/research/audio/AudioCodec/configs/ablation-adversarial-msstft.json)
- 旧配置默认不启用 adversarial training

### Phase 2: Controlled Ablation

在不改模型主干的情况下做最小消融：

1. 只加 discriminator
2. discriminator + feature matching
3. 视情况引入简化版 balancer

### Phase 3: If Needed

如果训练目标补齐后仍不够，再考虑结构侧继续对齐：

1. `RVQ` 语义进一步向 `audiocraft` 靠齐
2. `4 x 2048` 量化器几何
3. `SEANet` 宽度 / `true_skip` / `pad_mode`

## Acceptance Criteria

- 训练脚本支持 generator/discriminator 双优化器训练
- 新目标在单样本 overfit 上不会明显破坏现有可训练性
- 小规模全数据训练听感相较当前 `encodec-inspired` 明显减少电流音 / 数字沙砾感
- 变更过程保留清晰配置与实验路径，便于后续归因

## Experimental Entry Points

单样本 overfit 验机：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/ablation-adversarial-msstft.json \
  --output-dir artifacts/debug-overfit-adversarial-msstft \
  --steps 4000 \
  --device cuda \
  --overfit-example-path /absolute/path/to/example.flac \
  --tensorboard
```

小规模全数据训练：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/ablation-adversarial-msstft.json \
  --output-dir artifacts/linux-adversarial-msstft \
  --device cuda \
  --tensorboard
```

## Immediate Next Step

先用 `ablation-adversarial-msstft` 路线做一次单样本 overfit 和一次小规模全数据训练，对比当前 `encodec-inspired` 是否显著减轻电流音和高频沙砾感。
