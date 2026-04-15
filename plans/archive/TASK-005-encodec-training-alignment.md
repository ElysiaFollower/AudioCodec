Owner: ely
Status: archived
Last reviewed: 2026-04-15

# TASK-005 Encodec Training Alignment

## Archive Note

本计划已完成核心目标：通过对齐 `Encodec / Audiocraft` 的训练目标与训练机制，把当前 `SEANet + EMA RVQ` 路线从“单样本可学住、全数据音质不佳”推进到“全数据训练可达到高保真重建”。最终起关键作用的不是单独某一个 loss，而是以下因素的组合：

- `MS-STFT discriminator`
- generator adversarial loss
- feature matching
- 本地 `Balancer`
- generator optimizer 与 reference 对齐
- 足够长的训练步数

阶段性结果表明：

- 单样本 overfit 已能达到接近真假难分
- 全数据训练在约 `40000 step` 时已能实现高保真 speech 重建
- 当前剩余工作的重点已不再是“把模型训通”，而是进入与传统 codec 的系统 benchmark 对比

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

## Outcome Summary

- 已新增 `adversarial.py` 和 `balancer.py`
- `train.py` 已支持 generator/discriminator 双优化器与 `resume`
- 已完成 `ablation-adversarial-msstft-balanced` 路线验证
- 当前主结论：训练目标与训练机制对齐后，模型具备高保真重建能力

## Follow-up

后续工作转入新的 benchmark 计划，重点是：

1. 调用 `MP3` 等传统 codec
2. 统一导出 neural codec 与传统 codec 的重建结果
3. 统计压缩率、码率与保真度
4. 形成课程项目的最终对比结论
