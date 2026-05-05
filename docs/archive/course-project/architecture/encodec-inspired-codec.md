Owner: ely
Status: archived
Last reviewed: 2026-05-05

# Encodec-Inspired 架构说明

> Archived note: 本文档记录课程阶段从轻量 baseline 转向 SEANet+EMA-RVQ 的历史过程。当前活跃 baseline 说明见 [Current Codec Baseline](/Users/ely/workspace/research/audio/AudioCodec/docs/architecture/current-codec-baseline.md)。

## 1. 为什么从 baseline 转向

第一版 `Conv + RVQ + Conv` baseline 已经证明：

- 数据流、训练循环、日志与 checkpoint 机制是通的
- 模型可以学习语音内容与时序对齐

但试听结果显示，重建波形存在明显电流音与电音感，尚未达到“可用”级别。  
这说明当前 baseline 的问题不在训练是否收敛，而在模型上限偏低。

因此第二阶段不再继续微调小型 baseline，而是转向一个 `encodec-inspired` 路线：

- `SEANet` 风格 encoder / decoder
- `weight norm` 卷积
- encoder / decoder 末端 `LSTM`
- `EMA + k-means init` 的更成熟 `RVQ`

## 2. 当前实现范围

本仓库中的 `encodec-inspired` 路线不是完整复刻外部 `Encodec` 工程，而是提取对当前 speech codec baseline 最关键的设计：

- 非流式、离线语音重建
- `SEANet` 风格的多级下采样/上采样结构
- `Residual block + dilation`
- `EMA` 码本更新
- 首批训练批次的 `k-means` 初始化
- dead code replacement

早期阶段曾暂不引入：

- 对抗损失
- 多判别器
- 语言模型或熵模型
- 分布式训练特定逻辑
- 真正的流式/causal 处理

截至 2026-05-05，当前高保真 baseline 已经采用 `MS-STFT discriminator + feature matching + balancer` 训练路线；语言模型/熵模型和真正流式处理仍未进入主训练闭环。

## 3. 默认实验配置

推荐配置文件：`configs/encodec-inspired.json`

核心参数：

- sample rate: `16 kHz`
- channels: `mono`
- architecture: `seanet`
- encoder / decoder ratios: `[8, 5, 4, 2]`
- total stride: `320`
- frame rate: `50 Hz`
- latent dim: `128`
- codebook size: `1024`
- RVQ stages: `24`

名义码率约为：

`50 frame/s * 24 stages * log2(1024) = 12000 bit/s = 12 kbps`

这里有意把码率抬高到 `12 kbps` 附近，因为历史阶段的优先目标是先把重建音质做到可用。当前科研阶段继续保留 `12 kbps` 作为高码率饱和对照，同时重点观察 `2 / 4 kbps` 低码率区间。

## 4. 与 baseline 的关系

- `configs/baseline.json` 仍然保留，作为课程最小闭环与消融参考
- `configs/ablation-mel-loss.json` 仍然保留，作为 baseline 的感知损失消融
- `configs/encodec-inspired.json` 作为下一阶段主力训练配置

报告叙事可以按这条线组织：

1. 先展示轻量 baseline 的可训练性与局限
2. 指出“可懂内容但不可用音质”的失败模式
3. 说明为何引入 `SEANet + EMA RVQ`
4. 再展示新的主结果与 `MP3` 对比
