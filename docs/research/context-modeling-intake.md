Owner: ely
Status: active
Last reviewed: 2026-05-05

# 上下文建模研究收集

## 目的

本文档是 `TASK-008` 的研究收集主表。它不决定最终实现路线，只把外部论文和代码映射到当前问题：

> 在 `waveform -> downsampled latent -> RVQ embedding/codes -> code prior` 这条表示链上，上下文建模在哪一层最能把 speech 时间冗余转化为真实收益？

每条材料都必须区分收益类型：保真率、rate-distortion、entropy、token efficiency、推理效率。

## 第一轮结论

- `EnCodec` 和 `LMCodec` 给 code-prior / entropy 方向提供了最直接证据：上下文模型不一定改善 waveform reconstruction，但可以降低实际传输 bits 或减少需要发送的 token 层级。
- `SoundStream`、`EnCodec`、`DAC` 和 `AudioDec` 说明当前仓库的 `SEANet + RVQ` 主干是合理实验基底；它们更多是 codec baseline 和实现参考，而不是单独证明 Mamba 或某个插入层级。
- `Convolutional Transformer`、`BigCodec` 这类 speech coding 工作支持 early/latent context 值得验证，但它们通常同时改变模型规模、结构、量化策略或训练 recipe，不能直接当作本项目结论。
- `SpeechTokenizer`、`Mimi/Moshi` 主要支持 codec token hierarchy 和 speech LM token efficiency 视角；它们会改变 semantic/acoustic token 目标，第一阶段只能作为指标和 tokenizer 设计参考。
- Mamba 当前应作为上下文模型族候选，而不是第一轮唯一方向。实现计划至少需要 `TCN` 或 matched conv、`LSTM/GRU`、`Transformer`、`Mamba/SSM` 四类对照。

## 候选材料表

| ID | 材料 | 类型 | 主要插入层级 | 对应收益 | 主链接 / 代码 | 对当前项目的可复用点 | 差距与 matched baseline 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | SoundStream: An End-to-End Neural Audio Codec | neural audio codec | early conv / latent / RVQ embedding | rate-distortion, streaming efficiency | [paper](https://research.google/pubs/soundstream-an-end-to-end-neural-audio-codec/) | RVQ、causal encoder/decoder、variable bitrate 作为 codec baseline 参考 | 没有直接回答上下文插入层级；若复用思想，必须和当前 SEANet/RVQ 固定变量分开 |
| R02 | EnCodec: High Fidelity Neural Audio Compression | neural audio codec + entropy model | latent/RVQ codes/code prior | fidelity, rate-distortion, entropy | [paper](https://arxiv.org/abs/2210.13438), [code](https://github.com/facebookresearch/encodec) | 与当前 SEANet/RVQ baseline 最近；Transformer entropy model 可作为 code-prior 参考 | entropy 收益不能和 nominal bitrate 混写；Transformer prior 应独立于 reconstruction codec 对照 |
| R03 | DAC: High-Fidelity Audio Compression with Improved RVQGAN | neural audio codec implementation | latent/RVQ embedding | fidelity, rate-distortion | [paper](https://arxiv.org/abs/2306.06546), [code](https://github.com/descriptinc/descript-audio-codec) | RVQGAN 训练、codebook 稳定性和公开工程实现可参考 | 主要面向通用音频；不是 speech-only，也不证明时间上下文模型位置 |
| R04 | AudioDec | streaming neural audio codec | early/latent/post-quantized decoder | fidelity, latency, streaming efficiency | [paper](https://arxiv.org/abs/2306.15564), [code](https://github.com/facebookresearch/AudioDec) | 低延迟 streaming codec 的评测字段和工程拆分可参考 | 关注 streaming codec 系统，不是表示层级消融；不能替代 matched context baseline |
| R05 | LMCodec: A Low Bitrate Speech Codec with Causal Transformer Models | speech codec + causal Transformer | RVQ codes/code prior | entropy, token efficiency, low bitrate speech fidelity | [paper/project](https://research.google/pubs/lmcodec-a-low-bitrate-speech-codec-with-causal-transformer-models/) | 最直接支持 code-prior 方向：用 causal Transformer 建模 codec token 条件分布 | 首轮未找到官方代码；若借鉴，需要重新实现并和简单 AR/TCN/LSTM prior 对照 |
| R06 | Convolutional Transformer for Neural Speech Coding | neural speech coding | waveform-proximal / latent encoder-decoder | rate-distortion, fidelity | [paper](https://research.google/pubs/convolutional-transformer-for-neural-speech-coding/) | 支持在 speech codec encoder/decoder 中引入 temporal attention/context 的早期证据 | 模型结构与当前 SEANet/RVQ 不同；需要 matched conv/TCN baseline 防止把容量误判为上下文层级收益 |
| R07 | BigCodec | low-bitrate neural speech codec | early/latent sequence modeling | low bitrate fidelity, efficiency | [paper](https://arxiv.org/abs/2409.05377), [code](https://github.com/Aria-K-Alethia/BigCodec) | 低码率 speech codec 目标接近；可参考其长程依赖和低码率报告方式 | 可能同时改变规模、tokenizer 和训练策略；第一阶段只提取实验问题，不迁移完整架构 |
| R08 | SpeechTokenizer | RVQ speech tokenizer | RVQ code hierarchy | token efficiency, speech LM utility | [paper](https://arxiv.org/abs/2308.16692), [code](https://github.com/ZhangXInFD/SpeechTokenizer) | 可帮助定义 semantic/acoustic code 层级指标和 speech token 下游视角 | 语义 tokenizer 会引入额外目标；当前项目默认不加 semantic distillation |
| R09 | Mimi / Moshi | streaming speech/audio tokenizer for dialogue model | codec tokens / token stream prior | token efficiency, latency, streaming | [paper](https://arxiv.org/abs/2410.00037), [code](https://github.com/kyutai-labs/moshi) | 可参考低帧率 token stream、streaming latency 和 speech LM benchmark 字段 | 改变 frame rate 和语义目标会违反当前 baseline 固定变量；只作为后续扩展参考 |
| R10 | Mamba / Mamba-2 | sequence model family | candidate context module at any layer | inference efficiency, long-context efficiency | [Mamba paper](https://arxiv.org/abs/2312.00752), [Mamba-2 paper](https://arxiv.org/abs/2405.21060), [code](https://github.com/state-spaces/mamba) | 提供 SSM/Mamba family 实现入口；适合和 Transformer 比较 long-context memory/latency | 不是 codec 证据；必须先证明上下文层级有效，再评价 Mamba 是否更值得用 |

## 按插入层级整理

### Waveform / early feature

初步候选是 `Convolutional Transformer`、`BigCodec`、`SoundStream`、`AudioDec`。这些材料说明 speech/audio codec 的 encoder/decoder 侧可以利用 temporal context，但它们通常和模型容量、causality、stride、loss recipe 绑定在一起。

当前实现含义：

- 不能直接把 early-context 收益归因给 Mamba。
- 最小对照应包含当前 SEANet baseline、matched wider/deeper conv、TCN 或 local causal conv，再加 Transformer/Mamba。
- 指标重点是同 nominal bitrate 下的 reconstruction quality 和训练/推理成本。

### Downsampled latent

`EnCodec`、`DAC`、`BigCodec` 对 latent/RVQ bottleneck 的设计最有参考价值。这个层级序列长度为当前项目默认 `50 Hz`，比 waveform 更适合做层级对照。

当前实现含义：

- latent context 是最适合先做工程插桩的位置之一，因为不改变 payload accounting。
- 必须记录 context module 的参数量、感受野、latency 和是否 causal。
- 需要和 matched TCN/Transformer 对照，避免只比较 Mamba vs no-context。

### Post-RVQ embedding

第一轮材料中没有强证据表明 post-RVQ embedding refiner 单独优于 latent context 或 code prior。相关 codec 的 decoder 本身会在 quantized embedding 上建模局部上下文，但这通常是 decoder 基础能力。

当前实现含义：

- post-RVQ refiner 可以作为一个候选层级，但不能引入额外 side channel。
- 如果实现，只能消费 quantized embedding / codes 内可恢复的信息。
- 报告中要区分 nominal bitrate 不变的 fidelity gain 与真实 compression gain。

### RVQ codes / code prior

`EnCodec` 和 `LMCodec` 是 code-prior 方向的第一批核心材料。它们支持把离散 codes 的可预测性转化成 entropy-coded bitrate 或 token transmission savings。

当前实现含义：

- code-prior 实验应先离线导出当前 RVQ codes，再训练 prior 估计 cross-entropy / bits-per-code。
- 不应把 code prior 的 entropy 收益写成 codec reconstruction fidelity 收益。
- 最小 prior baseline 应包含 unigram / previous-frame local prior、TCN/LSTM、Transformer、Mamba。

## 第一阶段缺口

- 尚未系统收集 neural speech codec entropy coding 论文，尤其是非 Transformer 的 autoregressive / recurrent prior。
- 尚未验证哪些候选有可复现实验配置和开源训练脚本。
- 尚未确定本项目应先测 redundancy diagnostics，还是直接实现最小 context module。
- 尚未定义统一结果表字段；Commit 3 应把 `quality`、`nominal_bitrate`、`estimated_entropy_bitrate`、`tokens_per_second`、`latency`、`memory` 分开。

## 下一步

基于本文档已进入实现层计划：

- `docs/research/context-modeling-experiment-plan.md`

后续代码实现从该计划出发：

1. 定义最小实验矩阵，不超过一条主 codec baseline 加两到三个上下文插入层级。
2. 定义结果采集 schema，明确哪些指标来自现有 `evals/scripts`，哪些需要新增。
3. 写实现计划，列出模块边界、配置字段、导出 codes 的命令和 benchmark 汇总流程。
