Owner: ely
Status: active
Last reviewed: 2026-05-05

# 上下文建模研究收集

## 目的

本文档是 `TASK-008` 的研究收集主表。它不决定最终实现路线，只把外部论文和代码映射到当前问题：

> Neural speech codec 已经具备局部时序建模后，语音中是否仍存在可利用的长程时间冗余；如果存在，它在 `waveform -> downsampled latent -> RVQ embedding/codes -> code prior` 哪个层级最容易转化为真实收益？

每条材料都必须区分收益类型：保真率、rate-distortion、entropy、token efficiency、推理效率。

## 第一轮结论

- `EnCodec` 和 `LMCodec` 给 code-prior / entropy 方向提供了最直接证据：上下文模型不一定改善 waveform reconstruction，但可以降低实际传输 bits 或减少需要发送的 token 层级。
- `SoundStream`、`EnCodec`、`DAC` 和 `AudioDec` 说明当前仓库的 `SEANet + RVQ` 主干是合理实验基底；它们更多是 codec baseline 和实现参考，而不是单独证明 Mamba 或某个插入层级。
- `Convolutional Transformer`、`BigCodec` 这类 speech coding 工作支持 early/latent context 值得验证，但它们通常同时改变模型规模、结构、量化策略或训练 recipe，不能直接当作本项目结论。
- `SpeechTokenizer`、`Mimi/Moshi` 主要支持 codec token hierarchy 和 speech LM token efficiency 视角；它们会改变 semantic/acoustic token 目标，第一阶段只能作为指标和 tokenizer 设计参考。
- Mamba 当前应作为上下文模型族候选，而不是第一轮唯一方向。实现计划至少需要 `TCN` 或 matched conv、`LSTM/GRU`、`Transformer`、`Mamba/SSM` 四类对照。
- 主题重定后，新增 focused intake：
  - `docs/research/long-range-redundancy-intake.md`
  - 该文档补充 long-range Transformer speech coding、codec-token language modeling、time-invariant / multi-scale / low-frame-rate / dynamic-frame-rate tokenizer，以及为什么长程架构尚未成为通用 codec 标配。

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
| R11 | Ultra Low-Bitrate Speech Coding with Pretrained Transformers | low-bitrate speech codec + pretrained Transformer embeddings | early/latent semantic conditioning | low-bitrate fidelity, long-range dependency | [paper/project](https://research.google/pubs/ultra-low-bitrate-speech-coding-with-pretrained-transformers/) | 直接指出 convolutional/recurrent codec 有效感受野限制压缩效率；支持 long-range redundancy 主题 | 引入外部 pretrained speech embeddings，变量不同；不能直接作为当前 SEANet/RVQ 层级结论 |
| R12 | TiCodec | fewer-token neural speech codec | time-invariant utterance code + frame-level code | token efficiency, downstream TTS | [paper](https://arxiv.org/abs/2310.00014) | 直接处理 utterance-level time-invariant information，说明长程不变信息可减少 frame-level tokens | 改变 tokenizer 目标和结构；偏 TTS token efficiency，不是固定 baseline 下的 context insertion |
| R13 | SNAC | multi-scale neural audio codec | RVQ at different temporal resolutions | token efficiency, multi-scale compression | [paper](https://arxiv.org/abs/2410.14411), [code](https://github.com/hubertsiuzdak/snac) | 通过 variable frame-rate quantizers 利用多时间尺度结构 | 改变 frame rate / RVQ accounting；只作为后续 tokenization redesign 参考 |
| R14 | WavTokenizer | efficient acoustic discrete codec tokenizer | low token rate + contextual windows | token efficiency, audio LM utility | [paper](https://arxiv.org/abs/2408.16532), [code](https://github.com/jishengpeng/WavTokenizer) | 说明 extended contextual windows 和低 token rate 对 audio LM 有用 | 变量很多：VQ space、attention、discriminator、token rate；不能替代 go/no-go diagnostics |
| R15 | Stable Codec / TAAE | Transformer-based speech codec | Transformer codec architecture | low-bitrate fidelity, tokenization for speech pipelines | [project](https://stability-ai.github.io/stable-codec-demo/), [code](https://github.com/Stability-AI/stable-codec) | 证明大 Transformer codec 可达 400/700 bps 低码率 speech coding | 整体 codec scaling，不是层级归因；工程依赖重，如 FlashAttention / no CPU inference |
| R16 | AudioLM / SoundStorm / Moshi / LongCat | codec-token language modeling / speech LLM tokenizer | codec tokens / semantic-acoustic tokens | long-term consistency, token efficiency, streaming latency | [AudioLM](https://arxiv.org/abs/2209.03143), [SoundStorm](https://arxiv.org/abs/2305.09636), [Moshi](https://arxiv.org/abs/2410.00037), [LongCat](https://arxiv.org/abs/2510.15227) | 说明长程结构确实存在于 speech/audio token streams，并能被 LM 利用 | 多数是 generation/dialogue/tokenizer 目标，不是 compression RD/entropy 层级归因 |
| R17 | Single-Codec | disentangled speech codec / tokenizer | time-invariant embedding + phonetic discrete sequence | low-bitrate speech coding, token efficiency | [paper](https://arxiv.org/abs/2406.07422) | 直接把全局不变信息和 temporal modeling 引入 speech codec/tokenizer，报告 304 bps single-sequence codec | 改变 tokenizer/codebook 结构，不能直接替代固定 SEANet/RVQ 表示链上的层级归因实验 |
| R18 | Temporally Flexible Coding / CodecSlime | variable / dynamic frame-rate neural speech codec | frame allocation / token rate schedule | temporal redundancy compression, low-bitrate fidelity | [TFC](https://arxiv.org/abs/2505.16845), [CodecSlime](https://arxiv.org/abs/2506.21074) | 直接把固定帧率 codec 的 speech temporal redundancy 浪费作为问题，说明 token/frame allocation 是强相关路线 | 主要收益来自 frame-rate redesign，不是 fixed-frame baseline 上的 long-context insertion；应作为 go/no-go 后的后续分支 |

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
- 尚未定义 code-prior baseline 的 token ordering：stage-major、time-major、coarse-to-fine conditioning 哪个先做。
- 尚未定义 go/no-go 阈值：long/full context 相比 local 至少带来多少 bits-per-code 或 predictability improvement 才进入 codec context 训练。
- 尚未定义如何把 steady-state segment redundancy 映射到后续分支：如果收益集中在静音、长元音或缓慢变化区间，应考虑 dynamic/variable frame-rate redesign，而不是只扩大 context window。

## 下一步

基于本文档已进入实现层计划：

- `docs/research/context-modeling-experiment-plan.md`
- `docs/research/long-range-redundancy-intake.md`

后续代码实现从该计划出发：

1. 实现 representation export，导出长片段 / full utterance 的 latent、quantized embedding 和 RVQ codes。
2. 定义 result schema 和 manifest metadata，显式记录 `context_scope`、`context_window_seconds`、`context_window_frames`。
3. 先跑 E0-E2 go/no-go diagnostics，再决定是否进入 codec context training 或 dynamic frame-rate / tokenization redesign。
