Owner: ely
Status: active
Last reviewed: 2026-05-05

# 科研实验协议

## 1. 固定变量

第一阶段所有实验默认固定：

- dataset: 当前 LibriSpeech clean 训练路径和确定性 split。
- sample rate: `16 kHz`。
- channels: mono。
- backbone: 当前 `SEANet + EMA RVQ`。
- train clip: `2.0 s`，除非明确做 long-context 实验。
- frame rate: `50 Hz`。
- latent dim: `128`。
- codebook size: `1024`。
- loss recipe: `MS-STFT discriminator + feature matching + balancer`。
- optimizer / batch size / train steps: 从当前主力 config 继承。
- compute: 正式训练与效率指标以 Linux `4 x A100` 机器为准；macOS 开发机只作为编辑、轻量单测和 smoke 环境。

除非实验目标就是研究某变量，否则不要同时改：

- frame rate
- RVQ structure
- loss recipe
- data split
- front-end representation
- semantic distillation

## 2. 统一 reporting 字段

每个 run 必须记录：

| Field | Meaning |
|---|---|
| run_id | 唯一实验名 |
| variant | B0/B1/L-mamba/R-mamba/P-mamba 等 |
| insertion_point | waveform / latent / post_rvq / code_prior |
| context_model | none / lstm / tcn / transformer / mamba |
| bitrate_kbps_nominal | 由 frame rate、RVQ stages、codebook bits 计算 |
| entropy_bitrate_kbps_est | code prior 或 entropy 估计，若无则为空 |
| tokens_per_sec | `frame_rate * num_streams` |
| params | 模型参数量 |
| train_steps | 训练步数 |
| data_hours | 训练数据小时数 |
| seed | 随机种子 |
| step_time_seconds | 平均训练 step time |
| peak_gpu_memory | 峰值显存 |
| rtf | 推理 real-time factor |
| causal | 是否 causal |

没有这些字段时，不能把小幅收益写成确定结论。

## 3. Stage 0: Baseline freeze

目标：

- 确认当前 baseline 仍可跑通。
- 冻结后续对照的 anchor。

实验：

- `B0-12k`: 当前主力 `configs/ablation-adversarial-msstft-balanced.json`。
- `B0-8k / B0-4k / B0-2k`: 当前 bitrate ladder configs。
- `B1-no-mixer`: `seanet_lstm_layers=0`。

验收：

- smoke forward/backward 通过。
- 能导出 reconstruction。
- benchmark script 能对固定 manifest 评分。

执行建议：

- 在 macOS 上只要求轻量测试和 shape/smoke 能说明代码路径没有明显错误。
- 在 Linux A100 上重新跑训练 smoke，作为正式实验矩阵的起点。
- 第一阶段用 4 张 A100 并行跑 4 个单卡实验，例如 B0、B1、L-MA、R-MA；暂不把 DDP 放到关键路径。

## 4. Stage 1: Redundancy profile

目标：

测量不同表示层级中“可被上下文预测”的时间冗余。

表示层：

- waveform 或 early feature。
- downsampled latent。
- post-RVQ quantized embedding。
- RVQ codes。

方法：

- 从 frozen B0 checkpoint 导出 representations。
- 对每层训练 context probe：
  - contextless predictor
  - TCN/LSTM
  - Transformer
  - Mamba
- 对连续表示报 MSE、negative log likelihood proxy 或 predictive coding error。
- 对离散 codes 报 bits/code、NLL、perplexity。

预期：

- waveform predictability 最高，但不一定对应最好的 codec RD gain。
- code predictability 直接对应 entropy coding value。

关键分析：

> 哪一层最可预测，和哪一层最能把 predictability 转成压缩收益，可能不是同一层。

## 5. Stage 2: Insertion-point RD matrix

目标：

测量上下文模型插入不同层级后，对同码率重建质量的影响。

第一批矩阵：

| ID | insertion | model | Purpose |
|---|---|---|---|
| B0 | baseline | SkipLSTM | anchor |
| B1 | bottleneck | none | measure existing LSTM contribution |
| L-TCN | latent | TCN | local context baseline |
| L-TR | latent | Transformer | high-quality context baseline |
| L-MA | latent | Mamba | selective SSM latent test |
| R-TCN | post-RVQ | TCN | local code-context refiner |
| R-TR | post-RVQ | Transformer | strong refiner baseline |
| R-MA | post-RVQ | Mamba | selective SSM refiner |

第一轮 bitrates：

- `4 kbps`: low-bitrate main decision point。
- `12 kbps`: high-bitrate saturation check。

第二轮 bitrates：

- `2 / 4 / 8 / 12 kbps` for winning variants。

重建指标：

- actual bitrate kbps。
- compression ratio vs PCM16。
- multi-scale STFT。
- log spectral distance。
- STOI。
- SI-SDR。
- listening samples。
- ASR WER 或 semantic preservation。
- 可选 ViSQOL。

预期：

- latent context gain 小到中等，低码率更明显。
- post-RVQ refiner 在 2/4 kbps 最可能改善 artifact。
- 如果 waveform-proximal model 最强，当前 assumption 需要修正。

## 6. Stage 3: Code-prior entropy matrix

目标：

测量 discrete RVQ codes 的上下文冗余能否变成实际 bitstream 收益。

实验：

- 冻结 B0 或最佳 codec。
- 导出 train/val/test codes `[B, K, T]`。
- 训练：
  - unigram/contextless prior
  - ngram prior
  - LSTM/TCN prior
  - Transformer prior
  - Mamba prior

指标：

- bits/code。
- bits/frame。
- estimated entropy-coded bitrate。
- NLL by RVQ stage。
- context length sweep。
- encode/decode RTF if entropy coder is implemented。

预期：

- prior 能明显低于 contextless。
- Mamba 的质量可能接近 Transformer。
- Mamba 的核心优势应在长上下文 memory/latency，而非短上下文 NLL。

## 7. Stage 4: Mamba efficiency and streaming

目标：

把 Mamba 从“另一个上下文模型”中区分出来。

实验：

- 固定最有效插入点，比较 Transformer vs Mamba。
- context length: `2s / 8s / 32s / 120s`。
- 模式：offline non-causal、causal chunked、stateful streaming。

指标：

- peak memory vs context length。
- RTF vs context length。
- latency / chunk size。
- state size。
- quality or NLL degradation from offline to streaming。

预期：

- Transformer 在短 offline setting 可能质量略强。
- Mamba 应在长上下文和 stateful streaming 中显示更平稳的内存/延迟曲线。

## 8. 结论判定规则

### 支持主 hypothesis

满足至少两条：

- redundancy probe 显示 latent/code 层存在可利用上下文冗余。
- latent 或 post-RVQ context model 在同码率下稳定改善低码率重建。
- code prior 显著降低 bits/code。
- Mamba 在质量接近 Transformer 时显著降低长序列 memory/RTF。

### 不支持主 hypothesis

出现以下情况：

- matched baselines 消除 Mamba 收益。
- 所有收益只来自参数量增加。
- code predictability 不高，prior 无法降低 entropy。
- Mamba 在长序列效率上也没有优势。

### 可投稿但转向的结果

- 如果重建无明显收益，但 code prior 强：写成 speech codec token redundancy / entropy modeling paper。
- 如果 post-RVQ refiner 强：写成 context-refined neural speech codes。
- 如果 waveform-proximal 强：写成原始信号层上下文建模被低估，但需加强信号处理 baseline。

## 9. 推荐首周最小闭环

1. 新增 `TemporalMixer` 抽象。
2. 支持 `none / lstm`，先跑 B0/B1 smoke。
3. 接入 Mamba fallback，跑 latent shape test。
4. 实现 post-RVQ refiner 的接口，不改变 codes。
5. 写 representation/code dump 脚本。
6. 在 `4 kbps` 和 `12 kbps` 跑 20k short matrix。
7. 用固定 43 条 test manifest 出第一张机制表。
