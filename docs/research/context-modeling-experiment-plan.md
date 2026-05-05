Owner: ely
Status: active
Last reviewed: 2026-05-05

# 上下文建模实现与实验计划

## 目的

本文档把 `docs/idea.md` 和 `docs/research/context-modeling-intake.md` 细化到实现层。目标不是一次性跑完论文矩阵，而是让下一阶段代码实现能稳定回答三个问题：

1. 时间冗余在哪些表示层可测；
2. 哪些层级的上下文建模能转成 codec 收益；
3. Mamba 是否比 matched TCN / LSTM / Transformer 更值得用。

## 当前工程事实

- 稳定 codec baseline 是 `SEANetRVQCodec`，路径为 `waveform -> encoder -> latent -> EMA RVQ -> quantized embedding -> decoder`。
- `BaseCodecModel.forward()` 已返回 `latent`、`quantized`、`codes` 和 `reconstruction`，适合做 representation export。
- `evals/scripts/export_neural_codec.py` 已支持 `--save-codes`，但还不能保存 latent / quantized embedding，也不能统一写 context experiment metadata。
- 当前 SEANet encoder/decoder 已有 `seanet_lstm_layers=2`，其 `SkipLSTM` 位于 downsampled bottleneck 附近。第一轮 context 实验应保持该 baseline 不变，不能把 baseline 自带 LSTM 误报为新增收益。
- `evals/scripts/score_outputs.py` 已覆盖 reconstruction 指标：`actual_bitrate_kbps`、`compression_ratio_vs_pcm16`、`si_sdr_db`、`log_spectral_distance`、`multi_scale_stft`、`stoi`。

## 什么叫插入时间序列建模

这里的“插入”不是说原 codec 完全逐帧独立。当前 SEANet 已经有卷积感受野和 bottleneck `SkipLSTM`。本项目的插入定义更窄：

> 在固定的 codec 表示边界上，把原本会直接传给下一步的序列张量 `x [B, C, T]`，替换为同形状的 `x_ctx = x + TemporalMixer(x)`，然后让后续模块只消费 `x_ctx`。

其中：

- `B` 是 batch；
- `C` 是当前表示维度，例如 latent dim 或 feature channels；
- `T` 是该表示层的时间 frame 数；
- 一个 frame 是 `x[:, :, t]`，表示当前层在时间位置 `t` 的向量，而不是 raw waveform sample；
- `TemporalMixer` 可以是 TCN、LSTM、Transformer 或 Mamba，但输入输出必须保持 `[B, C, T]`，不得改变 frame rate、latent dim、RVQ stage 数或 codebook size。

工程接口必须满足：

```text
TemporalContext.forward(x: Tensor[B, C, T]) -> Tensor[B, C, T]
```

不同模型族只是在内部如何沿 `T` 混合信息不同：

- TCN：保持 `[B, C, T]`，用 causal 或 non-causal 1D convolution 跨 frame 混合；
- LSTM：转成 `[T, B, C]` 或 `[B, T, C]`，沿时间递推，再转回 `[B, C, T]`；
- Transformer：转成 `[B, T, C]`，用 self-attention 跨 frame 混合，再转回；
- Mamba：转成 `[B, T, C]`，用 SSM/selective scan 跨 frame 混合，再转回。

默认实现应使用 residual wrapper：

```text
x_ctx = x + output_projection(mixer(input_projection(norm(x))))
```

`identity` baseline 定义为 `x_ctx = x`。这样新增路径可以和 baseline 做形状、payload 和训练稳定性对照。

## 插入点的精确定义

### Latent pre-RVQ

原路径：

```text
waveform -> encoder -> latent -> RVQ -> quantized -> decoder
```

插入后：

```text
waveform -> encoder -> latent -> TemporalMixer(latent) -> RVQ -> quantized -> decoder
```

含义：

- `TemporalMixer` 在量化前跨 latent frames 交换信息；
- RVQ codes 会改变，因此它测试的是 rate-distortion / fidelity 是否改善；
- payload 仍只由 RVQ codes 决定，不允许额外发送 mixer hidden state。

### Post-RVQ embedding

原路径：

```text
codes -> RVQ decode -> quantized -> decoder
```

插入后：

```text
codes -> RVQ decode -> quantized -> TemporalMixer(quantized) -> decoder
```

含义：

- transmitter 仍只发送 RVQ codes；
- receiver 通过共享模型参数从 codes 恢复 `quantized`，再运行同一个 `TemporalMixer`；
- 该实验只能证明 nominal payload 不变时 reconstruction fidelity 是否提升；
- 如果任何额外 per-frame side information 被写入 manifest 或 bitstream，该 run 无效。

### Code prior

原 codec reconstruction 路径不变。Code prior 不生成新的 waveform，也不改变 decoder 输入。

它只学习离散序列概率：

```text
codes [B, K, T] -> Prior -> p(code_{k,t} | previous codes)
```

含义：

- `T` 仍是 codec frame；
- `K` 是 RVQ stage；
- prior 可以按 stage 单独建模，也可以把 `(t, k)` 展平成 token 序列，但必须记录 token ordering；
- 该实验只报告 bits-per-code、estimated entropy bitrate 和 token efficiency，不报告 reconstruction fidelity gain。

### Early feature

Early feature 插入需要把 SEANet encoder 显式拆成两段：

```text
waveform -> encoder_prefix -> early_feature -> TemporalMixer(early_feature) -> encoder_suffix -> latent -> RVQ -> decoder
```

含义：

- `early_feature [B, C_s, T_s]` 的 `T_s` 通常高于 latent 的 `T`；
- 该层最接近用户关于 waveform-proximal context 的假设；
- 它需要重构 SEANet stage 边界，工程风险高，排在 representation export、code prior、latent 和 post-RVQ 之后。

## Causality 与可比较性

- `context.causal=true`：`x_ctx[:, :, t]` 只能依赖 `x[:, :, <=t]`，适合 streaming claim。
- `context.causal=false`：`x_ctx[:, :, t]` 可以依赖整段 utterance，适合 offline fidelity / RD claim。
- causal 和 non-causal 结果不能放在同一行直接比较，必须在结果表中记录 `context_causal`。
- 当前 SEANet 使用对称/reflect padding，整体更接近 offline baseline。若要写 streaming claim，必须单独建立 causal baseline。

下面这些不算合法的“插入时间序列建模”：

- 改变 sample rate、frame rate、codebook size、RVQ stage 数或 loss recipe；
- 只把 encoder/decoder 整体加宽加深，却没有固定边界的 `x -> x_ctx` 对照；
- post-RVQ refiner 发送额外 side channel；
- 用 code prior 的 entropy 改善来宣称 reconstruction fidelity 改善；
- 只比较 Mamba 和 no-context，而没有 matched TCN/LSTM/Transformer baseline。

## 最小实验矩阵

第一轮只用 `configs/ablation-adversarial-msstft-balanced-4kbps.json` 作为主实现锚点。代码稳定后再复制到 `2 / 8 / 12 kbps` ladder。

| 阶段 | 目的 | 插入层级 | 模型族 / baseline | 主要输出 | 是否训练 codec |
| --- | --- | --- | --- | --- | --- |
| E0 export | 固定可复现实验样本和表示导出 | latent / quantized / codes | current baseline only | representation manifest | no |
| E1 diagnostics | 测时间冗余是否可见 | latent / quantized / codes | autocorrelation, nearest previous frame, unigram / previous-frame code entropy | redundancy metrics JSONL | no |
| E2 code-prior | 测 codes 能否转成 entropy 收益 | RVQ codes / code prior | unigram, previous-frame, TCN, LSTM, Transformer, Mamba | bits-per-code, estimated entropy bitrate | no codec retrain |
| E3 latent context | 测量化前 context 是否改善 RD | latent pre-RVQ | identity, matched TCN, LSTM, Transformer, Mamba | reconstruction metrics + params/latency | yes |
| E4 post-RVQ context | 测无 side-channel refiner 是否只改善保真率 | quantized embedding pre-decoder | identity, matched TCN, LSTM, Transformer, Mamba | reconstruction metrics + no payload change proof | yes |
| E5 early feature | 测 waveform-proximal 假设 | early SEANet feature | matched conv/TCN first, then Transformer/Mamba | reconstruction metrics + cost | yes, after E3/E4 |

`E5` 不作为第一批代码实现入口，因为它需要把 SEANet encoder 拆成可插入 stage。它必须进入论文级矩阵，但应在 `E0-E4` 工具链稳定后做。

## 实现边界

### Codec context module

新增配置建议放入 `CodecExperimentConfig`：

```text
context.enabled: bool
context.insertion_point: none | latent_pre_rvq | post_rvq_embedding
context.family: identity | tcn | lstm | transformer | mamba
context.causal: bool
context.hidden_dim: int
context.num_layers: int
context.kernel_size: int
context.dropout: float
```

所有 codec 内 context module 都必须是同形状 residual mixer。它不是新的 encoder、decoder 或 quantizer，也不能单独改变 payload accounting。

新增模块建议：

- `src/audiocodec/models/context.py`：统一 `[B, C, T] -> [B, C, T]` 接口。
- `IdentityContext`：用于验证配置路径不改变 baseline。
- `TCNContext`：第一简单 baseline，参数量应和后续模型族记录在结果表。
- `LSTMContext`：recurrent baseline，不能和 SEANet 自带 `SkipLSTM` 混淆。
- `TransformerContext`：Transformer-family baseline。
- `MambaContext`：若依赖未安装，配置为 `family=mamba` 时必须给出清晰错误；不能静默退化成 identity。

插入语义：

- `latent_pre_rvq`：`latent_context = context(latent)`，RVQ 消费 `latent_context`，decoder 消费 quantized embedding。
- `post_rvq_embedding`：RVQ codes 仍来自原 latent，`quantized_context = context(rvq_output.quantized)`，decoder 消费 `quantized_context`；不得向 bitstream 增加任何 side channel。
- `none` / `identity` 必须在相同 seed、相同配置下保持 baseline 行为可解释。

### Representation export

新增或扩展 eval 脚本，建议输出到 `evals/outputs/context-modeling/<run_id>/`：

```text
manifest.jsonl
representations/
  <id>.codes.pt
  <id>.latent.pt
  <id>.quantized.pt
reconstructions/
  <id>.wav
run.json
```

每个 manifest row 至少包含：

- `id`, `source_path`, `duration_seconds`, `num_samples`, `sample_rate`, `channels`
- `checkpoint_path`, `checkpoint_step`, `config_path`, `codec_label`
- `frame_rate`, `hop_length`, `num_frames`, `latent_dim`
- `num_quantizers`, `codebook_size`, `bits_per_code`
- `nominal_bitrate_kbps`, `rvq_payload_bits`, `rvq_payload_bytes`
- `codes_path`, `latent_path`, `quantized_path`, `reconstruction_path`

### Code-prior experiments

Code-prior 不改 reconstruction codec。它消费 frozen `codes.pt`，输出 prior metrics：

```text
evals/outputs/context-priors/<run_id>/
  train_metrics.jsonl
  val_metrics.jsonl
  summary.json
  config.json
```

核心公式：

- `nominal_bitrate_kbps = frame_rate * num_quantizers * bits_per_code / 1000`
- `estimated_entropy_bitrate_kbps = frame_rate * sum(stage_bits_per_code) / 1000`
- `entropy_savings_ratio = 1 - estimated_entropy_bitrate_kbps / nominal_bitrate_kbps`

最小 prior baseline 顺序：

1. unigram per stage；
2. previous-frame / local Markov prior；
3. TCN；
4. LSTM；
5. Transformer；
6. Mamba。

如果 Mamba 依赖未固定，先实现到 Transformer，并把 Mamba 标为 blocked，而不是替换研究问题。

## 结果表字段

统一结果表建议写到 `evals/outputs/context-modeling/results.jsonl` 和 `summary.csv`。每一行代表一个 run 或一个 scored aggregate：

```text
run_id
stage
insertion_point
context_family
context_causal
config_path
checkpoint_path
checkpoint_step
dataset_split
codec_label
sample_rate
frame_rate
num_quantizers
codebook_size
nominal_bitrate_kbps
estimated_entropy_bitrate_kbps
actual_bitrate_kbps
compression_ratio_vs_pcm16
si_sdr_db
log_spectral_distance
multi_scale_stft
stoi
prior_val_bits_per_code
params_total
params_context
latency_ms_per_second_audio
peak_memory_mb
notes
```

报告规则：

- Reconstruction 指标只用于 `E3/E4` 或 baseline export；code-prior 只能报告 entropy / token efficiency。
- `actual_bitrate_kbps` 对 neural RVQ baseline 仍等于 RVQ nominal payload，除非真实 entropy coding 已实现。
- `estimated_entropy_bitrate_kbps` 是 prior 估计，不等于真实文件大小。
- `post_rvq_embedding` 的 payload 必须和 identity / baseline 相同，否则实验无效。

## 数据与命令

本地 macOS 只负责 sanity，不负责正式结论：

```bash
./scripts/harness-check.sh
git diff --check
conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help
conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v
```

Linux A100 smoke 继续使用当前基线命令：

```bash
PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda
```

正式实验默认沿用配置中的 LibriSpeech speech 数据路径；如果机器路径不同，必须通过 `--dataset-root` 覆盖并写入 run metadata。

Benchmark 样本继续使用 `evals/data/manifests/test.jsonl`。传统 codec 和 neural codec 结果继续通过 `evals/scripts/score_outputs.py` 汇总，context-specific 字段由新增 aggregator 补齐。

## 下一阶段提交顺序

1. Commit 4：实现 representation export 和 result schema，不训练新模型。
2. Commit 5：实现 code-prior entropy baselines，先到 unigram / previous-frame / TCN / Transformer。
3. Commit 6：实现 `latent_pre_rvq` context module，先跑 identity / TCN / Transformer smoke。
4. Commit 7：补 LSTM / Mamba family，并固定 Mamba 依赖或明确阻塞。
5. Commit 8：实现 `post_rvq_embedding` refiner，并加入 no-side-channel 检查。
6. Commit 9：拆 SEANet early feature 插入点，进入 waveform-proximal 对照。

## 明确非目标

- 不改变 frame rate、codebook size、loss recipe、front-end 或 RVQ payload accounting。
- 不引入 semantic distillation、音乐数据集或通用音频扩展。
- 不把 Mamba 作为唯一主线。
- 不把 estimated entropy bitrate、nominal bitrate、actual file bitrate 和 reconstruction quality 混成一个指标。
- 不在没有 matched TCN/LSTM/Transformer baseline 前写论文级强 claim。
