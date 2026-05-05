Owner: ely
Status: active
Last reviewed: 2026-05-05

# 序列建模插入点架构

## 1. 当前 baseline 数据流

当前主力 codec 数据流：

```text
waveform [B, 1, samples]
  -> SEANet encoder
  -> downsampled latent [B, D, T]
  -> EMA RVQ
  -> codes [B, K, T]
  -> quantized embedding [B, D, T]
  -> SEANet decoder
  -> reconstruction [B, 1, samples]
```

关键默认值：

- sample rate: `16 kHz`
- hop length: `320`
- frame rate: `50 Hz`
- latent dim: `128`
- codebook size: `1024`
- code bits per RVQ stage: `10 bits/frame`
- nominal bitrate: `50 * K * 10 bits/s`

当前 `SEANetEncoder` 和 `SEANetDecoder` 中已有 `SkipLSTM`。它们是第一批最自然的替换或对照位置。

## 2. 插入点定义

### P0: waveform-proximal context

位置：

```text
waveform / early high-rate feature -> context module -> encoder stack
```

用途：

- 验证用户直觉：原始或近原始信号层的时间冗余是否最值得上下文建模。

风险：

- 序列长度高，成本大。
- 冗余以局部 signal correlation 为主，conv/TCN 可能更合适。

默认优先级：

- 第二阶段对照，不作为第一天主线。

### P1: latent bottleneck context

位置：

```text
SEANet encoder conv stack -> bottleneck context module -> latent projection/RVQ
```

以及 decoder 对称位置：

```text
quantized embedding -> decoder input projection -> context module -> decoder upsampling stack
```

用途：

- 验证下采样后连续 latent 是否是上下文建模的自然起点。

第一批变体：

- `B0`: baseline `SkipLSTM`
- `B1`: no mixer
- `L-lstm`: matched LSTM
- `L-tcn`: matched TCN
- `L-transformer`: lightweight local/global Transformer
- `L-mamba`: Mamba temporal block

### P2: post-RVQ embedding refiner

位置：

```text
latent -> RVQ -> quantized embedding -> context refiner -> decoder
```

用途：

- 在不改变 RVQ payload 的情况下，测试 decoder 是否能通过 code context 改善重建。

约束：

- `codes` 是唯一传输信息。
- refiner 的输入只能来自 `quantized embedding` 和允许的 causal/non-causal context。
- `decode(codes)` 必须能复现同一路径。

第一批变体：

- `R-none`: no refiner
- `R-tcn`: matched TCN refiner
- `R-transformer`: matched Transformer refiner
- `R-mamba`: Mamba refiner

### P3: code-prior context

位置：

```text
frozen codec codes [B, K, T] -> prior model -> next-code / masked-code prediction
```

用途：

- 直接估计 code sequence 的可压缩冗余。
- 连接 entropy coding 和 speech token modeling。

建模方式：

- 时间主序：`(t, q)` flatten 成 token sequence。
- 多流建模：每个 RVQ stage 一个 stream，模型共享或分头预测。
- 分 stage 报告 NLL，避免后级 RVQ 被前级支配。

第一批变体：

- contextless unigram prior
- ngram / Markov prior
- LSTM/TCN prior
- Transformer prior
- Mamba prior

## 3. TemporalMixer 接口要求

所有插入模块应统一遵守：

```text
input:  [B, C, T]
output: [B, C, T]
```

内部可以转换为 `[B, T, C]`，但外部接口不变。

推荐抽象：

```text
TemporalMixer(kind="none" | "lstm" | "tcn" | "transformer" | "mamba")
```

通用要求：

- residual connection：`y = x + f(norm(x))`。
- 支持 causal/non-causal 配置。
- 记录 parameter count。
- 不改变 frame rate、latent dim、RVQ stages、codebook size。

## 4. 不同模块的研究角色

### TCN / Conv

角色：

- 强局部 signal baseline。
- 用于判断 waveform-proximal 层是否其实只需要局部上下文。

### LSTM

角色：

- 当前 baseline 已使用的 recurrent 对照。
- 用于判断 Mamba 是否只是替代已有 recurrent context。

### Transformer

角色：

- 上下文建模质量强基线。
- 用于判断 Mamba 是否在质量上有必要。

注意：

- 短片段下 Transformer 可能很强，不能用短序列质量证明 Mamba 独特。

### Mamba

角色：

- selective SSM 对照。
- 主要假设是长序列效率、状态递推、低内存 streaming。

注意：

- 不应预设 Mamba 在所有插入点重建质量优于 Transformer。

## 5. Bitstream 边界

必须严格区分：

```text
encoder/decoder context module:
  可以改善同 nominal bitrate 重建质量
  不能直接降低 transmitted code payload

code prior:
  可以降低 entropy-coded bitrate
  不直接改善 frozen codec 的 reconstruction

frame-rate / RVQ-stage 改动:
  可以降低 token/sec 或 nominal kbps
  但会改变主变量，不能和 mixer 插入混在同一个结论里
```

任何实验如果引入额外 side information，都必须单独 accounting，不能算作同码率比较。
