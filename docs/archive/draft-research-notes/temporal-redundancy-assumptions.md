Owner: ely
Status: active
Last reviewed: 2026-05-05

# 时间冗余 Assumptions

## 1. 总假设

本项目的总 assumption 是：

> speech codec 的表示链中存在不同类型的时间冗余；上下文序列建模可以在某些层级利用这些冗余，从而改善同码率质量、降低 entropy-coded bitrate，或提高长音频 token modeling 效率。

这里的关键是“不同类型”。

原始 waveform 的时间冗余最容易肉眼理解，但它未必是 Mamba/Transformer 最值得介入的层级。原始波形的强相关性大量来自局部平滑、周期、相位和短时频谱结构；这些结构往往已经被卷积、滤波器组、STFT/LPC 或 SEANet 前端有效利用。

相比之下，latent 和 code 层的冗余更少、更抽象，但更接近信息选择问题。上下文模型在这些层级的收益可能更容易转化为压缩或 token modeling 收益。

## 2. 需要验证的 Assumptions

### A0: 上下文建模收益来自时间冗余，而不是只来自模型变大

验证方式：

- 每个插入点都要有 `none / LSTM or TCN / Transformer / Mamba` 对照。
- 报告 parameter count、RTF、训练 step time 和显存。
- 至少主结论需要 matched-size 或 matched-compute baseline。

预期：

- 如果所有上下文模型都涨，而 Mamba 不特殊，结论应写成“context modeling helps”，不是“Mamba helps”。
- 如果只有更大模型涨，且 matched baseline 不涨，则 assumption 不成立。

### A1: waveform 层冗余最大，但不一定最适合 selective model

直觉来源：

- waveform 相邻采样点高度相关。
- 传统 codec 和信号处理长期利用时域/频域冗余。

风险修正：

- 这类冗余大多是局部、连续、近似平移不变。
- `Conv / TCN / STFT-like` bias 可能比 Mamba/Transformer 更有效率。
- 在 16 kHz sample-level 直接做全局上下文建模，计算上不经济，也可能把容量浪费在 phase tracking。

验证方式：

- 不先重写 waveform front-end。
- 做轻量 waveform-proximal 对照：在 encoder 早期或高时间分辨率 feature 上加入 TCN/Transformer/Mamba。
- 同时记录训练速度、显存和 quality gain。

预期：

- waveform-proximal Mamba 不应是最强主结果。
- 如果它显著优于 conv/TCN，项目结论需要改写为“原始信号层 selective modeling 被低估”。

### A2: downsampled latent 层保留了更适合上下文模型的信息冗余

直觉来源：

- SEANet encoder 已把 waveform 压成 `50 Hz` latent timeline。
- 每个 latent frame 已经聚合局部 waveform context。
- 这时剩余冗余更可能对应 phonetic event、prosody、speaker/timbre continuity 等跨帧结构。

验证方式：

- `B0`: 当前 SEANet-LSTM baseline。
- `B1`: no-mixer baseline，关闭 `SkipLSTM`。
- `L-lstm / L-tcn / L-transformer / L-mamba`: 在 latent bottleneck 加上下文 mixer。
- 做 bitrate sweep，优先 `2 / 4 / 12 kbps`。

预期：

- latent 上下文模型在低码率比高码率更可能有收益。
- 如果 `B1` 与 `B0` 几乎无差别，说明当前 bottleneck LSTM 贡献有限，Mamba 替换空间也有限。

### A3: post-RVQ 层是同码率重建收益的高价值位置

直觉来源：

- RVQ codes 是实际传输信息。
- 低码率下每帧 code 信息不足，decoder 需要利用邻域和长程 context 补足细节。
- post-RVQ refiner 不改变 bitstream，只改变 decoder 如何使用已传输 codes。

验证方式：

- 在 `quantized embedding [B, D, T]` 后、decoder 前插入 refiner。
- 对照 `none / LSTM or TCN / Transformer / Mamba`。
- 确认 `encode()` 输出 codes 不变，`decode(codes)` 不需要额外 side channel。

预期：

- 2/4 kbps 下 post-RVQ context refiner 最可能改善 MS-STFT、LSD、听感 artifact。
- SI-SDR 可能不稳定，因为听感和点对点波形误差并不完全一致。

### A4: code-prior 层最直接对应实际压缩率

直觉来源：

- 如果 RVQ code sequence 可被上下文预测，entropy coding 可以用更少 bits 表示同一串 codes。
- 这不改变 reconstruction，但改变实际 bitstream 的可压缩性。

验证方式：

- 冻结 codec，导出 `[K, T]` RVQ code streams。
- 训练 contextless、ngram、LSTM/TCN、Transformer、Mamba prior。
- 报告 bits/code、per-stage NLL、estimated entropy-coded bitrate。
- 做 context length sweep，例如 `1s / 4s / 16s / 60s`。

预期：

- code-prior 是 Mamba 最可能展现长序列优势的位置。
- 如果只用 2 秒 clip，Mamba 相比 Transformer/LSTM 的优势可能不明显。

### A5: Mamba 的独立价值是效率与流式状态，而不是必然更高质量

验证方式：

- 对同一插入点比较 Transformer 和 Mamba。
- 短片段看 quality / NLL。
- 长片段看 RTF、peak memory、latency、streaming state size。
- 做 causal/stateful decoding 补充实验。

预期：

- Mamba 质量可能接近 Transformer，而不是显著超过。
- 真正强项应体现在长音频、低显存、常数状态递推和无限流式处理。

## 3. 三类压缩收益必须分开

### 同码率质量收益

定义：

- nominal kbps 不变。
- 重建质量更好。

来源：

- latent/post-RVQ context modeling 改善 decoder 使用 codes 的方式。

注意：

- 这不是 bitstream 更短，而是 rate-distortion 曲线更好。

### 实际 entropy-coded bitrate 收益

定义：

- RVQ codes 不变或质量不变。
- code prior 让 entropy coding 使用更少 bits。

来源：

- code-level context model 降低 bits/code。

注意：

- 这不一定减少 downstream LLM token 数。

### token efficiency 收益

定义：

- tokens/sec 更低，或固定上下文窗口覆盖更长音频。

来源：

- 更低 frame rate、层级 token、semantic/acoustic disentanglement，或减少 RVQ streams。

注意：

- 当前 `50 Hz * K streams` 的 token 密度对 spoken LM 仍偏高。

## 4. 预期结果形态

理想但保守的结果不是“每个指标都涨”，而是一张清晰机制表：

| Layer | Redundancy visibility | Best model family | Expected gain |
|---|---:|---|---|
| waveform-proximal | highest | conv / TCN | weak selective-model gain |
| downsampled latent | medium | TCN / Transformer / Mamba | modest low-bitrate RD gain |
| post-RVQ embedding | medium-low | context refiner | strongest reconstruction gain at low bitrate |
| RVQ codes | symbolic | Transformer / Mamba prior | strongest entropy gain |

如果实验得到相反结论，也有价值，但必须按 assumption 被证伪来写，而不是硬解释成 Mamba 成功。
