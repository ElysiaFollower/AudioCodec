# AudioCodec 5-Minute Slides Draft

> 目标：5 分钟内讲清楚项目的研究假设、核心方法、关键结果与局限。  
> 原则：每页只保留一个主论点，口头解释负责补足细节，不在页面上堆积过多文字。  
> 风格：正文内容按正式 presentation 文案来写，同时保留足够多可直接口述的句子，以降低现场卡顿风险。

---

## Slide 1. Title + Research Assumption

### 页面标题
**Speech-Specific Neural Codec: Can Domain Specialization Beat Classical Compression?**

### 页面主内容
- 任务：仅针对 `speech` 设计神经音频编解码器
- 主线：`encoder -> RVQ bottleneck -> decoder`
- 核心问题：在相近甚至更低的码率下，是否能比传统 codec 保留更高保真度？
- 研究假设：
  - `speech` 只是音频空间中的一个特定子分布
  - 当任务范围从“所有音频”收缩到“仅 speech”时，数据熵下降、可预测性上升
  - 因此，针对 speech 专门训练的 neural codec 在理论上应当具备超越通用传统 codec 的潜力

### 口头展开
这项工作的起点不是“神经网络一定更强”，而是一个更具体的假设：传统 codec 必须服务于广义音频，而我们的任务只关心 speech。  
一旦任务域被收缩到 speech，数据分布本身就变得更窄、更规律、更可预测。  
所以从理论上讲，专门针对 speech 训练的 neural codec，应该能够把传统通用 codec 无法充分利用的结构性冗余进一步压缩掉。

### 页面素材建议
- 左侧：一句话研究问题
- 右侧：speech domain / general audio domain 的示意图

---

## Slide 2. Minimal Codec Idea: Encoder + RVQ + Decoder

### 页面标题
**Minimal Neural Codec Pipeline**

### 页面主内容
- 输入：`16 kHz` 单声道 speech waveform
- 编码：encoder 将波形映射到低速率 latent sequence
- 压缩：`RVQ` 把连续 latent 变成离散码字序列
- 解码：decoder 从离散 token 重建波形

### 口头展开
最小系统其实可以非常直观地理解为一个自编码器。  
输入的原始语音先被 encoder 压缩成一串更短的表示，这就相当于编码；然后 decoder 再根据这串表示恢复出目标波形，这就是解码。  
真正让它变成“codec”而不是普通 autoencoder 的关键，在于中间不是随便保留一个连续隐向量，而是要把它变成可以存储、可以传输的离散 token。

### RVQ 解释
- `RVQ = Residual Vector Quantization`
- 第 1 个码本先近似原始 latent，得到 `r1`
- 后续码本逐级量化残差：`x-r1 -> r2`, `x-r1-r2 -> r3`, ...
- 最终重建时把各级码字相加：`r1 + r2 + r3 + ...`

### 口头展开
这里最值得讲清楚的是 RVQ。  
它不是只用一个码本一次性硬量化，而是逐级逼近残差。  
所以你可以把它理解成：每多加一个 quantizer，就多给系统一次“补细节”的机会。  
这也正是后面我们能够只通过调整 `num_quantizers` 就构造出 `2 / 4 / 8 / 12 kbps` 多码率版本的原因。

### 页面素材建议
- 一张最简结构图：`waveform -> encoder -> narrow bottleneck -> RVQ -> decoder -> waveform`
- 中间 bottleneck 要画窄，突出“压缩”

---

## Slide 3. From Baseline to Final System

### 页面标题
**Why the Baseline Was Not Enough**

### 页面主内容
- 初始基线：`Conv + RVQ + Decoder`
- 问题：能重建内容，但有明显电流音、数字沙砾感
- 最终系统引入三类关键改进：
  - 更强骨干：`SEANet + EMA RVQ`
  - 更强训练目标：`adversarial + feature matching`
  - 更稳定训练机制：`Balancer + Adam(0.5, 0.9)`

### 口头展开
项目早期并不是一开始就成功。  
最初的 `Conv + RVQ + Decoder` 可以学会“说了什么”，但始终带着明显的电子音和高频破碎。  
真正让系统跨过“能重建内容”和“能高保真重建”这条分界线的，不只是更长训练，而是训练目标本身的升级。  
最后有效的 recipe 包括更强的 `SEANet` 主干、更稳定的 `EMA RVQ`、以及更接近 Encodec/Audiocraft 的 adversarial 训练机制。

### 技术重点
- `SEANet`：更强的局部建模与时序表达能力
- `EMA RVQ`：更稳定的码本更新与量化训练
- `MS-STFT discriminator`：把“听起来是否自然”纳入优化目标
- `Balancer`：避免不同 loss 梯度量级失衡导致静音塌缩

### 页面素材建议
- 左侧：baseline 架构
- 右侧：final architecture
- 中间用箭头标“from reconstructable to high-fidelity”

---

## Slide 4. Experimental Setup

### 页面标题
**Benchmark Design**

### 页面主内容
- 数据：`speech-only`, `16 kHz`, deterministic held-out split
- Neural codec operating points:
  - `Neural-12k`
  - `Neural-8k`
  - `Neural-4k`
  - `Neural-2k`
- Classical codecs:
  - `Opus`
  - `MP3`
  - `AAC`
  - `FLAC` as lossless anchor
- 评价维度：
  - 压缩成本：`actual bitrate`, `compression ratio`
  - 重建质量：`STOI`, `LSD`, `MS-STFT`
  - 主观试听：代表性样本人工抽查

### 口头展开
这里有一个 benchmark 口径必须明确：  
我们不按“目标码率”下结论，而是按“实际输出码率”下结论。  
因为传统 lossy codec 虽然可以接受目标码率，但最终是否命中、命中多少，取决于各自的 rate control 机制。  
所以真正公平的比较方式，是统一记录压缩后的真实文件大小，再去比较质量。

### 页面素材建议
- 一张小表：neural ladder `2 / 4 / 8 / 12 kbps`
- 一张小表：metrics list

---

## Slide 5. Main Results

### 页面标题
**Main Result: Neural Codec Wins in the Ultra-Low-Rate Regime**

### 页面主内容
- `Neural-2k` 的实际码率约为 `2.00 kbps`
- 对应传统 codec 的“2 kbps 目标点”实际并不在 `2 kbps`
  - `Opus-2k -> 5.45 kbps`
  - `MP3-2k -> 8.18 kbps`
  - `AAC-2k -> 10.67 kbps`
- `Neural-2k` 在更低实际码率下仍保持高可懂度与高保真度
- `Neural-8k / 12k` 与主流传统 codec 达到可比高保真水平
- `Opus` 在 `8-12 kbps` 区间仍然是强基线，不能轻视

### 口头展开
最关键的结论集中在超低码率区间。  
`Neural-2k` 是所有系统里唯一真正工作在 `2 kbps` 附近的点，而且仍然保持了很强的语音可懂度和很好的主观保真度。  
相比之下，传统 codec 在名义上的 `2 kbps` operating point 下，实际码率往往高得多。  
这意味着我们的优势不是“在相同实际码率下勉强打平”，而是“在显著更低的实际码率下仍然保住了质量”。

### 补充说明
- `Neural-8k` 在主观试听里通常比 `Opus-8k` 更清亮
- 但客观指标并不总是完全一致
- 这说明指标只是对听感的拟合，不是绝对 ground truth

### 页面素材建议
- 主图：rate-distortion 曲线
- 主表：代表性 operating points 对比表
- 可选：加入音频播放按钮

---

## Slide 6. Limitation and Future Work

### 页面标题
**Limitation: Timbre Drift at 2 kbps**

### 页面主内容
- 主要局限不在“听不清”，而在“音色偶发漂移”
- 代表性现象：
  - 内容基本正确
  - 说话人后半段声线发生变化
- 直接看原始频谱与重建频谱，差异并不总是显著
- 但 `absolute log-spectral difference` 可以把这类局部误差更清楚地显示出来

### 口头展开
`Neural-2k` 的主要失败模式并不是内容崩坏，而是 timbre drift。  
也就是说，模型大多数时候能把“说了什么”保留下来，但在少数困难样本上，说话人的音色会发生偏移。  
这个现象很值得注意，因为它提示我们：问题不是语言内容建模失败，而是 speaker color / timbre fidelity 还没有被完全约束住。

### Future Work
- 如果某种频谱差异表示能够稳定对应到“声线漂移”这一语义
- 那么它就可能被升格为新的 loss 或辅助训练目标
- 也就是说，这个 case study 不只是失败展示，也给出了后续优化方向

### 页面结论
**Conclusion:**  
For speech-specific compression, a properly trained neural codec can outperform classical codecs in the ultra-low-rate regime, while still leaving identifiable research space in timbre preservation.

### 页面素材建议
- 直接放 `case_timbre_drift.svg`
- 可选：一对漂移样本的音频播放按钮

---

## 备注：推荐的 5 分钟时间分配

- Slide 1: `40s`
- Slide 2: `60s`
- Slide 3: `60s`
- Slide 4: `35s`
- Slide 5: `90s`
- Slide 6: `55s`

总计约 `340s`，也就是 `5min 40s` 左右。  
正式汇报时建议：
- 压缩 Slide 2 的 RVQ 解释到 `45s`
- Slide 4 只讲 benchmark 口径，不展开全部指标
- 把总时长控制在 `5min ~ 5min 15s`

---

## 备注：后续转成正式 slides 时的设计原则

- 不把整段讲稿全部堆到页面上
- 页面只保留关键词、公式、图和结论句
- `draft.md` 更像 presenter script，`html slides` 再负责视觉压缩
- 第三、四节可以嵌入代表性音频按钮，增强直观性
