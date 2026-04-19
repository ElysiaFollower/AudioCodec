// Owner: ely
// Status: draft
// Last reviewed: 2026-04-19

#set page(margin: (x: 22mm, y: 20mm))
#set par(justify: true, leading: 0.72em)
#set text(lang: "zh", size: 10.5pt)
#set heading(numbering: "1.")
#show figure.caption: set text(size: 9pt)
#show heading.where(level: 1): set text(size: 13pt, weight: "bold")
#show heading.where(level: 2): set text(size: 11pt, weight: "bold")

#let paper-table(..args) = block[
  #set text(size: 8.8pt)
  #set par(justify: false, leading: 0.62em)
  #table(
    stroke: none,
    inset: (x: 5pt, y: 3pt),
    ..args,
  )
]

#align(center)[
  #text(size: 18pt, weight: "bold")[AudioCodec 项目报告]
  #linebreak()
  #text(size: 11pt)[从课程基线到高保真 Speech Neural Codec 的设计、调试与 Benchmark 对比]
]

#v(1.2em)

#strong[摘要]

本项目围绕 speech neural codec 的任务展开，目标是在仅面向 `16 kHz` 单声道语音的设定下，实现一个可训练、可评测、可解释的神经音频编解码系统，并在统一 benchmark 上与传统音频压缩算法比较压缩率与重建质量。项目早期从 `Conv + RVQ + Decoder` 的最小闭环出发，快速验证了离散 token 化与波形重建链路的可行性，但同时暴露出明显的电流音和数字伪影，说明仅靠轻量卷积骨干与重建损失无法获得可用级的听感。随后系统升级为 `SEANet + EMA RVQ`，并最终通过引入 `MS-STFT discriminator`、feature matching、梯度 `Balancer` 与更接近 Encodec/Audiocraft 的优化器配方，把训练结果推进到高保真语音重建。

在最终系统中，我们保持骨干和训练目标不变，仅通过调整 `RVQ` stage 数量构建 `2 / 4 / 8 / 12 kbps` 的 neural bitrate ladder，并将其与 `Opus / MP3 / AAC / FLAC` 在统一测试集上进行比较。实验结果表明，神经编解码器在极低实际码率区域展示出明显优势：`neural-2k` 的实际码率约为 `2.00 kbps`，平均 `STOI` 达到 `0.886`，说明在极限压缩下仍保持了很强的语音可懂度。与此同时，传统 codec 在低目标码率设置下往往无法命中目标 operating point，例如 `opus-2k`、`mp3-2k` 和 `aac-2k` 的实际码率分别约为 `5.45 kbps`、`8.18 kbps` 和 `10.67 kbps`。不过，实验也表明 `Opus` 在 `8-12 kbps` 区间仍是很强的基线，而 `neural-2k` 在个别样本上会出现声线漂移等 timbre artifact。总体而言，本项目验证了 neural codec 在低码率 speech compression 上的实际竞争力，同时也说明高保真结果依赖于合理的训练目标与优化机制，而不仅仅是更强的编码器结构。

= Project Introduction

== Problem Definition

本项目聚焦于面向语音场景的端到端神经音频编解码。具体而言，输入被统一限制为 `16 kHz`、单声道 speech waveform，系统需要学习一条从连续波形到离散 token 表示、再从离散 token 回到重建波形的完整压缩链路。项目最终并不只停留在“模型能训通”，而是要回答一个更实质的问题：在相同或相近的实际压缩预算下，learned codec 是否能够在 speech 任务上达到优于传统 codec 的压缩效率和重建质量。

这里需要特别说明的是，项目最初的口径中包含了“VAE + RVQ”这一表述，但随着实现细节的落地，最终系统更准确地说属于 `RVQ autoencoder / VQ-VAE family`，而不是严格意义上的 variational autoencoder。这一表述调整并非概念修辞，而是源于实现层面的事实：最终系统不包含显式的 KL 正则项，核心设计重点是离散表征、残差量化与感知型训练目标。

== Working Hypothesis

本项目在正式实现前，其实有一个非常明确的理论预期。我们并不把传统音频压缩算法的局限简单理解为“人工设计不够聪明”，而更倾向于把它理解为一个分布层面的约束问题：主流传统 codec 需要服务尽可能广的音频分布，因此它们必须在语音、音乐、环境声、瞬态噪声等多种统计结构之间折中，最终形成一种面向广义音频的通用压缩策略。这样的通用性具有工程价值，但也意味着 codec 无法把全部建模能力集中在单一子分布上。

如果把任务范围刻意限制在 speech 领域，这个问题的统计结构就会发生变化。与“所有音频”相比，语音信号的生成机制更窄、更规则，也更具可预测性：说话人的发声器官、共振峰结构、语音节律、基频变化和短时频谱形态都明显受限于人类语言系统的物理与生理规律。用信息论语言描述，这意味着任务分布的有效熵更低，而可被模型利用的结构性冗余更多。因此，我们在项目开始前就预期：如果神经编解码器只在 speech 数据分布上训练，并且学习的是 speech-specific 的压缩表示，那么在理论上它应当有机会获得优于通用传统 codec 的压缩效率。

这个假设还有一个重要的边界。即使这种“专有分布上的理论上限”真实存在，也不自动意味着任何 neural codec 都会在训练中学到更优的压缩策略。理论可压缩性上限，和具体模型是否通过有限数据、有限优化过程、有限参数预算去逼近该上限，是两回事。本项目的技术路线可以被理解为对这一假设的经验检验：我们先验证 speech-only learned codec 是否真的能形成有竞争力的低码率 operating points，再通过 benchmark 检验这种优势是否在实际结果中体现出来。

== Project Scope

项目的范围被刻意收紧到 speech-only、offline reconstruction 的场景。这种约束并非技术保守，而是为了保证课程周期内能够完成一个真正闭环的系统。在本次实现中，系统不追求通用音频能力，不覆盖音乐、环境声或多说话人混合场景，也不实现流式推理、低延迟端侧部署和实时系统优化。与其在多个方向上浅尝辄止，本项目优先保证以下三点同时成立：第一，模型实现和训练链路正确；第二，能够形成多码率 operating points；第三，benchmark 设计具有可追溯性和可解释性。

== Why This Topic

选择 speech neural codec 作为项目主题，主要出于两个动机。第一，传统音频压缩算法虽然成熟，但其设计哲学仍然以手工设计的变换、量化和编码管线为主，而 neural codec 提供了另一种路线：直接从数据中学习压缩表示、重建路径和感知优化目标。第二，speech 场景相比音乐场景更容易在有限算力和有限时间内得到稳定可听的结果，同时又具有明确的低码率应用价值，因此非常适合作为课程项目中的实验对象。

从研究叙事角度看，这一主题还有一个额外优势：`RVQ` 产生的离散 token 不仅可用于重建，还天然适合作为更大语音生成系统中的中间表示。因此，这个项目虽然以 compression 为主线，但其技术结果同样可以被理解为一种 speech tokenizer 的构建过程。

== Project Deliverables

项目最终交付包含四类长期资产。第一类是模型与训练代码，包括数据加载、配置系统、神经编解码结构、量化器、损失函数和训练循环。第二类是 benchmark 与评测脚本，包括 neural codec 导出、传统 codec wrapper、统一的 manifest 和客观指标统计。第三类是实验结果本身，包括多码率 checkpoint、重建音频样例、统一 summary 表和 rate-distortion 图。第四类是文档与报告资产，包括架构说明、开发计划、benchmark runbook 和当前这份项目报告。

== Development Environment

整个项目使用 `Python + PyTorch + torchaudio` 实现，训练主要在 Linux 服务器上的 A100 GPU 上进行，benchmark 则依赖 `ffmpeg` 提供的 `libopus`、`libmp3lame`、`aac` 和 `flac` 编码器。工程层面使用 JSON 配置驱动不同实验分支，通过 `checkpoints/`、`metrics.jsonl`、`samples/` 和 `evals/outputs/` 来维持实验的可追溯性。报告撰写则使用 `Typst`，其原因是它在保持接近 Markdown 的可写性的同时，能够提供接近 TeX 的表格与图文排版能力。

= Technical Details

== Theory Background

Neural codec 的核心思想，是把传统 codec 中原本分离的分析变换、低维表示、量化和重建过程统一到一个端到端可训练系统中。对 speech 来说，这个过程可以粗略理解为：编码器将原始波形映射为较低时间分辨率的 latent sequence，量化器把连续 latent 映射为有限离散 token，而解码器再根据这些 token 恢复波形。与传统频域变换方案相比，这种方法的优势在于表示不再被先验固定，而是由训练目标与数据分布共同决定。

在本项目中，`RVQ` 的作用尤其关键。单级向量量化通常难以在有限码本大小下兼顾表示能力和训练稳定性，而 residual vector quantization 通过多级逐步逼近残差的方式，把一个高保真的连续表示拆成多个低复杂度离散决策。这样做一方面使码率控制更直接，另一方面也使得系统可以在不改动骨干和训练目标的情况下，仅通过调整量化 stage 数量构建一条 bitrate ladder。

需要强调的是，仅有 autoencoder 和 RVQ 并不能自动带来高质量的感知重建。音频任务中的一个核心困难在于：逐点误差、线性频谱误差和人耳主观质量之间并不完全一致。因此，训练目标的设计往往和骨干设计同样重要。这也是本项目后期把重心从“继续换模型”转向“对齐训练机制”的直接原因。

从实现角度看，本项目最终系统可以抽象为下面的四步：

1. 编码器把长度为 `T` 的波形 `x in R^(1 x T)` 变成长度约为 `T / 320` 的 latent 序列 `z in R^(128 x L)`。
2. `RVQ` 用多个离散 codebook 逐级逼近 `z`，输出 shape 为 `[num_quantizers, L]` 的离散索引。
3. 离散索引查表并求和得到量化后的 latent `z_q`。
4. 解码器根据 `z_q` 重建波形 `x_hat`。

这一路径里有两个和传统 codec 很不一样的点。第一，压缩表示不是手工定义的变换域系数，而是从数据中学到的 latent manifold。第二，码率控制不是通过切换 psychoacoustic rule 或量化步长完成，而是通过离散 token 的预算直接完成。这也是为什么本项目后期可以在不改动骨干和 loss 的前提下，只靠改变 quantizer stage 数量构造一条多码率曲线。

== Baseline Design and Early Simplification

项目最初从一套刻意简化的基线系统出发：`1D Conv Encoder -> RVQ -> 1D Conv Decoder`。其设计原则很明确，即优先保证实现短、变量少、可快速训练，并在课程周期内构建一个完整的 learned compression 闭环。编码器采用逐层 stride 下采样卷积与轻量残差块，解码器则使用镜像式的转置卷积恢复波形；量化器使用最朴素的 greedy residual vector quantizer；损失函数由 waveform L1、multi-scale STFT 以及量化器相关约束组成。

这一阶段最重要的贡献并不是音质，而是工程闭环。模型具备明确的 `encode / decode / forward` 接口，训练端完成了 deterministic split、随机裁剪、周期验证、checkpoint、试听样例导出等最小实验基础设施。更重要的是，基线很快验证了一个关键信号：系统确实能学到语音内容和时序对齐，说明“从 waveform 到 discrete token 再回 waveform”的计算图本身是通的。

然而，试听结果也立刻暴露出根本问题：尽管可以清楚听出语音内容，重建波形仍然带有明显电流音、电颤感和高频数字颗粒。这一现象具有诊断意义。它说明当前系统的瓶颈不在“训练是否收敛”，而在“当前模型和目标函数是否足以支撑可用级音质”。因此，项目后续的发展不再是简单延长基线训练，而是沿着更强骨干和更合理感知目标两个方向推进。

== Model Evolution

本项目的模型演进可以概括为两次关键转折。第一次转折，是从最小 `Conv + RVQ` 基线转向 `SEANet + EMA RVQ` 的更强骨干。第二次转折，是从“仅依赖重建损失”转向“对齐 Encodec/Audiocraft 的训练目标与训练机制”。前者解决的是表达能力与量化稳定性问题，后者解决的则是感知音质与训练动力学问题。

在第一次转折中，项目引入了 Encodec-inspired 的 `SEANet` 风格编码器与解码器。新的骨干包含更成熟的 `SConv1d / SConvTranspose1d` padding 逻辑、带 dilation 的 residual block、瓶颈处的 `SkipLSTM`、以及 `weight norm`。量化器也不再是最朴素的查表式 RVQ，而升级为带 `EMA` 更新、`k-means` 初始化和 dead code replacement 的量化器。这一步直接提升了单样本 overfit 的可行性和码本稳定性，但并没有自动解决全数据训练中的高频破碎和数字沙砾问题。

第二次转折来自更系统的失败分析。项目先通过 `mel loss` 和 encodec-like loss reweighting 等低风险实验验证了一个判断：音质瓶颈并不只来自骨干偏弱，更来自训练目标与真实听感之间的不匹配。随后系统正式引入 `MS-STFT discriminator`、generator adversarial loss、feature matching、梯度 `Balancer` 和更接近 reference 的 optimizer 配方。事实证明，真正把系统推到高保真水平的，正是这一整套训练机制的对齐，而不是单独某一个 trick。

== Final Model Architecture

最终采用的主力系统是一套 `SEANet + EMA RVQ` speech codec。其编码器和解码器均围绕 `SEANet` 结构实现：输入为单声道 16 kHz 波形，编码端使用比率为 `[8, 5, 4, 2]` 的多级下采样，因此总 hop length 为 `320`，相应 frame rate 为 `16000 / 320 = 50 Hz`。模型 latent 维度固定为 `128`，瓶颈附近加入 `2` 层 `SkipLSTM`，以建模局部卷积难以覆盖的更长时程依赖。骨干宽度保持在 `32` filters，这一选择兼顾了训练稳定性与算力成本。

量化器部分采用 `EMAResidualVectorQuantizer`。码本大小固定为 `1024`，因此每个 code 的名义信息量约为 `10 bits`。在最终 benchmark 阶段，系统保持骨干、codebook size、frame rate 和训练目标不变，仅通过调整 `num_quantizers` 改变码率。由于 `50 Hz x 10 bits = 500 bit/s` 对应一个 quantizer stage，因此系统的名义码率可以近似写为 `0.5 x num_quantizers kbps`。据此，项目构造了 `24 / 16 / 8 / 4` stage 对应的 `12 / 8 / 4 / 2 kbps` neural bitrate ladder。

如果进一步展开到可复现层面，编码器、量化器和解码器的职责分工如下。

首先，编码器使用 `SEANetEncoder`。每一级下采样都包含两类操作：一类是带 dilation 的 residual block，用来在当前时间分辨率上扩展感受野并聚合局部上下文；另一类是 stride convolution，用来降低时间分辨率并把更多建模预算转移到 channel 维度。当前实现里，时间分辨率按 `[8, 5, 4, 2]` 连续收缩，因此原始 `16000 Hz` 波形最终对应 `50 Hz` 的 latent frame rate。这一选择非常关键，因为它直接决定了“每秒生成多少帧离散 token”，也直接进入码率公式。

其次，量化器使用 `EMAResidualVectorQuantizer`。与单级 VQ 不同，当前实现采用逐级残差逼近。给定编码器输出 `z`，第一个 codebook 先选择最接近的离散向量 `q_1`，然后对残差 `r_2 = z - q_1` 继续量化，依次得到 `q_2, q_3, ..., q_N`，最终量化表示是这些 stage 的求和 `z_q = sum_i q_i`。直观上，这相当于把一个高保真连续表示分解成多个更低复杂度的离散修正。实现上，码本并不是通过普通反向传播直接更新，而是通过 exponential moving average 维护 `embedding`、`embedding_avg` 和 `cluster_size`，再配合首次训练批次的 `k-means` 初始化以及 dead code replacement 来提高稳定性。其核心优点是：在大 codebook、长训练下，码本利用率通常比直接随机初始化更稳，也更接近成熟 neural codec 的工程实现。

最后，解码器使用与编码器镜像的 `SEANetDecoder`。它先把量化后的 latent 沿时间轴逐步上采样，再经过多层 residual block 和最终卷积恢复波形。由于编码器与解码器共享同样的 hop length，且当前系统是离线非流式设定，因此整个路径可以稳定地实现“任意长度 utterance -> token sequence -> reconstruction”的一一对应。也正因此，benchmark 阶段的 neural codec 导出脚本并不依赖训练时的随机裁剪逻辑，而是可以直接按 utterance 整句导出重建结果。

如果只保留最小复现信息，这套系统的核心超参数其实可以压缩成下面四个：`sample_rate = 16 kHz`、`hop_length = 320`、`latent_dim = 128`、`codebook_size = 1024`。其余多码率 operating points 都只是在此基础上改变 `num_quantizers`。这也是本项目认为“码率变化应尽量与骨干/训练目标变化解耦”的主要原因。

#figure(
  paper-table(
    columns: (1.8fr, 2.4fr),
    align: left + horizon,
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Component*], [*Setting*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Backbone], [SEANet encoder / decoder with dilated residual blocks and 2-layer SkipLSTM],
    [Sample rate], [16 kHz mono speech],
    [Downsampling ratios], [[8, 5, 4, 2], giving hop length 320 and frame rate 50 Hz],
    [Latent dimension], [128],
    [Quantizer], [EMA RVQ with codebook size 1024, k-means init, dead code replacement],
    [12 kbps operating point], [24 quantizers],
    [8 kbps operating point], [16 quantizers],
    [4 kbps operating point], [8 quantizers],
    [2 kbps operating point], [4 quantizers],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Final neural codec architecture and bitrate ladder.]
)

== Training Objective Evolution

训练目标的演进，是本项目中最关键的主线之一。第一版基线使用的是最常见、也最容易落地的组合：waveform L1、multi-scale STFT、再加上量化器的 commitment/codebook loss。这种设计能够快速建立基本可训练性，但它默认了一个较强假设：只要时域和线性频域误差足够小，主观听感也会同步改善。后续实验很快表明，这个假设在 speech codec 上并不充分。

接下来引入的 `mel loss` 和 encodec-like loss reweighting，属于中间过渡阶段。它们的意义不是最终答案，而是帮助项目确认“目标函数确实在影响听感收敛速度”。尤其是在保持骨干不变的单样本 overfit 验机中，弱化 waveform 项、强化多尺度频谱项能够更快地逼近自然人声，这说明原始目标函数与真正需要优化的 perceptual quality 存在偏差。

最终真正起决定作用的，是 adversarial 路径的建立与稳定化。当前系统的 generator 端损失包括四部分：waveform 重建项、multi-scale STFT 重建项、adversarial generator loss 和 feature matching loss；判别器使用多尺度 STFT 频谱域判别器，generator/discriminator 优化器均采用与 reference 更接近的 `Adam(beta1=0.5, beta2=0.9)`；多个损失项的梯度则通过本地 `Balancer` 做重新配平，而不是简单做固定权重求和。事实表明，这一训练制度是项目从“能听清内容但音质差”转向“全数据达到高保真”的真正分界点。

为了让这一点更具体，有必要把最终训练目标写成“组件权重 + 梯度配平”的形式。设生成器输出为 `x_hat`，参考波形为 `x`，则当前 generator 端涉及的主要损失分量可以近似记为：

$L_G = 0.1 L_"wave" + 2.0 L_"stft" + 4.0 L_"adv" + 4.0 L_"fm" + L_"commit"$

其中 `L_wave` 是 waveform L1，`L_stft` 是三种 FFT size 上的 multi-scale STFT magnitude + log-magnitude loss，`L_adv` 是生成器对判别器的 hinge adversarial term，`L_fm` 是 feature matching loss，`L_commit` 则对应量化器的 commitment penalty。判别器端的目标则是标准的 hinge discriminator objective，即同时拉开真实音频与重建音频在多尺度 STFT 判别器上的响应。需要强调的是，这个式子描述的是各训练分量的目标权重，而不是最终在代码里做一次标量求和后直接反传的唯一 objective。

这里最容易被低估的不是 adversarial 或 feature matching 本身，而是 `Balancer`。如果简单用固定系数把这些 loss 直接相加，某个梯度量级过大的项就可能主导整个更新，进而导致静音塌缩或 feature matching 发散。当前实现里，`Balancer` 的做法是：对每个 loss 分别计算其对共享张量的梯度范数，维护对应的 EMA norm，再按预设权重比例把这些梯度重新缩放后回传。也就是说，报告里的 `0.1 / 2.0 / 4.0 / 4.0` 不应被理解为“实际 backward 时直接相加的系数”，而应理解为“希望 generator 端各训练分量在梯度层面占据的相对贡献比例”。这一点是本项目与早期失败实验之间最关键的差异之一。

还有一个对复现很重要的点是 optimizer 语义。最终成功实验使用的是 `Adam` 而不是 `AdamW`，并采用 `(beta1, beta2) = (0.5, 0.9)`、`weight_decay = 0`。这看起来像细节，但在对抗训练里其实属于 recipe 的组成部分。项目中曾经真实遇到过因为 generator 端沿用重建任务常见的 `AdamW(0.9, 0.999)` 而导致训练动力学不对的情况，因此这个设置值得在报告里明确写出。

#figure(
  paper-table(
    columns: (1.5fr, 2.7fr),
    align: left + horizon,
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Training element*], [*Final setting*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Waveform loss], [weight 0.1],
    [Multi-scale STFT loss], [weight 2.0 with FFT sizes 512 / 1024 / 2048],
    [Mel loss], [disabled in final high-fidelity recipe],
    [Adversarial loss], [enabled, weight 4.0, hinge objective],
    [Feature matching], [enabled, weight 4.0],
    [Discriminator], [MS-STFT discriminator over five spectral scales],
    [Gradient balancing], [local Balancer enabled],
    [Optimizer], [Adam, learning rate 3e-4, betas (0.5, 0.9), no weight decay],
    [Training clips], [2.0 s clips, batch size 8],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Final training recipe that produced the reportable neural codec results.]
)

== Key Debugging and Engineering Lessons

项目开发过程中最重要的工程经验，并不是某条特定配置，而是诊断顺序。首先，`single-example overfit` 被证明是整个训练链路的必要闸门。多次关键决策都依赖于这一验机实验：如果单样本都学不住，优先怀疑实现错误或训练语义；如果单样本能学住但全数据仍差，则问题更可能在泛化、目标函数或优化机制。项目最终的对抗训练修复，也是通过 overfit 路线先验证后才推广到全数据训练。

其次，音频 codec 里“数值指标改善”并不必然等于“听感改善”。本项目早期多次遇到这种情况：共享频谱指标有所下降，但试听中仍有明显电流音、数字沙砾感或声线变化。因此，试听样例并不是客观指标之外的补充展示，而是调试过程中的核心证据。尤其在 `2 kbps` 等极低码率点上，`STOI` 可以仍然很高，但 timbre 已经开始漂移；这类现象只能通过人工试听和 case study 结合理解。

第三，对抗训练的失败模式必须被当作算法问题而不是“训练波动”。项目实际踩到过静音塌缩、feature matching 发散、优化器语义不对、resume 状态不完整等问题。后期之所以能逐步纠正这些问题，很大程度上得益于配置开关、实验隔离和显式归档，使得每一轮改动都可以被定位为“哪个组件改变了什么”，而不是在同一配置上不断覆盖历史。

== Bitrate Ladder Design

在最终 benchmark 设计中，项目刻意避免了“每个码率都重新设计一套模型”的做法，而是固定成功的神经编解码 recipe，仅通过 `num_quantizers` 控制压缩预算。这种设计有两个优点。第一，它让 rate-distortion 曲线的解释更干净，因为不同 operating point 之间的性能变化主要反映离散预算变化，而不混入骨干结构或训练目标变化。第二，它使 neural codec 的实际码率几乎精确可控，这一点在传统 codec baseline 上反而并不天然成立。

具体来说，在当前配置下，一个量化 stage 对应约 `0.5 kbps`，因此 `4 / 8 / 16 / 24` 个 quantizers 分别对应 `2 / 4 / 8 / 12 kbps`。长期训练结果表明，这条 ladder 的四个点都能够获得可用乃至高保真的主观质量。尤其值得注意的是，即便在 `2 kbps` 的极限压缩点上，系统仍然保持了较强的语音清晰度，只是个别样本会出现 timbre drift 和说话人声线变化。

#figure(
  paper-table(
    columns: (1.15fr, 0.8fr, 0.9fr, 1.0fr, 2.75fr),
    align: (center, center, center, center, left),
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Model*], [*Quantizers*], [*Nominal kbps*], [*Best / total steps*], [*Observation*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Neural-12k], [24], [12.0], [91000 / 100000], [High-fidelity reconstruction; gains after 40k became incremental],
    [Neural-8k], [16], [8.0], [98000 / 100000], [Stable high-fidelity point with lower bitrate than 12k],
    [Neural-4k], [8], [4.0], [148000 / 150000], [Still highly intelligible, with slightly stronger low-rate artifacts],
    [Neural-2k], [4], [2.0], [157000 / 180000], [Content remains clear; occasional timbre drift appears in hard samples],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Final neural bitrate ladder and best checkpoints used in benchmark.]
)

== Benchmark Pipeline Design

为了避免训练代码和评测代码耦合，最终 benchmark 被单独组织在 `evals/` 目录下。整个流程分为四步：先通过 deterministic split 构建统一测试 manifest；然后导出四个 neural codec operating point 的重建结果；接着调用 `ffmpeg` wrapper 运行 `Opus / MP3 / AAC / FLAC`；最后统一计算 `actual_bitrate_kbps`、`compression_ratio_vs_pcm16`、`STOI`、`LSD`、`multi_scale_stft` 和 `SI-SDR` 等指标。这种设计的目的，是保证所有 codec 都在完全相同的 utterance 集合上比较，并且所有结果都可以追溯到统一 manifest。

benchmark 方法学里一个非常关键的原则，是不把“目标码率”直接当作最终比较坐标。对 neural codec 来说，离散 token 预算由 `frame_rate x num_quantizers x log2(codebook_size)` 决定，因此实际码率几乎可以精确控制；但传统 lossy codec 的 `target bitrate` 更像 operating point 输入，而不是必须精确命中的数学约束。也正因如此，报告中的主结论一律建立在 `actual bitrate` 和真实压缩字节数上，而不是“向编码器请求了多少 kbps”。

从实现上看，benchmark 的数据与结果组织也是刻意设计过的。第一步，`build_manifest.py` 根据训练配置中的 deterministic split 规则生成 benchmark manifest，从而避免“训练集与测试集定义不一致”或“人工挑样本”带来的歧义。第二步，`export_neural_codec.py` 使用 checkpoint 逐条导出 neural reconstructions，并根据 `num_frames * num_quantizers * log2(codebook_size)` 统计 neural payload bytes，而不是错误地拿 checkpoint 大小充当压缩成本。第三步，`run_traditional_codec.py` 只负责把统一 manifest 中的音频交给 `ffmpeg` 编码/解码，因此所有传统 codec 都走同一套输入输出接口。第四步，`score_outputs.py` 在统一的文件级 manifest 上计算指标并汇总成 `summary.csv` 与 `per_file_metrics.jsonl`，为报告中的表格、图和 case study 提供同一数据源。

= Experiment Results

== Experimental Questions

本项目实验部分主要回答三个问题。第一，在同一神经编解码框架下，是否可以构建一条从 `2 kbps` 到 `12 kbps` 的有效 speech bitrate ladder。第二，在真正低码率的 operating points 上，neural codec 与传统 codec 相比是否具有更好的质量-码率折中。第三，系统在极低码率下的主要退化形式是什么：是语音内容丢失、背景噪声增强，还是 timbre/speaker identity 的漂移。

== Training Validation

训练验证可以分成两个层面。第一个层面是单样本 overfit。最终成功的 `ablation-adversarial-msstft-balanced` 路线在单样本 overfit 中表现出非常稳定的下降趋势：验证总损失从 `step 1000` 的 `2.56` 下降到 `step 4000` 的 `1.15`，对应的 waveform loss 从 `0.0077` 下降到 `0.0020`，STFT 项则从 `1.28` 降到 `0.57`。更重要的是，试听结果显示 `step 1000` 已经具有较高保真度，而 `step 4000` 几乎达到真假难分。这个结果非常关键，因为它证明 adversarial + balanced 训练链路已经不是“勉强跑通”，而是真正具备高质量拟合能力。

第二个层面是全数据训练。四条 neural ladder 的最佳验证步数分别落在 `91k / 100k`、`98k / 100k`、`148k / 150k` 和 `157k / 180k`，这里前一个数字是最佳 checkpoint step，后一个数字是该 run 的总训练步数。就本项目这四条成功 run 而言，较低码率点的最佳 checkpoint 的确出现在更靠后的训练阶段，但这应被理解为一条经验观察，而不是脱离当前实验设置的一般规律。以 `12 kbps` 路线为例，验证总损失从 `step 1000` 的 `5.00` 持续下降到 `step 91000` 的 `2.18`；主观试听则表明在 `40000 step` 左右已经达到高保真，之后的继续训练仍有小幅改善。也就是说，最终高保真结果并不是“短训偶然撞出来的样例”，而是长程训练和稳定训练制度共同作用的结果。

== Benchmark Setup

benchmark 使用当前仓库定义的 deterministic held-out split，总计 `43` 条 speech utterances，采样率统一为 `16 kHz` 单声道。需要明确的是，这里的 `test` 不是 LibriSpeech 官方 `test-clean`，而是从 `clean/train.100` 目录按项目配置切出的 repo-local held-out slice。神经编解码器使用四个训练完成的 operating points：`Neural-12k`、`Neural-8k`、`Neural-4k` 和 `Neural-2k`。传统 baseline 则包括 `Opus`、`MP3`、`AAC` 和无损 `FLAC`。其中 `FLAC` 仅作为无损锚点，用于验证评测管线是否存在读写或对齐问题，而不参与低码率 lossy 结论。

评价指标分为三类。压缩成本指标包括 `compressed_bytes`、`actual_bitrate_kbps` 和 `compression_ratio_vs_pcm16`；内容和感知相关指标包括 `STOI`、`LSD` 和 `multi_scale_stft`；波形一致性指标则以 `SI-SDR` 为补充。需要强调的是，本项目不把单一指标当作最终判据。尤其在极低码率区域，`STOI` 能说明 intelligibility，`LSD` 和 `multi_scale_stft` 更能反映频谱形态误差，而 `SI-SDR` 在 timbre drift case 上往往会更敏感。因此，最终结论依赖于多指标与试听结果的综合判断。

== Objective Results

从 benchmark 的整体结果看，本项目已经形成了一条可解释的 neural rate-distortion ladder。四个神经 operating points 的实际码率几乎严格落在目标附近：`neural-12k`、`neural-8k`、`neural-4k` 和 `neural-2k` 的实际码率分别约为 `12.01`、`8.00`、`4.00` 和 `2.00 kbps`。这一点本身就值得强调，因为它意味着 neural codec 的压缩预算是高度可控的。

与之相对，传统 codec 在超低目标码率区间表现出明显的 target-actual 脱钩。尤其在 `2 kbps` 目标点，`Opus`、`MP3` 和 `AAC` 的实际码率分别约为 `5.45`、`8.18` 和 `10.67 kbps`。换言之，这些实现并没有在真正的 `2 kbps` operating point 上工作，而是落在各自实现允许的更高码率 floor 附近。这一现象并不是实验噪声，而是 benchmark 方法学上的重要发现：传统 codec 可以接受目标码率输入，但最终应该以实际输出大小和实际码率作为比较基准。

进一步看不同传统 codec 的低码率 sweep，可以发现这种 floor 现象并不是偶发的，而是系统性的。`MP3` 在 `2 / 4 / 8 kbps` 三个目标点上都落到了几乎相同的 `8.18 kbps` 实际码率；`AAC` 的 `2 / 4 / 8 kbps` 三个目标点也都集中在约 `11 kbps` 左右；`Opus` 则在 `2 / 4 kbps` 上共同落在约 `5.45 kbps`。这说明在超低目标码率区域，target bitrate 更适合被理解为“请求某个 operating point”的输入参数，而不是最终比较坐标。也正因如此，本项目把 actual bitrate 作为所有结果分析与图表绘制的统一横轴。

作为 sanity check，`FLAC` 锚点结果也证明评测管线本身是可信的。无损重建对应的 `LSD` 和 `multi_scale_stft` 都为 `0`，`STOI` 接近 `1.0`，这意味着 manifest、重采样、文件读写和指标实现之间没有出现破坏性偏差。因此，后续不同 codec 之间的差异可以较放心地解释为压缩策略差异，而不是评测链路错误。

#figure(
  paper-table(
    columns: (1.2fr, 1fr, 1fr),
    align: center + horizon,
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Codec label*], [*Target point*], [*Actual bitrate (kbps)*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Neural-2k], [2 kbps], [2.00],
    [Opus-2k], [2 kbps], [5.45],
    [MP3-2k], [2 kbps], [8.18],
    [AAC-2k], [2 kbps], [10.67],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Target bitrate and actual bitrate diverge strongly for traditional codecs in the ultra-low-rate regime.]
)

在真正低码率区域，neural codec 展示出了最强的实验价值。`Neural-2k` 的平均 `STOI` 达到 `0.886`，`LSD` 为 `10.08`，`multi_scale_stft` 为 `1.022`。作为对比，名义上的 `Opus-2k` 虽然仍能保留一定可懂度，但其实际码率已经达到 `5.45 kbps`，同时 `STOI` 只有 `0.747`，`LSD` 和 `multi_scale_stft` 也明显更差。也就是说，在真正极低的实际码率下，当前 neural codec 已经能够用更少的 bit 维持更强的语音可懂度和更低的频谱失真。

中等码率区域的结果更细腻。`Neural-8k` 对 `MP3` 和 `AAC` 的优势十分明显，尤其在 `LSD` 和 `multi_scale_stft` 上远好于传统实现；而与 `Opus-8k` 相比，结果则呈现出一种“指标分裂”的状态：`Opus-8k` 的 `STOI` 仍然高于 `Neural-8k`，但 `Neural-8k` 在 `LSD` 和 `multi_scale_stft` 上明显更优，而且在本项目的主观试听中，其重建往往显得更清亮、更接近原声。这一现象也再次提醒我们，客观指标终究只是对主观听感的近似拟合，而不是绝对的裁决标准；尤其在高保真区间，不同指标分别刻画的是可懂度、频谱偏差与主观透明度的不同侧面。类似地，在 `12 kbps` 左右，`Opus-12k` 在当前客观指标上整体优于 `Neural-12k`。因此，本项目的实验结论不应写成“neural codec 在所有低码率区间全面战胜传统 codec”，更稳妥的表述是：neural codec 在极低实际码率下展示出显著优势，在更高码率下达到与主流传统 codec 可比的高保真水平，而 `Opus` 在 `8-12 kbps` 区间仍然是一个非常强、且必须认真对待的传统基线。

#figure(
  paper-table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Codec*], [*Actual kbps*], [*STOI*], [*LSD*], [*MS-STFT*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Neural-2k], [2.00], [0.886], [10.08], [1.022],
    [Opus-2k], [5.45], [0.747], [27.17], [2.664],
    [MP3-2k], [8.18], [0.865], [37.88], [3.979],
    [AAC-2k], [10.67], [0.866], [33.79], [3.475],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Ultra-low-rate comparison. The neural 2 kbps point is the only system that truly operates near 2 kbps while maintaining strong intelligibility.]
)

如果把“same-quality efficiency”也显式展开，结论会更清楚。以 `Neural-2k` 为例，如果只看语音可懂度这一最基础的质量目标，那么传统 codec 需要明显更高的实际码率才能达到或超过其 `STOI = 0.886`。在当前测得的 operating points 中，`Opus` 至少需要 `8.30 kbps`，`AAC` 至少需要 `13.29 kbps`，而 `MP3` 则需要到 `16.23 kbps` 才能超过这一阈值。这意味着，哪怕只用最保守的可懂度口径来衡量，当前 neural codec 在极低码率区间依然展示出了明显的压缩效率优势。不过，为避免把离散 operating point 误读为插值结论，正文不再单独给出“追平某指标需要多少 kbps”的推导表，而是保留一张主结果表，并把完整 benchmark 长表放到附录中。

#figure(
  paper-table(
    columns: (1.35fr, 0.95fr, 0.9fr, 1fr, 0.85fr, 0.9fr, 1.0fr),
    align: (left, center, center, center, center, center, center),
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Codec*], [*Target*], [*Actual kbps*], [*Comp. ratio*], [*STOI*], [*LSD*], [*MS-STFT*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Neural-12k], [12 kbps], [12.01], [21.32x], [0.923], [9.52], [0.943],
    [Neural-8k], [8 kbps], [8.00], [31.98x], [0.918], [9.58], [0.953],
    [Neural-4k], [4 kbps], [4.00], [63.97x], [0.908], [9.67], [0.965],
    [Neural-2k], [2 kbps], [2.00], [127.93x], [0.886], [10.08], [1.022],
    [Opus-12k], [12 kbps], [12.34], [20.76x], [0.972], [7.88], [0.743],
    [Opus-8k], [8 kbps], [8.30], [30.85x], [0.956], [26.99], [2.496],
    [Opus-4k], [4 kbps], [5.45], [47.00x], [0.747], [27.17], [2.664],
    [MP3-16k], [16 kbps], [16.23], [15.77x], [0.961], [27.49], [2.468],
    [MP3-2k], [2 kbps], [8.18], [31.29x], [0.865], [37.88], [3.979],
    [AAC-16k], [16 kbps], [17.27], [14.82x], [0.967], [30.73], [2.881],
    [AAC-2k], [2 kbps], [10.67], [24.01x], [0.866], [33.79], [3.475],
    [MP3-default], [-], [24.29], [10.54x], [0.987], [22.07], [1.777],
    [Opus-default], [-], [63.06], [4.07x], [0.999], [4.24], [0.290],
    [AAC-default], [-], [74.99], [3.42x], [1.000], [2.94], [0.179],
    [FLAC], [-], [145.88], [1.76x], [1.000], [0.00], [0.000],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Representative benchmark summary used in the main text. The full benchmark table is moved to Appendix A, and all conclusions are derived from actual bitrate rather than target bitrate alone.]
)

除了 target sweep，本项目还额外保留了 `default mode`，用于观察传统 codec 在不受人为目标码率约束时的自然 operating point。结果同样很有代表性：三类传统 codec 在 default mode 下都达到了几乎透明的听感，但它们对应的实际码率也显著更高，说明这些高保真结果依赖的是更宽裕的 bit budget，而不是在同码率条件下天然更优。

#figure(
  paper-table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Codec default*], [*Actual kbps*], [*Compression ratio*], [*STOI*], [*LSD*]),
    table.hline(y: 1, stroke: 0.5pt),
    [MP3 default], [24.29], [10.54 x], [0.987], [22.07],
    [Opus default], [63.06], [4.07 x], [0.999], [4.24],
    [AAC default], [74.99], [3.42 x], [1.000], [2.94],
    [FLAC], [145.88], [1.76 x], [1.000], [0.00],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Codec-native default operating points. High subjective fidelity is achieved at substantially higher actual bitrates than the ultra-low-rate neural points.]
)

#figure(
  image("assets/rd_stoi.svg", width: 100%),
  caption: [Rate-distortion view using STOI. Neural codec forms a controllable low-rate ladder, while traditional codecs occupy different actual operating points.]
)

#figure(
  image("assets/rd_lsd.svg", width: 100%),
  caption: [Rate-distortion view using log spectral distance. Neural codec has a clear advantage over MP3 and AAC in the low-rate regime, while Opus remains the strongest classical speech baseline.]
)

#figure(
  image("assets/rd_msstft.svg", width: 100%),
  caption: [Rate-distortion view using multi-scale STFT distance. The neural ladder remains compact and predictable, whereas classical codecs show codec-specific rate-control floors at low target bitrates.]
)

== Subjective Listening Findings

试听结果与客观指标总体一致，但也揭示了一些指标难以完全表达的细节。首先，四个 neural operating points 都已达到“可用”级别，这一点本身非常重要。尤其是 `12 kbps` 和 `8 kbps`，在多数样本上已经可以达到相当高的透明度；`4 kbps` 在保持较强清晰度的同时引入了更明显的低码率痕迹；`2 kbps` 则最具研究价值，因为它往往仍然能保持很好的内容可懂度，但 timbre 保真开始成为主要瓶颈。

在 benchmark 的抽样主观试听中可以观察到，传统 codec 的 `default` operating point 与原声几乎难以区分，而这与实际码率也完全一致：`mp3-default`、`opus-default` 和 `aac-default` 的实际码率分别约为 `24.29`、`63.06` 和 `74.99 kbps`。因此，default mode 的意义并不在于“证明传统 codec 更强”，而在于揭示 codec-native operating point 会自然落在更高比特预算上，从而提供更接近无损的听感。

对 neural codec 而言，最典型的主观问题出现在 `Neural-2k` 的个别样本上。例如 `1069-133709-0005` 与 `1069-133709-0027` 两条 utterance 在试听中都出现了说话人后半段声线发生变化的现象。它们的共同特点是：`STOI` 仍然保持在 `0.87` 左右，说明内容与清晰度并未丢失；但 `SI-SDR` 较低，且 `LSD` 与 `multi_scale_stft` 相对其他 neural operating points 更差。这种失败模式很适合作为低码率 case study，因为它不是“完全听不清”，而是“内容正确但 timbre/speaker color shift”。也正因如此，本项目对 `2 kbps` 结果的解释必须诚实：它在压缩效率上极具说服力，但在说话人音色保真上仍存在边界。

=== Case Study: Timbre Drift at 2 kbps

为了把 `neural-2k` 的主要失败模式说清楚，仅靠均值表是不够的。更直接的做法是挑出若干代表性失败样本，把“内容基本正确但说话人音色发生偏移”这件事单独讲清楚。当前最具代表性的两个样本来自主观试听与客观指标交叉筛出的 utterance，它们恰好也落在 `neural-2k` 较差样本之列。

#figure(
  paper-table(
    columns: (1.6fr, 0.9fr, 0.9fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Sample ID*], [*STOI*], [*SI-SDR*], [*LSD*], [*MS-STFT*]),
    table.hline(y: 1, stroke: 0.5pt),
    [1069-133709-0005], [0.878], [0.089], [10.62], [1.080],
    [1069-133709-0027], [0.872], [1.755], [10.80], [1.125],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Representative `neural-2k` failure cases. Intelligibility remains high, but waveform and spectral fidelity already degrade enough to produce audible timbre drift.]
)

这两个样本的共同点很值得强调。它们的 `STOI` 仍然处在较高区间，因此模型在内容层面并没有明显崩坏；真正先退化的是波形细节和频谱细节，这会在主观上表现成说话人后半段声线“变了一个人”，或者背景里出现轻微但持续的数字感振颤。换句话说，`2 kbps` 的主要风险不是 intelligibility collapse，而是 speaker identity drift。

为了避免“整段时频图看起来都差不多，因此 artifact 不显著”这种误读，最终 case study 图不再展示整句频谱，而是只截取局部窗口，并显式绘制 `source / reconstruction / absolute log-spectral difference` 三联图。局部分析也支持这种做法：在漂移样本 `1069-133709-0005` 的 `6–7 s` 区间，source 与 reconstruction 的波形相关系数只有 `0.263`，局部 `LSD` 约为 `0.690`；而稳定对照样本 `1069-133709-0006` 在相同时间窗口上的相关系数约为 `0.891`，局部 `LSD` 约为 `0.482`。这说明 timbre drift 的核心不是“整段频谱完全变形”，而是局部高频和谐波细节在特定时间片发生了更强的偏移。更重要的是，一旦这种差异能够被相对稳定地可视化和量化，它就不再只是主观描述，而可能成为 future work 中可进一步建模的训练信号；换言之，如果某类差异真的对应到“声线”或 timbre 语义，那么改写 loss 以更直接约束这类差异，就是有希望的研究方向。

#figure(
  image("assets/case_timbre_drift.svg", width: 100%),
  caption: [Local-window case-study figure for `neural-2k`. The first two rows visualize the drift sample `1069-133709-0005` in the two time regions where the user reported audible timbre change; the third row provides a stable control sample `1069-133709-0006`. The rightmost column visualizes absolute log-spectral difference, making the artifact more explicit than a full-utterance spectrogram view.]
)

== Rate-Distortion Interpretation

如果把当前结果放到同一个 rate-distortion 视角下，项目的主要结论可以概括为三点。第一，当前 neural codec 的最大优势不在于“在所有码率都碾压传统 codec”，而在于“在真正极低的实际码率下仍维持了高可懂度和较低频谱失真”。这一点在 `Neural-2k` 上体现得最明显。第二，随着码率上升到 `8-12 kbps`，传统 codec 尤其是 `Opus` 会迅速变强，因此神经方法的叙事应当从“绝对胜负”转向“在哪个 operating point 更有优势”。第三，传统 codec 的 low-target sweep 暴露出明显的 rate-control floor，这使得 actual bitrate 成为比 target bitrate 更有意义的 benchmark 坐标。

从课程项目角度看，这一结论已经足够有说服力。因为项目并不是在大规模数据、超大模型或多年工程优化下与工业级 codec 正面对决，而是在有限时间和有限资源约束下，成功构建了一条多码率 learned speech codec 曲线，并在低码率区域展示出了清晰的压缩效率优势。更重要的是，这一优势并不是依赖单一偶然样本，而是通过统一 manifest、统一 scoring 和完整 bitrate ladder 支撑起来的。

== Failure Cases and Limitations

尽管结果总体成功，本项目仍然存在几个明确限制。第一，当前 benchmark 的数据规模仍然是课程项目尺度，测试集只有 `43` 条 speech utterances，因此不能把现有结果直接外推为大规模统计结论。第二，主观听测仍然是轻量级的人工试听，而非正式的 `MUSHRA` 或更大规模双盲感知实验。第三，指标集合虽然覆盖了压缩成本、可懂度和频谱误差，但尚未纳入 `ViSQOL` 等更强的感知指标，这使得高保真区域的结论仍然更依赖试听。

就模型本身而言，`Neural-2k` 的 timbre drift 已经说明：在极限低码率下，系统保住了内容但还没有完全保住说话人音色。另一方面，`Opus` 在 `8-12 kbps` 区间仍然很强，说明当前系统虽然已经证明了神经方法的优势区间，但还没有达到“统一压制所有传统 codec”的程度。换言之，这份工作最适合被理解为一个成功的课程级研究原型：它已经证明 learned speech codec 在低码率区域具有现实潜力，但仍有空间通过更完整的感知指标、更多数据和更精细的训练机制继续打磨。

== Final Conclusion

本项目最终完成了三个层面的目标。第一，在工程上，系统从零构建了一条完整的 speech neural codec 链路，覆盖模型、配置、训练、试听、benchmark 和报告资产。第二，在研究上，项目明确展示了从轻量基线到高保真系统的真正关键转折：决定音质跃迁的，不是单纯更换骨干，而是把训练目标与优化机制对齐到更合理的感知训练范式。第三，在实验上，项目通过 `2 / 4 / 8 / 12 kbps` 的 neural bitrate ladder 和与 `Opus / MP3 / AAC / FLAC` 的统一 benchmark，证明了 neural codec 在极低实际码率下的压缩效率优势，并且在更高码率下达到了具有竞争力的高保真重建。

因此，这个项目最重要的结论并不是“神经方法已经全面战胜传统 codec”，而是更具体、更可信的判断：在 speech 低码率压缩任务中，neural codec 具有明确的可行性和显著的低码率潜力；其成功依赖于骨干结构、离散表示和感知训练目标的协同设计；而传统 codec 仍然在更高码率或默认 operating point 上保持很强的现实竞争力。

== Future Work

如果继续推进，本项目最自然的下一步有四个方向。第一，补充更强的感知指标和更系统的主观听测，把当前“试听成功”的结论扩展为更正式的听觉评价。第二，在不改变当前主骨干的前提下，继续研究极低码率区域的 timbre 保真问题，尤其是 `2 kbps` 下的 speaker color drift。当前工作已经花了一定精力才找到一种能把该 artifact 直观展示出来的表示方式，而这本身就具有研究意义：如果 `absolute log-spectral difference` 或其变体能够稳定对应到声线偏移语义，那么它完全可能被继续发展为更直接面向 timbre 保真的 loss 或辅助判别信号。第三，完善 benchmark 图表和案例库，例如加入失败样本的频谱对比图与更系统的 artifact taxonomy。第四，如果把项目从课程原型继续推进到研究原型，可以尝试引入更强的熵模型、更多数据或更强的时序建模，以进一步提升同码率下对 `Opus` 的竞争力。

= Appendix A: Full Benchmark Table

为了让正文保持可读性，主结果部分只保留了最有解释力的 operating points。本附录则给出完整 benchmark 长表，以便读者核对所有 target sweep、default operating point 与 neural bitrate ladder 的原始统计值。

#figure(
  paper-table(
    columns: (1.2fr, 0.9fr, 0.85fr, 1fr, 0.9fr, 0.95fr, 1.0fr),
    align: (left, center, center, center, center, center, center),
    table.hline(y: 0, stroke: 0.9pt),
    table.header([*Codec*], [*Target*], [*Actual kbps*], [*Comp. ratio*], [*STOI*], [*LSD*], [*MS-STFT*]),
    table.hline(y: 1, stroke: 0.5pt),
    [Neural-12k], [12 kbps], [12.01], [21.32x], [0.923], [9.52], [0.943],
    [Neural-8k], [8 kbps], [8.00], [31.98x], [0.918], [9.58], [0.953],
    [Neural-4k], [4 kbps], [4.00], [63.97x], [0.908], [9.67], [0.965],
    [Neural-2k], [2 kbps], [2.00], [127.93x], [0.886], [10.08], [1.022],
    [Opus-16k], [16 kbps], [16.12], [15.88x], [0.985], [7.52], [0.683],
    [Opus-12k], [12 kbps], [12.34], [20.76x], [0.972], [7.88], [0.743],
    [Opus-8k], [8 kbps], [8.30], [30.85x], [0.956], [26.99], [2.496],
    [Opus-4k], [4 kbps], [5.45], [47.00x], [0.747], [27.17], [2.664],
    [Opus-2k], [2 kbps], [5.45], [47.00x], [0.747], [27.17], [2.664],
    [Opus-default], [-], [63.06], [4.07x], [0.999], [4.24], [0.290],
    [MP3-16k], [16 kbps], [16.23], [15.77x], [0.961], [27.49], [2.468],
    [MP3-12k], [12 kbps], [8.18], [31.29x], [0.851], [36.11], [3.715],
    [MP3-8k], [8 kbps], [8.18], [31.29x], [0.865], [37.88], [3.979],
    [MP3-4k], [4 kbps], [8.18], [31.29x], [0.865], [37.88], [3.979],
    [MP3-2k], [2 kbps], [8.18], [31.29x], [0.865], [37.88], [3.979],
    [MP3-default], [-], [24.29], [10.54x], [0.987], [22.07], [1.777],
    [AAC-16k], [16 kbps], [17.27], [14.82x], [0.967], [30.73], [2.881],
    [AAC-12k], [12 kbps], [13.29], [19.27x], [0.906], [33.60], [3.357],
    [AAC-8k], [8 kbps], [10.98], [23.33x], [0.870], [33.76], [3.464],
    [AAC-4k], [4 kbps], [10.82], [23.67x], [0.867], [33.78], [3.471],
    [AAC-2k], [2 kbps], [10.67], [24.01x], [0.866], [33.79], [3.475],
    [AAC-default], [-], [74.99], [3.42x], [1.000], [2.94], [0.179],
    [FLAC], [-], [145.88], [1.76x], [1.000], [0.00], [0.000],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Full benchmark summary. This appendix preserves the entire tested operating-point set for traceability.]
)

= References

1. A. van den Oord, O. Vinyals, and K. Kavukcuoglu. *Neural Discrete Representation Learning*. 2017.
2. N. Zeghidour et al. *SoundStream: An End-to-End Neural Audio Codec*. 2021.
3. A. Defossez et al. *High Fidelity Neural Audio Compression*. 2022.
4. Facebook Research / Meta AI. *Encodec* and *Audiocraft* official implementations and documentation.
5. FFmpeg documentation for `libopus`, `libmp3lame`, `aac`, and `flac`.
6. 本项目仓库中的架构文档、配置文件、训练计划、benchmark 脚本与实验结果。
