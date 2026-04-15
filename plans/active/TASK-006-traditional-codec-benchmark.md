Owner: ely
Status: active
Last reviewed: 2026-04-15

# TASK-006 Traditional Codec Benchmark

## Goal

在当前高保真 neural codec 已成立的前提下，设计并落地一套面向 `16 kHz mono speech` 的 benchmark，用来系统比较：

- `neural codec` 和传统/经典 codec 在相同码率下谁的重建质量更好
- 为达到相近质量，各自需要多少码率

本阶段先收敛实验设计与评价口径，再实现 `evals/` 下的独立 benchmark 代码。

## Background

- `TASK-005` 已经把当前 `SEANet + EMA RVQ` 路线推进到高保真重建
- 当前最强实验路线是 `configs/ablation-adversarial-msstft-balanced.json`
- 该路线当前 nominal neural bitrate 约为 `12 kbps`
  - `frame_rate = 16000 / (8*5*4*2) = 50 Hz`
  - `bits_per_frame = 24 quantizers * log2(1024) = 240 bits`
  - `50 * 240 = 12000 bit/s`
- 在固定 backbone / fixed loss / fixed frame rate / fixed codebook size 的条件下，
  neural bitrate 可以通过 `num_quantizers` 近似线性调节：
  - `bitrate_kbps = 0.5 * num_quantizers`
  - 因此自然形成一条 neural bitrate ladder
- 课程目标不止于“模型能训好”，还要回答：
  - neural codec 是否优于传统 codec
  - 优势体现在码率、保真度，还是两者兼有
  - 结论是否建立在合理且可复现的 benchmark 口径上

## Core Benchmark Questions

1. `Same-Bitrate Superiority`
   在多个代表性码率点上，neural codec 和传统 codec 谁的质量更高？

2. `Same-Quality Efficiency`
   为达到与 neural codec 相近的质量，传统 codec 分别需要多少 kbps？

3. `Artifact Profile`
   不同 codec 的主要伪影是什么：
   - 电流感
   - 高频沙砾
   - 发闷
   - 背景噪声
   - 前回声
   - 静音塌缩

## In Scope

- 在 `evals/` 下建立独立 benchmark 框架
- 引入主流且可实现的开源传统 codec baseline
- 统一导出 neural codec 与传统 codec 的重建结果
- 统一统计压缩成本、质量指标和试听样例
- 形成一版可直接写入报告/PPT 的结果表和图

## Out Of Scope

- 一次性支持所有传统 codec
- 多码率 neural retraining
- 完整正式 `MUSHRA`
- `POLQA/PESQ` 这类受限或不适合当前 MVP 的指标
- 论文级大规模统计显著性分析

## Codec Matrix

### Core Set

这是第一轮 benchmark 必须纳入的集合。

| Codec | Role | Implementation Path | Why It Matters |
|---|---|---|---|
| `Neural codec` | primary learned system | bitrate ladder checkpoints | benchmark 主体 |
| `Opus` | strongest open lossy baseline for speech | `ffmpeg -c:a libopus` or `opusenc/opusdec` | speech + low bitrate 主传统对手 |
| `MP3` | required classical baseline | `ffmpeg -c:a libmp3lame` or `lame` | 课程目标里明确要比较 |
| `AAC` | mainstream deployed lossy family | `ffmpeg -c:a aac` | 现实世界常见、补全主流部署面 |
| `FLAC` | lossless anchor | `ffmpeg -c:a flac` or `flac` | 无损 sanity / upper anchor，不参与低码率胜负叙事 |

### Extended Set

这些 codec 有价值，但不应阻塞第一轮结果。

| Codec | Role | Implementation Path | Note |
|---|---|---|---|
| `Vorbis` | open legacy lossy extension | `ffmpeg -c:a libvorbis` | 开放、成熟，但 speech 代表性不如 Opus |
| `Speex` | classic speech codec extension | `speexenc/speexdec` | 适合补充“经典 speech codec”视角 |

### Deferred / Appendix Set

| Codec | Why Deferred |
|---|---|
| `AAC (libfdk_aac)` | 可作为更强 AAC baseline，但涉及 `nonfree` build 和额外许可注意事项 |
| `AMR-WB / AMR-NB` | 更偏 telecom regime，采样率/实现摩擦较大 |
| `Codec2` | 极低码率通信导向，不适合和高保真主表混排 |
| `WavPack` | 对本项目主叙事帮助有限 |
| `LC3` | 可探索，但不作为第一轮课程 benchmark 主角 |

## Benchmark Structure

### Neural Bitrate Ladder

在不改变 backbone 和 loss 的前提下，第一轮 benchmark 假设以下 neural 模型都已训练完成：

| Model | `num_quantizers` | Nominal Bitrate | Role |
|---|---:|---:|---|
| `Neural-2k` | `4` | `2.0 kbps` | aggressive low-bitrate point |
| `Neural-12k` | `24` | `12.0 kbps` | current strongest high-fidelity point |
| `Neural-8k` | `16` | `8.0 kbps` | middle point |
| `Neural-4k` | `8` | `4.0 kbps` | aggressive compression point |

说明：

- 在当前配置下，码率步进是 `0.5 kbps`
- `2.0 kbps` 用于贴近 `Encodec` 低码率 regime 的参考点
- 如果后续还想补更细粒度低码率点：
  - 可进一步增加 `2.5 kbps (5 quantizers)`

### E0: Sanity / Anchor

目标：先验证评测管线没问题。

- `PCM -> FLAC -> decode`
- 指标应接近“完美重建”
- 如果 `FLAC` 指标明显偏差，优先怀疑：
  - 重采样
  - 对齐
  - 指标实现
  - 文件读写链路

### E1: Same-Bitrate Main Experiment

目标：回答“同码率下谁更好”。

- Neural codec：使用 bitrate ladder 的多个 checkpoint，对测试集做统一重建
- 传统 lossy codec：
  - `Opus`
  - `MP3`
  - `AAC`
- 码率档位：
  - neural codec：`2 / 4 / 8 / 12 kbps`
  - traditional codecs：先尝试 `2 / 4 / 8 / 12 / 16 kbps`
- 若某实现无法稳定给出精确 `12 kbps`：
  - 保留该 codec
  - 记录目标设置
  - 最终汇报 `actual bytes` 和 `actual bitrate`

### E2: Rate-Distortion View

目标：从曲线视角看传统 codec 的效率。

- 横轴：`actual_bitrate_kbps`
- 纵轴：
  - `ViSQOL`
  - `STOI`
  - `LSD`
- Neural codec：
  - 以 `2 / 4 / 8 / 12 kbps` 四个点构成 learned RD curve
- Traditional codecs：
  - 作为曲线或离散点集

### E3: Same-Quality Derived Analysis

目标：回答“达到相近质量要多少码率”。

- 不额外重做一套实验
- 从 `E2` 的结果中派生：
  - 追平 `Neural-2k / Neural-4k / Neural-8k / Neural-12k` 各自 `ViSQOL` 需要多少 kbps
  - 追平 `Neural-2k / Neural-4k / Neural-8k / Neural-12k` 各自 `STOI` 需要多少 kbps
- 这是压缩效率结论的主要来源

### E4: Subjective Triplets / AB

目标：避免完全依赖客观指标。

- 形式：
  - `Reference + Candidate A + Candidate B`
  - 问题只问：
    - “哪个更接近 reference 的基本音质？”
- 建议规模：
  - `8-12` 条样本
  - `3-5` 位听者
- 不承诺正式 `MUSHRA`
- 不把“自然度”和“接近 reference”混在同一 trial

### E5: Artifact Casebook

目标：补充定性解释。

- 选若干代表性样本
- 人工记录伪影标签：
  - 电流感
  - 高频沙砾
  - 发闷
  - 失真
  - 前回声
  - 背景噪声
  - 静音塌缩

## Dataset Policy

- 输入统一为 `16 kHz mono`
- 主 benchmark 使用当前 config 定义的 deterministic `test` split
- 两阶段执行：
  - quick loop：先取测试集里的固定小子集
  - final report：跑完整测试集
- 若时间紧张，MVP 先保证：
  - `30-50` 条未见过 speech utterances
  - 数据子集在所有 codec 间完全一致

## Compression Accounting Rules

这是最关键的口径约束之一。

### Original Reference Size

- 统一使用 `16-bit PCM wav` 的理论/实际字节数作为原始大小

### Traditional Codec Size

- 使用真实压缩文件大小
- 包含容器/头部开销
- 最终统一报告：
  - `compressed_bytes`
  - `actual_bitrate_kbps`
  - `compression_ratio_vs_pcm16`

### Neural Codec Size

- 不允许用 checkpoint 大小代替压缩成本
- 主统计口径使用 RVQ code payload 大小：
  - `frames * num_quantizers * ceil(log2(codebook_size)) / 8`
- 如果后续加入 side metadata，也必须计入
- 可选增强：
  - 导出 packed binary code stream 并报告 packed bytes

## Metric Tiers

### Primary Metrics

这些指标必须有，直接进入主结果表。

- `actual_bitrate_kbps`
  - 真正压缩成本
- `compression_ratio_vs_pcm16`
  - 对课程汇报最直观的空间效率指标
- `ViSQOL (speech mode)`
  - 感知质量 proxy
- `STOI`
  - speech intelligibility
- `LSD`
  - 抓频谱失真、高频破碎和沙砾感

### Secondary Metrics

这些指标强烈建议有，但不阻塞 MVP。

- `WER`
  - 衡量语义内容是否保住
  - 使用固定 ASR 模型
- `subjective_triplet_win_rate`
  - 用少量人工听感补足客观指标盲点

### Diagnostic Metrics

这些指标只用于解释现象，不作为 headline 结论。

- `multi_scale_stft`
  - 因训练中已使用，不能当主 benchmark 指标
- `SI-SDR`
  - 可看信号级偏差，但与听感并不稳定对应

## Metric Interpretation Rules

- 不能因为单个客观指标领先就宣布“全面胜出”
- `ViSQOL/STOI/LSD` 要联合解释
- 若客观指标和听感冲突：
  - 以试听样例和 triplet 结果补充说明
  - 不强行下单一结论
- `FLAC` 只做 anchor，不和 lossy 低码率主表混为一谈
- `Opus` 是 speech 低码率主传统对手
  - `MP3` 是课程指定 baseline
  - 若 neural codec 只赢 `MP3` 但输 `Opus`，结论必须诚实表达

## Deliverables

至少输出以下产物：

- `per-file` 结果表（`csv` or `jsonl`）
- 汇总表
- `rate-distortion` 图
- 少量试听样例
- artifact casebook
- benchmark 口径说明文档

## Proposed Layout

```text
evals/
  README.md
  data/
    manifests/
  traditional_codecs/
    mp3/
    opus/
    aac/
    flac/
  neural_codec/
  scripts/
  outputs/
```

设计原则：

- `src/` 只放模型、训练和推理主逻辑
- `evals/` 只放 benchmark、传统 codec 调用、结果导出和汇总分析
- 不把 `ffmpeg/libopus/libmp3lame` 等传统 codec 调用逻辑混入主训练入口

## Execution Strategy

### Phase 1: Benchmark Scaffold

1. 定义 manifest 格式
2. 定义统一输入输出目录结构
3. 固定 benchmark 样本子集与 neural checkpoint
4. 落 neural payload size 统计口径

### Phase 2: Core Codec Integration

1. 接入 `FLAC` sanity path
2. 接入 `MP3`
3. 接入 `Opus`
4. 接入 `AAC`

### Phase 3: Metric Pipeline

1. 输出 `actual_bitrate_kbps`
2. 输出 `compression_ratio_vs_pcm16`
3. 输出 `ViSQOL`
4. 输出 `STOI`
5. 输出 `LSD`
6. 若时间允许，再补 `WER`

### Phase 4: Reporting Assets

1. 生成 per-file 表
2. 生成 summary table
3. 生成 RD figure
4. 导出试听 triplets
5. 编写 artifact notes

## Acceptance Criteria

- `evals/` 下有独立 benchmark 代码
- 至少完成以下 codec：
  - `Neural-2k`
  - `Neural-4k`
  - `Neural-8k`
  - `Neural-12k`
  - `Opus`
  - `MP3`
  - `AAC`
  - `FLAC`
- 至少完成以下主指标：
  - `actual_bitrate_kbps`
  - `compression_ratio_vs_pcm16`
  - `ViSQOL`
  - `STOI`
  - `LSD`
- 结果能直接回答：
  - same-bitrate superiority
  - same-quality efficiency

## Risks and Guardrails

- 不要把 neural checkpoint 大小误当压缩率
- 不要只报告 nominal bitrate，必须同时报告 actual bytes
- 不要让主观样例和客观统计使用不同数据子集
- 若某 codec 无法稳定实现精确目标码率，必须在结果中明确写清
- `AAC (libfdk_aac)` 若后续接入，必须明确标注为 optional `nonfree` path

## Immediate Next Step

先实现 benchmark scaffold，并优先完成：

1. manifest 规范
2. `FLAC` sanity baseline
3. `MP3 / Opus / AAC` encode-decode baseline
4. multi-bitrate neural export spec
5. primary metrics schema
