Owner: ely
Status: active
Last reviewed: 2026-04-15

# TASK-006 Traditional Codec Benchmark

## Goal

在当前高保真 neural codec 结果已经成立的前提下，建立一套与传统音频压缩算法的对比评测流程，重点比较 `MP3` 与 neural codec 在 speech 场景下的压缩率、码率与保真度。

## Background

- `TASK-005` 已经把当前 `SEANet + EMA RVQ` 路线推进到高保真重建
- 课程项目的最终目标并不止于“模型能训好”，还需要回答：
  - 深度学习 codec 与传统 codec 相比是否值得
  - 在相近码率下，谁的保真度更高
  - 在相近听感下，谁的压缩率更优
- 为避免 benchmark 脚本和主训练代码耦合，传统 codec 调用与实验脚本应单独放入 `evals/`

## In Scope

- 在 `evals/` 下建立传统 codec benchmark 目录结构
- 实现 `MP3` baseline 的调用与结果导出
- 统一导出 neural codec 重建结果与传统 codec 重建结果
- 统计至少以下指标：
  - 文件大小 / 名义码率 / 压缩率
  - waveform / 频谱类保真指标
  - 必要的主观试听样例
- 形成一版可直接进入报告或 PPT 的对比表

## Out Of Scope

- 一次性支持所有传统 codec
- 复杂主观听测平台
- 论文级大规模 benchmark

## Proposed Layout

```text
evals/
  README.md
  traditional_codecs/
    mp3/
  scripts/
  outputs/
```

设计原则：

- `src/` 保持模型与训练主逻辑
- `evals/` 专门承载 benchmark、传统 codec 调用和结果汇总
- 不把 `ffmpeg/lame` 等传统 codec 调用逻辑混入主训练入口

## Execution Strategy

### Phase 1: Evaluation Scaffold

1. 建立 `evals/` 目录与说明
2. 约定 benchmark 输入输出格式
3. 固定一组测试样本与 neural codec checkpoint

### Phase 2: MP3 Baseline

1. 调用 `MP3` 编码器生成压缩结果
2. 解码回 waveform
3. 与 neural codec 重建统一比较

### Phase 3: Metrics and Comparison

1. 统计文件大小 / 压缩率 / 码率
2. 统计重建保真指标
3. 汇总对比表与试听样例

## Acceptance Criteria

- `evals/` 下有独立的传统 codec benchmark 代码
- 至少完成 `MP3` 与当前 neural codec 的一版可复现实验
- 结果可以直接回答“传统 codec 与 neural codec 在本任务上的优劣”

## Immediate Next Step

先在 `evals/` 下搭好 benchmark 骨架，并优先落 `MP3` baseline。
