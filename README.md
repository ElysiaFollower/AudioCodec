Owner: ely
Status: active
Last reviewed: 2026-05-05

# AudioCodec

本仓库用于推进一个面向 `speech` 的 neural codec / speech tokenizer 科研项目。

- 当前科研目标：研究在 `waveform -> downsampled latent -> RVQ embedding/codes -> code sequence/prior` 这条表示链上，上下文序列建模应在哪一层介入，才能把时间冗余转化为同码率质量收益、entropy-coded bitrate 收益或长音频 token modeling 效率收益。
- 当前工程基底：已经完成 `SEANet + EMA RVQ` speech codec baseline，并已有 `2 / 4 / 8 / 12 kbps` neural ladder 与传统 codec benchmark。
- 当前方法立场：Mamba 是 selective SSM 候选模型，不是唯一假设；后续实验必须同时比较 `TCN / LSTM / Transformer / Mamba` 等上下文模型。
- 当前阶段：一阶段先调研并收集 idea 相关研究与代码，整理证据和实验准备，再进入开发实现。

建议先读当前科研主线：

- [项目总览](./docs/overview.md)
- [研究想法](./docs/idea.md)
- [ADR 0002：围绕上下文插入位置定义科研问题](./docs/adr/0002-temporal-redundancy-research-scope.md)
- [当前 codec baseline](./docs/architecture/current-codec-baseline.md)
- [上下文建模研究收集](./docs/research/context-modeling-intake.md)
- [当前执行计划](./plans/active/TASK-008-context-sequence-modeling-research.md)
- [Harness 交接](./harness/session-handoff.md)

历史 baseline、课程报告与旧 benchmark 资料已经归档到 [docs/archive](./docs/archive/README.md)，不再定义当前设计。

## 环境安装

推荐直接使用仓库根目录下的 [environment.yaml](/Users/ely/workspace/research/audio/AudioCodec/environment.yaml)。

安装命令：

```bash
conda env create -f environment.yaml
conda activate audiocodec
```

如果环境已经存在，更新命令：

```bash
conda env update -f environment.yaml --prune
conda activate audiocodec
```

## 安装验证

```bash
python scripts/train_codec.py --help
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 数据路径

数据集根路径默认放在 [configs/baseline.json](/Users/ely/workspace/research/audio/AudioCodec/configs/baseline.json) 的 `dataset.root` 字段里，也可以在运行时覆盖：

```bash
python scripts/train_codec.py --dataset-root /path/to/LibriSpeech/dev-clean --smoke-test
```

## 当前科研 Anchor

- `configs/ablation-adversarial-msstft-balanced.json`
  `12 kbps` baseline anchor，使用 `SEANet + EMA RVQ + MS-STFT discriminator + feature matching + balancer`。
- `configs/ablation-adversarial-msstft-balanced-8kbps.json`
  `8 kbps` baseline anchor。
- `configs/ablation-adversarial-msstft-balanced-4kbps.json`
  `4 kbps` baseline anchor。
- `configs/ablation-adversarial-msstft-balanced-2kbps.json`
  `2 kbps` baseline anchor。

历史课程阶段配置，例如 `configs/baseline.json`、`configs/ablation-mel-loss.json` 和 `configs/encodec-inspired.json`，仍保留在仓库中用于复现，但它们不是当前科研计划的起点。

## 当前研究纪律

- 不把 Mamba 收益和上下文建模收益混为一谈。
- 不把 nominal bitrate、entropy-coded bitrate 和 tokens/sec 混为一谈。
- 不在没有 matched baseline 的情况下宣称模型优越性。
- 不让 post-RVQ refiner 引入额外 side channel；传输 payload 仍只能是 RVQ codes。
- 不预设 latent/code-level context 一定优于 waveform-level context；早期上下文是否更好是要被验证的核心假设之一。

## 评测目录

- `evals/`
  承载传统 codec baseline、benchmark 脚本和结果汇总，避免与 `src/` 的主训练代码耦合。
