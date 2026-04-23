Owner: ely
Status: active
Last reviewed: 2026-04-15

# AudioCodec

本仓库用于推进一个面向 `speech` 的神经音频编解码项目。

- 当前目标：在项目约束下，先完成一个 `Encoder-Decoder + RVQ` 的 speech neural codec baseline，并与 `MP3` 做压缩率和重建质量对比。

建议先读：

- [项目总览](./docs/overview.md)
- [基线架构说明](./docs/architecture/baseline-neural-codec.md)
- [Encodec-Inspired 架构说明](./docs/architecture/encodec-inspired-codec.md)
- [当前执行计划](./plans/active/TASK-006-traditional-codec-benchmark.md)
- [最新归档计划：训练对齐阶段](./plans/archive/TASK-005-encodec-training-alignment.md)
- [范围决策 ADR](./docs/adr/0001-course-project-scope.md)

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

## 当前实验配置

- `configs/baseline.json`
  轻量 `Conv + RVQ + Conv` baseline，主要用于最小可交付和消融。
- `configs/ablation-mel-loss.json`
  在 baseline 上打开 `mel loss` 的对照实验。
- `configs/encodec-inspired.json`
  当前主力路线，使用 `SEANet + EMA RVQ`，目标是把音质提升到可用级别。
- `configs/ablation-adversarial-msstft-balanced.json`
  当前已验证可用的高保真训练路线，使用 `MS-STFT discriminator + feature matching + balancer`。

## 评测目录

- `evals/`
  承载传统 codec baseline、benchmark 脚本和结果汇总，避免与 `src/` 的主训练代码耦合。
