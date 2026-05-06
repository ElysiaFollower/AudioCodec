Owner: ely
Status: active
Last reviewed: 2026-05-06

# AudioCodec

本仓库用于推进一个面向 `speech` 的 neural codec / speech tokenizer 科研项目。

- 当前科研目标：研究 neural speech codec 已经具备局部时序建模后，语音中是否仍存在可利用的长程时间冗余；如果存在，它在 `waveform -> downsampled latent -> RVQ embedding/codes -> code sequence/prior` 哪个表示层级最容易转化为同码率质量收益、entropy-coded bitrate 收益或长音频 token modeling 效率收益。
- 当前工程基底：已经完成 `SEANet + EMA RVQ` speech codec 训练、导出和评测代码；当前分支的实验默认自己训练所需 4kbps baseline，不依赖历史分支或本地旧 checkpoint。
- 当前方法立场：Mamba 是 selective SSM 候选模型，不是唯一假设；后续实验必须同时比较 `TCN / LSTM / Transformer / Mamba` 等上下文模型。
- 当前阶段：Phase 1 representation export、Phase 2 frozen diagnostics、Phase 3 analytic/trained code-prior、统一 pipeline、结果聚合和轻量结果包脚本已实现；一键脚本已改为先构建 manifest、必要时训练当前分支自己的 4kbps baseline，再继续诊断和打包。

建议先读当前科研主线：

- [项目总览](./docs/overview.md)
- [研究想法](./docs/idea.md)
- [ADR 0002：围绕上下文插入位置定义科研问题](./docs/adr/0002-temporal-redundancy-research-scope.md)
- [当前 codec baseline](./docs/architecture/current-codec-baseline.md)
- [上下文建模研究收集](./docs/research/context-modeling-intake.md)
- [长程时间冗余相关研究补充](./docs/research/long-range-redundancy-intake.md)
- [上下文建模实现与实验计划](./docs/research/context-modeling-experiment-plan.md)
- [当前执行计划](./plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md)
- [Harness 交接](./harness/session-handoff.md)

历史 baseline、课程报告与旧 benchmark 资料已经归档到 [docs/archive](./docs/archive/README.md)，不再定义当前设计。

## 环境安装

macOS / CPU 本地开发使用 [environment.yaml](/Users/ely/workspace/research/audio/AudioCodec/environment.yaml)。Linux A100 训练机使用 [environment-linux-cuda.yaml](/Users/ely/workspace/research/audio/AudioCodec/environment-linux-cuda.yaml)，它固定 `pytorch=2.5.1`、`torchaudio=2.5.1` 和 `pytorch-cuda=12.1`，避免 pip 版 PyTorch 与系统 NCCL/CUDA 动态库混用。

macOS / CPU 安装命令：

```bash
conda env create -f environment.yaml
conda activate audiocodec
```

如果环境已经存在，更新命令：

```bash
conda env update -f environment.yaml --prune
conda activate audiocodec
```

Linux A100 建议新建干净环境：

```bash
conda env create -f environment-linux-cuda.yaml
conda activate audiocodec-cu121
```

如果旧 `audiocodec` 环境已经出现 `libtorch_cuda.so` / NCCL symbol 错误，不建议在原环境上修补；直接使用新的 `audiocodec-cu121` 环境。

## 安装验证

```bash
python - <<'PY'
import torch, torchaudio
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
python scripts/train_codec.py --help
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 数据路径

数据集根路径默认放在 [configs/baseline.json](/Users/ely/workspace/research/audio/AudioCodec/configs/baseline.json) 的 `dataset.root` 字段里，也可以在运行时覆盖：

```bash
python scripts/train_codec.py --dataset-root /path/to/LibriSpeech/dev-clean --smoke-test
```

## 一键训练与诊断

当前分支的主入口是 [scripts/run-context-prior-pipeline.sh](/Users/ely/workspace/research/audio/AudioCodec/scripts/run-context-prior-pipeline.sh)。默认不需要外部 `checkpoint` 或预先写好的 `manifest`：脚本会用 4kbps config 构建 test manifest，把 baseline codec 训练到 `evals/outputs/context-modeling/codec-baseline/checkpoints/best.pt`，然后继续 representation export、diagnostics、code-prior 训练、结果聚合和轻量打包。

Linux A100 上建议先用 dry-run 看完整命令，再正式运行：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=4 PYTHON_BIN=python scripts/run-context-prior-pipeline.sh \
  --dataset-root /path/to/LibriSpeech/train-clean-100 \
  --output-root evals/outputs/context-modeling \
  --codec-device cuda \
  --device cuda \
  --train-device cuda \
  --train-steps 1000 \
  --train-sequence-length 512 \
  --train-batch-size 8 \
  --dry-run
```

去掉 `--dry-run` 后开始真实运行：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHON_BIN=python scripts/run-context-prior-pipeline.sh \
  --dataset-root /path/to/LibriSpeech/train-clean-100 \
  --output-root evals/outputs/context-modeling \
  --codec-device cuda \
  --device cuda \
  --train-device cuda \
  --train-steps 1000 \
  --train-sequence-length 512 \
  --train-batch-size 8 \
  2>&1 | tee logs/context-modeling-$(date +%Y%m%d-%H%M%S).log
```

如果 `evals/outputs/context-modeling/codec-baseline/checkpoints/best.pt` 已由同一次 self-contained pipeline 产出，脚本会复用它；换 `--output-root` 或加 `--force-codec-training` 可启动一套新的 baseline 训练。`--checkpoint` 和 `--manifest` 仍保留给复现实验使用，但默认路径不再依赖历史训练产物。

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
- 不把小窗口卷积/TCN 收益包装成长程时间冗余收益。
- 不把“局部时间建模”当成创新点；它是 baseline。
- 不把 nominal bitrate、entropy-coded bitrate 和 tokens/sec 混为一谈。
- 不在没有 matched baseline 的情况下宣称模型优越性。
- 不让 post-RVQ refiner 引入额外 side channel；传输 payload 仍只能是 RVQ codes。
- 不预设 latent/code-level context 一定优于 waveform-level context；早期上下文是否更好是要被验证的核心假设之一。

## 评测目录

- `evals/`
  承载传统 codec baseline、benchmark 脚本和结果汇总，避免与 `src/` 的主训练代码耦合。
- `scripts/run-context-prior-pipeline.sh`
  串起长程冗余诊断流水线，并在末尾默认调用轻量打包脚本。
- `scripts/pack-context-results.sh`
  从 pipeline 输出中白名单复制 `summary / metrics / manifest / run metadata` 和少量试听音频对，生成可下载的小目录；不会复制 `checkpoint.pt`、representation tensor 或整批 reconstruction wav。
