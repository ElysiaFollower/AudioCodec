Owner: ely
Status: active
Last reviewed: 2026-04-14

# Linux 训练说明

## 目标

在 Linux 训练机上完成四件事：

1. 跑通 baseline smoke test
2. 跑完整 baseline 训练
3. 跑 mel loss 消融训练并与 baseline 对比
4. 跑 `encodec-inspired` 主力训练并试听结果

## 环境

建议先用仓库根目录的 `environment.yaml` 创建环境。

如果需要 TensorBoard，可确保环境里安装了 `tensorboard`，然后在训练命令中加 `--tensorboard`。

## 数据准备

将 `LibriSpeech dev-clean` 或其兼容目录放到 Linux 训练机的本地路径，然后通过以下任一方式指定：

1. 修改 `configs/*.json` 中的 `dataset.root`
2. 运行训练命令时传 `--dataset-root`

当前仓库中的默认配置已经对齐到训练机上的 `train.100` 路径，因此命令模板默认不再显式传 `--dataset-root`。如果后续换机器或换数据目录，再用运行时参数覆盖即可。

## 推荐训练顺序

### 1. Baseline smoke test

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/baseline.json \
  --output-dir artifacts/linux-baseline-smoke \
  --smoke-test \
  --limit-train-examples 10 \
  --device cuda
```

作用：

- 检查数据读取是否正常
- 检查模型前向与反向是否正常
- 检查 checkpoint、日志和音频样例是否成功导出

### 2. Baseline main run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/baseline.json \
  --output-dir artifacts/linux-baseline-main \
  --device cuda \
  --tensorboard
```

### 3. Mel loss ablation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/ablation-mel-loss.json \
  --output-dir artifacts/linux-mel-ablation \
  --device cuda \
  --tensorboard
```

### 4. Encodec-inspired main run

当 baseline 的试听结果不足以达到“可用”级别时，优先切换到这一路线：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/encodec-inspired.json \
  --output-dir artifacts/linux-encodec-inspired \
  --device cuda \
  --tensorboard
```

这条配置默认使用：

- `SEANet` 风格 encoder / decoder
- `EMA + k-means init` 的 `RVQ`
- 约 `12 kbps` 名义码率
- `mel loss` 作为默认开启的感知增强项

## 输出目录结构

所有训练结果都会写到 `--output-dir` 指定的目录下，主要包括：

- `resolved_config.json`
  训练时实际使用的完整配置快照
- `metrics.jsonl`
  逐步追加的训练与验证指标日志
- `checkpoints/step-XXXXXX.pt`
  周期性 checkpoint
- `checkpoints/best.pt`
  当前验证集上最优的 checkpoint
- `samples/step-XXXXXX-source.wav`
  验证样例原音频
- `samples/step-XXXXXX-reconstruction.wav`
  对应重建音频
- `tensorboard/`
  仅在启用 `--tensorboard` 时生成

## TensorBoard

如果训练命令中开启 `--tensorboard`，训练脚本会把标量写到：

```bash
<output-dir>/tensorboard
```

启动命令：

```bash
tensorboard --logdir artifacts
```

如果你只想看某一组实验，也可以把 `--logdir` 指向单个输出目录。

## 当前限制

当前训练脚本已经覆盖 baseline 训练所需的核心流程，但还没有内置：

- `MP3` baseline 压缩与对比
- `PESQ` / `STOI` 等额外评测
- 训练结束后的自动汇总报告

这些步骤建议在训练结果稳定后单独实现。
