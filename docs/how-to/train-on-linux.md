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

smoke test 建议前台运行，便于立刻看到报错：

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

### 后台启动模板

完整训练默认建议后台运行，并把日志写到 `logs/`：

```bash
mkdir -p logs artifacts
LOG=logs/<run-name>.$(date +%F-%H%M%S).log

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python -u scripts/train_codec.py \
    --config <config-path> \
    --output-dir <output-dir> \
    --device cuda \
    --tensorboard \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
```

查看训练日志：

```bash
tail -f "$LOG"
```

### 单条音频 overfit 调试

当你怀疑“是代码逻辑有 bug，还是架构本身太弱”时，先跑这一层。

这条命令会：

- 只使用一条指定音频
- 训练和验证都复用同一条样本
- 固定取同一个前段 clip，而不是随机裁剪
- 把 batch size 固定成 `1`
- 保持与正常训练相同的 clip 时长配置，便于做实现验机

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_codec.py \
  --config configs/encodec-inspired.json \
  --output-dir artifacts/debug-overfit-one \
  --steps 2000 \
  --device cuda \
  --overfit-example-path /absolute/path/to/example.flac \
  --tensorboard
```

判断标准：

- 如果这条单样本训练长期无法把重建做得非常接近原音，优先怀疑数据流、损失、量化器或训练逻辑存在问题。
- 如果它能较好地记住这一条样本，但正常训练质量仍差，问题更可能在架构容量、码率预算或训练目标设计。

这不是绝对定理。对于带离散瓶颈的 codec，“不能完美重建”不一定百分之百说明代码错了，因为瓶颈预算、本身损失设计和优化设置也会限制上界。但如果连单条样本都明显学不住，通常应该先查实现，而不是先怪架构。

注意：

- 请为每次 overfit 调试使用新的 `--output-dir`
- 当前训练脚本不支持 resume，同一个输出目录重复运行会直接报错，避免混合旧日志和新结果

### Encodec-like loss 快速验机

这条实验只改 `loss`，不改 `SEANet + EMA RVQ` 架构，目的是快速判断：

- 去掉 `mel loss`
- 降低 waveform 权重
- 提高频谱重建权重

是否会让短程 overfit 更快接近可听的自然人声。

这不是完整复现 `Encodec` 的训练目标，因为当前仓库还没有：

- `MS-STFT discriminator`
- `feature matching`

所以这条配置只能测试“loss 方向是否更对”，不能把结果直接解读为“已经等价于 Encodec”。

对应配置：

```text
configs/ablation-encodec-like-loss.json
```

短程 overfit 命令：

```bash
mkdir -p logs artifacts
LOG=logs/debug-overfit-encodec-loss.$(date +%F-%H%M%S).log

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python -u scripts/train_codec.py \
    --config configs/ablation-encodec-like-loss.json \
    --output-dir artifacts/debug-overfit-encodec-loss \
    --steps 4000 \
    --device cuda \
    --overfit-example-path /absolute/path/to/example.flac \
    --tensorboard \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
```

建议和 `configs/encodec-inspired.json` 的 overfit 结果对照听：

- `step 1000`
- `step 2000`
- `step 4000`

如果新配置在这些固定步数下更早减少“电流音 / 电颤感”，说明当前 loss 更贴近听感目标。

### 2. Baseline main run

```bash
mkdir -p logs artifacts
LOG=logs/linux-baseline-main.$(date +%F-%H%M%S).log

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python -u scripts/train_codec.py \
    --config configs/baseline.json \
    --output-dir artifacts/linux-baseline-main \
    --device cuda \
    --tensorboard \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
```

### 3. Mel loss ablation

```bash
mkdir -p logs artifacts
LOG=logs/linux-mel-ablation.$(date +%F-%H%M%S).log

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python -u scripts/train_codec.py \
    --config configs/ablation-mel-loss.json \
    --output-dir artifacts/linux-mel-ablation \
    --device cuda \
    --tensorboard \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
```

### 4. Encodec-inspired main run

当 baseline 的试听结果不足以达到“可用”级别时，优先切换到这一路线：

```bash
mkdir -p logs artifacts
LOG=logs/linux-encodec-inspired.$(date +%F-%H%M%S).log

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python -u scripts/train_codec.py \
    --config configs/encodec-inspired.json \
    --output-dir artifacts/linux-encodec-inspired \
    --device cuda \
    --tensorboard \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
```

这条配置默认使用：

- `SEANet` 风格 encoder / decoder
- `EMA + k-means init` 的 `RVQ`
- 约 `12 kbps` 名义码率
- `mel loss` 作为默认开启的感知增强项
- 默认关闭 `mixed precision`，优先保证数值稳定

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
