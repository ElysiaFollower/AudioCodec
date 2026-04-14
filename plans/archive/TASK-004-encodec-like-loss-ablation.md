Owner: ely
Status: archived
Last reviewed: 2026-04-14

# TASK-004 Encodec-Like Loss Ablation

## Archive Note

本实验已完成一次短程 overfit 验机，证明 `encodec-like` loss 方向值得保留，但不足以单独解决全数据训练音质问题；后续将把该结论并入新的系统性对齐计划。

## Goal

在不改 `SEANet + EMA RVQ` 架构的前提下，快速验证更接近 `Encodec` 参考配方的 loss 方向，是否能让短程 overfit 更早得到更自然的重建音色。

## Background

- 当前 `encodec-inspired` 路线已经能在单样本 overfit 中逐步逼近原声，但 `1000/2000 step` 仍有明显电流音。
- 参考 run `/Users/ely/workspace/research/audio/MambaCodec/outputs/dora/xps/67ecbee7/.hydra/config.yaml` 显示：
  - `losses.l1 = 0.1`
  - `losses.msspec = 2.0`
  - `losses.mel = 0.0`
  - 另有 `adv + feat`，这是当前仓库尚未实现的部分。
- 因此，这一轮先测试“去掉 mel、降低 waveform 权重、提高频谱重建权重”是否能在短程试听上更接近参考系统。

## Hypothesis

- 当前系统的主要瓶颈之一，是优化目标和真实听感之间仍有偏差。
- 如果 loss 方向更接近参考系统，即使还没有引入 discriminator，也应当在固定单样本 overfit 的 `1000/2000/4000 step` 上更早减少电流音和电颤感。

## In Scope

- 新增实验配置 [configs/ablation-encodec-like-loss.json](/Users/ely/workspace/research/audio/AudioCodec/configs/ablation-encodec-like-loss.json)
- 保持模型架构、量化器、训练脚本不变
- 使用固定单样本 overfit 做短程验机

## Out Of Scope

- 在这一轮中引入 `MS-STFT discriminator`
- 在这一轮中引入 `feature matching`
- 直接将该实验流程写入稳定的 Linux 训练手册

## Experiment Config

相对于 [configs/encodec-inspired.json](/Users/ely/workspace/research/audio/AudioCodec/configs/encodec-inspired.json)：

- `waveform_weight: 1.0 -> 0.1`
- `stft_weight: 1.0 -> 2.0`
- `mel_weight: 0.5 -> 0.0`

## Command

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

## Evaluation

重点对照这些固定步数的主观听感：

- `step 1000`
- `step 2000`
- `step 4000`

对比对象：

- [configs/encodec-inspired.json](/Users/ely/workspace/research/audio/AudioCodec/configs/encodec-inspired.json) 的 overfit 结果
- [configs/ablation-encodec-like-loss.json](/Users/ely/workspace/research/audio/AudioCodec/configs/ablation-encodec-like-loss.json) 的 overfit 结果

## Acceptance Criteria

- 新配置能够正常启动 overfit 训练
- 不出现 `NaN/Inf`
- 在固定步数试听上，如果更早减少电流音，则说明这一 loss 方向值得继续投入
- 如果无明显主观改善，则归档该计划，不进入稳定训练文档
