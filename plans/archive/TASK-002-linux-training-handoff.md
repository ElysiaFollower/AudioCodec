Owner: ely
Status: archived
Last reviewed: 2026-04-14

# TASK-002: Linux 训练 handoff 与 mel loss 消融

## Archive Note

Linux 训练 handoff、mel 消融配置和稳定操作手册已经落地；后续实验不再以本计划为主线推进。

## 目标

为 Linux 训练机准备一组可直接执行的训练方案，覆盖：

- baseline smoke test
- baseline full run
- mel loss 消融 run

## 背景

当前开发机是 macOS，本地环境主要用于代码开发和前向验证，不作为正式训练平台。

因此需要将训练环节显式整理成 handoff 形式，确保迁移到 Linux 后无需重新梳理命令和输出路径。

## 约束

- 训练将在 Linux 机器上进行
- 数据路径依赖目标机器本地挂载，不应写死进仓库
- baseline 与 mel ablation 必须尽可能只差一个变量

## 验收标准

满足以下条件即可关闭本任务：

1. 存在一份可直接运行的 mel 消融配置
2. 存在一份 Linux 训练说明文档
3. 能明确说明输出路径、日志文件和 checkpoint 位置
4. 若启用 TensorBoard，训练脚本能输出对应日志目录

## 执行顺序

1. 先跑 baseline smoke test，确认训练链路正常
2. 再跑 baseline full run
3. 最后跑 mel loss ablation
4. 对比两组实验的 `metrics.jsonl`、验证 loss、重建样例和听感结果

## 推荐比较口径

baseline 与 mel ablation 的比较应固定以下条件：

- 相同数据集和切分
- 相同步数预算
- 相同 batch size
- 相同随机种子
- 相同输出检查频率

变化项仅保留：

- `baseline.json`: `mel_weight = 0.0`
- `ablation-mel-loss.json`: `mel_weight = 0.5`

## 当前产物

- 配置文件：`configs/baseline.json`
- 配置文件：`configs/ablation-mel-loss.json`
- 训练入口：`scripts/train_codec.py`
- 长期说明：`docs/how-to/train-on-linux.md`
