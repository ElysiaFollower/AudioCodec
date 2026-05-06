<!--
职责：定义本项目被新 agent 无歧义接手的初始化契约。
边界：不要记录业务实现进度；进度放 progress.md，具体任务放 plans/active/。
-->

Owner: ely
Status: active
Last reviewed: 2026-05-06

# 初始化契约

## 自举条件

- 能启动：`PYTHONPATH=src python scripts/train_codec.py --help`
- 能测试：`./scripts/harness-check.sh`、`git diff --check`、`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- 能看进度：`harness/progress.md` 和 `harness/feature_list.json`
- 能接手下一步：`harness/session-handoff.md` 和 `plans/active/`

## 环境

- 技术栈：Python、PyTorch、torchaudio、SEANet-style encoder/decoder、EMA RVQ、traditional codec eval scripts
- 运行时版本：`environment.yaml` 固定 `python=3.11`
- 依赖安装：`conda env update -f environment.yaml --prune && conda activate audiocodec`
- 本地服务：无常驻本地服务；训练和评测通过命令行脚本运行
- 数据边界：speech-only；默认数据集根路径来自 config 的 `dataset.root` 或训练命令的 `--dataset-root`

## 标准命令

```sh
conda env update -f environment.yaml --prune
conda activate audiocodec

./scripts/harness-check.sh
git diff --check
PYTHONPATH=src python scripts/train_codec.py --help

conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v

PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda
```

## 初始化验收清单

- [x] 从仓库入口能找到当前项目目标和 idea。
- [x] 从仓库入口能找到环境安装方式。
- [x] 至少一个 harness 验证命令可运行。
- [x] feature list 能表达当前 WIP 和 passing evidence。
- [x] 新 agent 只看仓库能回答：是什么、怎么跑、怎么测、当前进度、下一步。
- [ ] Linux A100 smoke 需要在训练机上重新验证。

## 已知缺口

- 当前 macOS 开发机完整单测需要 `KMP_DUPLICATE_LIB_OK=TRUE` 绕过 OpenMP runtime 冲突；这不是 Linux A100 训练命令的一部分。
- 尚未在本轮运行真实训练 smoke。
- Mamba/SSM 依赖尚未固定；只有 long-context diagnostics 通过 gate 后才评估是否引入 Mamba。
- 研究收集、方向定义和 E0-E2 本地工具链已完成；当前 active item 是 `long-range-redundancy-diagnostics-and-context-route`，下一步是在训练机用真实 checkpoint/manifest 跑 pipeline，并下载轻量结果包分析。
