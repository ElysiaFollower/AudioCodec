Owner: ely
Status: active
Last reviewed: 2026-05-05

# AudioCodec Agent Guide

## 项目一句话

本仓库是一个面向 `speech` 的 neural audio codec 科研工作区：以 `SEANet/VAE-style encoder + RVQ` 为稳定主干，研究时间上下文建模应该在 `waveform -> latent -> RVQ embedding/codes -> code prior` 哪一层介入，最终用 speech benchmark 与 MP3 等传统压缩比较压缩率和保真率。

## 当前阶段

- 当前 idea 已基本明确，唯一主入口是 `docs/idea.md`。
- 一阶段任务是调研并初步收集 idea 相关研究与代码，整理可验证假设，然后准备开发和实验。
- 当前不是直接重写 codec、固定 Mamba 方案、扩展多数据集或跑完整训练矩阵。

## 事实来源地图

- `README.md`：项目入口、环境、常用命令和当前科研 anchor。
- `docs/overview.md`：项目定位、范围和活跃文档地图。
- `docs/idea.md`：当前研究 idea 的 source of truth。
- `docs/architecture/current-codec-baseline.md`：稳定 codec baseline 与第一阶段保持不变的变量。
- `docs/adr/`：长期设计决策；重要方向变化必须补 ADR 或更新现有 ADR。
- `plans/active/`：当前任务合同；默认只能有一个 active plan。
- `harness/feature_list.json`：功能状态机，受 WIP=1 和 evidence 约束。
- `harness/progress.md`：跨会话进度日志。
- `harness/decisions.md`：会影响后续 agent 判断的重要决策。
- `harness/session-handoff.md`：最新可恢复状态、风险、验证证据和下一步。
- `evals/`：传统 codec benchmark、导出、评分和评测样本。
- `docs/archive/`：历史课程阶段和旧草稿；只作参考，不覆盖活跃文档。

## 启动流程

1. 先运行 `./init.sh`，读取它打印的启动、验证和下一步命令。
2. 读取 `docs/idea.md`、`plans/active/`、`harness/feature_list.json`、`harness/session-handoff.md`。
3. 确认 `feature_list.json` 最多一个 `active`，并把本轮工作绑定到该 active item。
4. 如果任务超过半天、跨多个文件或涉及设计取舍，先更新或创建 `plans/active/TASK-xxx.md`。

## 硬性规则

1. 仓库是唯一事实来源；不要把项目状态只留在聊天里。
2. `AGENTS.md` 只做路由，不堆积专题规则；细节进入 `docs/`、`harness/`、`plans/`、脚本或测试。
3. 默认 WIP=1；新增 active feature 或 active plan 前，先关闭、阻塞或归档旧项。
4. `docs/idea.md` 是当前 idea 的最高优先级来源；归档文档不能反向约束当前方向。
5. 一阶段先做 idea 相关研究收集和实验准备；未形成证据前不要直接宣称实现路线正确。
6. Mamba 只是上下文模型候选；结论必须与 TCN、LSTM、Transformer 或更简单 baseline 区分。
7. 不混用 nominal bitrate、entropy-coded bitrate、token efficiency 和重建保真率。
8. 只针对 speech 领域；扩到音乐、通用音频或语义 tokenizer 需要新决策记录。
9. 不让 post-RVQ refiner 引入额外 side channel；传输 payload 默认只能是 RVQ codes。
10. `passing` 必须有验证命令和 evidence；不要凭主观判断标完成。
11. 永久文档保留最少元数据：Owner、Status、Last reviewed。
12. 会话结束前必须更新 progress、feature list、handoff，并记录未验证内容。

## 验证阶梯

- Harness 门禁：`./scripts/harness-check.sh`
- 静态检查：`git diff --check`
- CLI sanity：`PYTHONPATH=src python scripts/train_codec.py --help`
- 本机完整单测：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`

macOS 上 `KMP_DUPLICATE_LIB_OK=TRUE` 只是本地 OpenMP workaround，不要写进 Linux 训练命令。

## 完成定义

一次任务完成必须同时满足：

- 行为或文档变化符合 active plan 的验收标准；
- 相关验证命令已运行，失败项被解释并写入 handoff；
- `harness/feature_list.json` 状态和 evidence 与实际一致；
- `harness/progress.md` 记录本轮进展和下一步；
- 重要方向取舍写入 `harness/decisions.md` 或 `docs/adr/`；
- `harness/session-handoff.md` 能让新 agent 三分钟内恢复；
- 未留下无说明的临时日志、缓存、模型输出或机器特定状态。
