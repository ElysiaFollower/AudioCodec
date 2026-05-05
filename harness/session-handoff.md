Owner: ely
Status: active
Last reviewed: 2026-05-05

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 已有提交：`8c54781 docs: define context modeling research baseline`
- 已有提交：`11e4bf3 docs: add context modeling research intake`
- 当前待提交目标：Commit 3，建立实现层实验计划与结果采集协议。
- Commit 3 新增/修改范围：
  - added: `docs/research/context-modeling-experiment-plan.md`
  - modified: `docs/research/context-modeling-intake.md`
  - modified: `README.md`
  - modified: `docs/overview.md`
  - modified: `evals/README.md`
  - modified: `plans/active/TASK-008-context-sequence-modeling-research.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh`
  - 结果：通过，`Harness 检查通过，共 0 个警告。`
- `./init.sh`
  - 结果：通过，能打印当前阶段、建议阅读文件、环境命令、聚焦验证、完整验证和 Linux A100 smoke 命令。
- `git diff --check`
  - 结果：通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
  - 结果：通过，能打印训练 CLI 参数。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
  - 结果：通过，`Ran 25 tests`, `OK`。

## 本会话改动

- 按用户确认的计划创建分支 `feat/context-modeling`。
- 已创建首个科研定义基线提交：`8c54781 docs: define context modeling research baseline`。
- 建立第一版研究收集主表 `docs/research/context-modeling-intake.md`，覆盖 SoundStream、EnCodec、DAC、AudioDec、LMCodec、Convolutional Transformer、BigCodec、SpeechTokenizer、Mimi/Moshi、Mamba/Mamba-2。
- 新增 `docs/research/context-modeling-attempt-log.md`，记录首轮 primary source verification，明确本轮不下载 PDF、不 clone 外部 repo。
- 新增 `docs/research/context-modeling-blockers.md`，记录官方代码缺口、entropy coding 覆盖不足和 Mamba codec 直接证据不足。
- README、overview 和 TASK-008 已指向研究收集产物，并把下一步改为实现层细化与实验/结果采集规划。
- 新增 `docs/research/context-modeling-experiment-plan.md`，定义 E0-E5 最小实验矩阵、context config 接口、representation export manifest、code-prior entropy 公式、统一结果表字段、数据命令和 Commit 4+ 顺序。
- 明确当前 SEANet baseline 已带 `SkipLSTM`，后续实验不能把 baseline 自带 LSTM 当成新增 context 收益。
- 修正 `evals/README.md` 中传统 codec benchmark 手册链接到 `docs/archive/course-project/how-to/run-traditional-codec-benchmark.md`。
- 使用 `harness-project-initializer-zh` scaffold 补齐缺失的 harness 工件，并替换全部 scaffold 占位符。
- 将 `AGENTS.md` 重写为短路由入口，明确要求 agent 自主维护 `harness/progress.md`、`harness/feature_list.json`、`harness/session-handoff.md`。
- 将当前 active item 更新为一阶段研究收集与实验准备，而不是继续停留在 idea 待确认。
- 将 TASK-008 改成一阶段任务合同，范围包括论文/代码收集、插入层级证据整理、matched baseline 和实验准备。
- 新增 `scripts/harness-check.sh`，检查 AGENTS 长度、硬规则数量、占位符、feature schema、WIP=1、passing evidence 和 handoff 标题。
- 更新 README、overview、issue template 和 package description，减少旧课程项目语境对新 agent 的干扰。
- 首个提交只固化科研项目定义，不新增上下文建模代码、不固定 Mamba 路线、不创建最终实验矩阵。

## 本会话决策

- 分支名使用 `feat/context-modeling`，按用户偏好覆盖默认 `dev/` 前缀。
- 首个提交采用阶段小提交策略，范围限定为“科研定义基线”。
- `AGENTS.md` 只保留入口、事实来源、硬规则、验证阶梯和完成定义；专题细节放入 docs、harness、plans 或脚本。
- 当前唯一 active feature 是 `phase-1-research-intake-and-experiment-prep`。
- 一阶段先围绕 `docs/idea.md` 做研究收集和实验准备；不直接实现 Mamba-only codec 或固定 latent/code-level context。
- Commit 4 不直接实现 Mamba；先实现 representation export 和 result schema，保证后续 code-prior / latent context / post-RVQ context 共享同一结果采集路径。
- 后续如果 agent 忘记更新状态文件，应优先增强 `scripts/harness-check.sh`，而不是继续往 `AGENTS.md` 堆规则。

## 仍损坏或未验证

- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖和版本尚未固定。
- 一阶段研究收集主表和实现层实验计划已创建，但仍等待用户审查方向是否对齐；因此 active feature 暂不标为 passing。
- 首轮未下载论文 PDF、未 clone 外部仓库，也未验证外部代码可运行性。
- LMCodec 和 Convolutional Transformer 暂未找到官方代码；Mamba codec 直接证据仍不足。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- Harness：`./scripts/harness-check.sh` 通过，0 warnings。
- 静态检查：`git diff --check` 通过。
- CLI sanity：训练脚本 help 在 `audiocodec` conda 环境通过。
- 单测：25 tests OK。
- 进度文件同步：`harness/feature_list.json`、`harness/progress.md`、`harness/session-handoff.md` 已同步。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

等待用户审查 Commit 3 的方向。若确认对齐，下一步进入 Commit 4：实现 representation export 和 result schema，不训练新模型、不引入 Mamba 依赖。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
