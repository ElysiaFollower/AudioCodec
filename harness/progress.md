Owner: ely
Status: active
Last reviewed: 2026-05-05

# Progress

## 2026-05-05

- 将项目定位从课程型 MP3 对比 baseline 更新为科研型 temporal redundancy / context sequence modeling in neural speech codecs。
- 新增 ADR 0002，明确 Mamba 不是唯一假设；主问题是上下文序列建模在 codec 表示链上的收益边界。
- 新增 assumption 文档，拆分 waveform、latent、post-RVQ、code-prior 和 Mamba efficiency 的验证假设。
- 新增插入点架构文档，定义 P0 waveform-proximal、P1 latent bottleneck、P2 post-RVQ refiner、P3 code-prior。
- 新增科研实验协议，固定 reporting 字段、Stage 0-4 实验矩阵、预期结果和结论判定规则。
- 新增启动手册，记录第一天实现顺序和不要先做的事项。
- 将 `TASK-007 Report Writing` 从 active 归档到 `plans/archive/`，新增 active `TASK-008 Context Sequence Modeling Research`。
- 将旧 `baseline-neural-codec.md` 标记为 archived，将课程报告目录标记为 archived。
- 用户指出旧文档和新文档混在 `docs/` 中会干扰新设计；已将课程阶段 ADR、旧架构、旧训练/benchmark 手册和报告目录整体归档到 `docs/archive/course-project/`。
- 新增 `docs/architecture/current-codec-baseline.md`，用于替代旧 `encodec-inspired` 文档在活跃科研文档中的角色。
- 用户指出上一版仍然没有把 idea 定义清楚，且过早约束了开发流程。已将过细的 assumption/protocol/how-to/插入点流程移动到 `docs/archive/draft-*`，新增 `docs/idea.md` 作为中心定义文档，并把 `TASK-008` 改成 idea definition 任务而非实现任务。
- 用户要求“用中文写”。已将当前活跃文档 `docs/idea.md`、`docs/overview.md`、ADR 0002、current codec baseline、archive README、TASK-008 和 README 的相关入口改为中文表达。
- 验证：
  - `git diff --check` 通过。
  - `PYTHONPATH=src python -m unittest discover -s tests -v` 在当前 Python 3.13 环境失败，因为未安装 `torch`。
  - `conda run -n audiocodec env PYTHONPATH=src python -m unittest discover -s tests -v` 因 macOS OpenMP runtime 冲突 abort。
  - `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，25 tests OK。
- 用户补充：当前机器是 macOS 开发机；正式训练会将仓库 clone 到 `4 x A100` Linux 机器上执行。因此 macOS OpenMP 问题只作为本地验证注意事项，不作为训练阻塞。
- 使用 `harness-project-initializer-zh` 的 scaffold 非破坏性补齐 repo-native harness：新增 `init.sh`、`scripts/harness-check.sh`、`harness/bootstrap-contract.md`、`harness/decisions.md`、`harness/observability.md`、`harness/evaluator-rubric.md`、`harness/quality.md`。
- 将 `AGENTS.md` 从项目管理指南重写为 74 行短路由，包含元数据、事实来源地图、启动流程、硬性规则、验证阶梯和完成定义，要求 agent 会话结束前维护 progress、feature list 和 handoff。
- 将 `harness/feature_list.json` 改成可校验状态机；当前唯一 active item 是 `phase-1-research-intake-and-experiment-prep`，表示一阶段调研收集与实验准备。
- 将 `plans/active/TASK-008-context-sequence-modeling-research.md` 从 idea 待确认任务更新为“一阶段：idea 相关研究收集与实验准备”任务合同。
- 更新 `docs/overview.md` 和 `README.md` 的当前阶段说明，避免后续 agent 继续停留在“先审 idea”或旧课程项目状态。
- 将 GitHub issue template 从 course project task 改为 research task；更新 `pyproject.toml` description，去掉课程项目定位。
- 本轮验证：
  - `./scripts/harness-check.sh` 通过，0 warnings。
  - `./init.sh` 通过，并打印恢复路径和验证命令。
  - `git diff --check` 通过。
  - `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
  - `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，25 tests OK。
- 用户确认当前科研方向与管理节奏：先开新分支，把项目作为科研项目固化需求定义和目标；后续再细化 idea 到实现层、写实现计划、实现代码并收集实验/benchmark 数据。
- 已切换到分支 `feat/context-modeling`，用于承载 context modeling research baseline。
- 首个提交范围确定为科研定义基线：项目重定位、idea、ADR、当前 codec baseline、harness、TASK-008、旧课程材料归档和仓库入口更新；不在该提交中新增上下文建模代码或固定 Mamba 路线。
- 后续提交节奏确定为阶段小提交：研究收集产物、实验/结果收集规划、实现计划、代码实现和实验记录分开提交。
