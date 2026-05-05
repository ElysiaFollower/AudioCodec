Owner: ely
Status: active
Last reviewed: 2026-05-05

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 已有提交：
  - `8c54781 docs: define context modeling research baseline`
  - `11e4bf3 docs: add context modeling research intake`
  - `9faade8 docs: define context modeling experiment plan`
  - `267c771 docs: clarify temporal context insertion semantics`
  - `b57556e docs: refocus context modeling on long-range redundancy`
  - `705a275 docs: define long-range redundancy research theme`
  - `51a8b09 docs: add long-range redundancy research refresh`
- 当前待提交目标：创建新的 active implementation spec，替换旧的 context sequence modeling research plan，并把后续工作收束到 long-range redundancy diagnostics 路线。
- 本轮新增/修改范围：
  - added: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - deleted: `plans/active/TASK-008-context-sequence-modeling-research.md`
  - modified: `README.md`
  - modified: `init.sh`
  - modified: `docs/archive/course-project/report/README.md`
  - modified: `harness/bootstrap-contract.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/quality.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./init.sh`
  - 结果：通过，能打印新的 active spec 路径 `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`。
- `./scripts/harness-check.sh`
  - 结果：通过，`Harness 检查通过，共 0 个警告。`
- `git diff --check`
  - 结果：通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null`
  - 结果：通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
  - 结果：通过，能打印训练 CLI 参数。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
  - 结果：通过，`Ran 25 tests`, `OK`。

## 本会话改动

- 用 `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md` 替换旧 active plan，保留 `TASK-008` 编号和 WIP=1。
- 新 spec 明确当前主线：先做 fixed-frame `SEANet + EMA RVQ` baseline 的 `E0-E2` go/no-go diagnostics，再决定是否进入 codec context training。
- 新 spec 按 phase 区分 **实现任务** 和 **实验任务**：
  - Phase 1：representation export 与结果 schema；
  - Phase 2：frozen representation redundancy diagnostics；
  - Phase 3：code-prior entropy baseline；
  - Phase 4：latent pre-RVQ context；
  - Phase 5：post-RVQ embedding context；
  - Phase 6：条件引入 Mamba / SSM 与 early feature；
  - Conditional Branch：dynamic / variable frame-rate。
- 写入默认 gate：long/full context 相比最佳 local baseline 至少带来 `>=5%` estimated entropy bitrate 或 predictability improvement，才进入 codec context training。
- 将 dynamic / variable frame-rate 定义为条件分支；只有 diagnostics 显示收益集中在静音、长元音或 steady-state segment 时另开任务，不混入 fixed-frame context 主线。
- 同步 README、`init.sh`、归档报告链接、feature list、bootstrap contract、quality 和 progress 的 active spec 路由。

## 本会话决策

- 旧 `TASK-008-context-sequence-modeling-research.md` 名称和内容已经不再准确，直接替换而不是继续小修。
- 保留 `TASK-008` 编号，不新建 `TASK-009`，避免制造第二个 active task。
- Phase 1 下一步只实现 representation export、long/full utterance manifest metadata 和 result schema。
- Mamba 不是 Phase 1 入口；只有 long-context 收益先成立，才评估 Mamba 是否值得引入。
- Dynamic / variable frame-rate 是重要后续分支，但当前不进入主线实现。

## 仍损坏或未验证

- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖尚未固定。
- Ultra Low-Bitrate Speech Coding、LMCodec、Single-Codec、TFC、CodecSlime 的官方代码或可复现实验设置仍需后续确认。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- `plans/active` 当前只有 `TASK-008-long-range-redundancy-diagnostics-spec.md` 一个 active 文件。
- Harness：`./scripts/harness-check.sh` 通过，0 warnings。
- 静态检查：`git diff --check` 通过。
- CLI sanity：训练脚本 help 在 `audiocodec` conda 环境通过。
- 单测：25 tests OK。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

提交本轮 spec 后，下一次开发从 Phase 1 开始：实现 representation export、long/full utterance manifest metadata 和 result schema；不训练新模型、不引入 Mamba 依赖、不做 dynamic frame-rate redesign。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
