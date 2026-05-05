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
  - `bf45862 docs: define long-range redundancy implementation spec`
- 当前待提交目标：实现 Phase 1 representation export 和结果 schema，支撑后续 E0-E2 long-range redundancy diagnostics。
- 本轮新增/修改范围：
  - modified: `evals/scripts/export_neural_codec.py`
  - modified: `tests/test_evals_scripts.py`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh`
  - 结果：通过，`Harness 检查通过，共 0 个警告。`
- `git diff --check`
  - 结果：通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help`
  - 结果：通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
  - 结果：通过，`Ran 10 tests`, `OK`。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
  - 结果：通过，能打印训练 CLI 参数。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
  - 结果：通过，`Ran 29 tests`, `OK`。

## 本会话改动

- `evals/scripts/export_neural_codec.py` 新增 `--save-representations`，一次 model forward 导出 `codes`、`latent`、`quantized` 和 reconstruction。
- 保留旧 `--save-codes` 行为；单独使用 `--save-codes` 仍写 `codes/<id>.pt`。
- `--save-representations` 会写 `representations/<id>.codes.pt`、`<id>.latent.pt`、`<id>.quantized.pt`，manifest 中同步记录 `codes_path`、`latent_path`、`quantized_path`。
- `manifest.jsonl` 新增 Phase 1 schema 字段：`schema_version`、`frame_rate`、`hop_length`、`latent_dim`、`nominal_bitrate_kbps`、`rvq_payload_bits`、`context_scope`、`context_window_seconds`、`context_window_frames`、`clip_scope`、`is_full_utterance`。
- `run.json` 新增 schema、frame/RVQ metadata、context metadata 和 `representation_kinds`。
- `tests/test_evals_scripts.py` 新增 focused coverage：context window 解析、manifest row schema、临时 checkpoint/audio 的 representation export integration sanity。
- `evals/README.md` 新增 Phase 1 representation export 命令，并明确短样本 sanity 不能写成长程结论。
- active spec、feature list、progress 和 `init.sh` 已更新为 Phase 1 export 已实现，下一步进入真实 checkpoint export sanity / Phase 2 diagnostics。

## 本会话决策

- Phase 1 只实现导出和 schema，不训练新模型。
- Export 使用 model forward 获取 latent/quantized/codes，避免额外 encode/decode 路径遗漏 quantized representation。
- `context_scope` 默认 `none`，真实 long/full export 可通过 CLI 显式标注 `local | medium | long | full_utterance`。
- `rvq_payload_bits` 记录 exact RVQ bits；旧 `payload_bits` 继续记录 byte-rounded payload bits，避免破坏既有 benchmark 语义。

## 仍损坏或未验证

- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 export；当前 integration sanity 使用临时 checkpoint/audio。
- Mamba/SSM 依赖尚未固定。
- Ultra Low-Bitrate Speech Coding、LMCodec、Single-Codec、TFC、CodecSlime 的官方代码或可复现实验设置仍需后续确认。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- Harness：`./scripts/harness-check.sh` 通过，0 warnings。
- 静态检查：`git diff --check` 通过。
- Focused export help：`evals/scripts/export_neural_codec.py --help` 通过。
- Focused tests：`tests.test_evals_scripts` 10 tests OK。
- CLI sanity：训练脚本 help 在 `audiocodec` conda 环境通过。
- 单测：29 tests OK。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

下一步用已有 4kbps checkpoint 和可访问的 long/full utterance manifest 跑一次真实 export sanity，确认 `latent / quantized / codes` shape 和 metadata；然后进入 Phase 2 frozen representation redundancy diagnostics。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
