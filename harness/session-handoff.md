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
- 当前待提交目标：补充长程时间冗余 focused research refresh，回答“现有研究是否考虑过长程/steady-state temporal redundancy、效果如何、为什么没有成为通用 codec 架构”。
- 本轮新增/修改范围：
  - added: `docs/research/long-range-redundancy-intake.md`
  - modified: `docs/research/context-modeling-intake.md`
  - modified: `docs/research/context-modeling-attempt-log.md`
  - modified: `docs/research/context-modeling-blockers.md`
  - modified: `README.md`
  - modified: `docs/overview.md`
  - modified: `plans/active/TASK-008-context-sequence-modeling-research.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh`
  - 结果：通过，`Harness 检查通过，共 0 个警告。`
- `git diff --check`
  - 结果：通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
  - 结果：通过，能打印训练 CLI 参数。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
  - 结果：通过，`Ran 25 tests`, `OK`。

## 本会话改动

- 按用户确认后的新主题补充 long-range focused research refresh。
- 新增 `docs/research/long-range-redundancy-intake.md`，覆盖：
  - Ultra Low-Bitrate Speech Coding with Pretrained Transformers；
  - LMCodec / EnCodec entropy model；
  - AudioLM / SoundStorm / Moshi / LongCat；
  - TiCodec / Single-Codec / SNAC / WavTokenizer / Stable Codec；
  - Temporally Flexible Coding / CodecSlime。
- 将核心结论写入文档：没有看到完全同题的 fixed SEANet/RVQ 表示层级归因工作；但已有研究分别从 long-context Transformer、code prior、time-invariant token、多尺度/低帧率 token、dynamic/variable frame rate 利用了长程或 steady-state temporal redundancy。
- 明确 TFC/CodecSlime 是最接近“时间冗余压缩”的相关方向，但它们的主路线是 frame-rate allocation / tokenization redesign，不替代当前 fixed baseline 上的 go/no-go diagnostics。
- 同步更新研究主表、attempt log、blockers、README、overview、TASK-008、feature list 和 progress。

## 本会话决策

- 当前第一阶段仍保持 fixed-frame SEANet/RVQ baseline；不因为 TFC/CodecSlime 直接跳到 dynamic frame-rate redesign。
- E0-E2 go/no-go diagnostics 仍是下一步：先导出长片段 / full utterance 的 latent、quantized embedding 和 RVQ codes，比较 local、medium、long、full context 的 predictability / entropy 曲线。
- 如果收益主要集中在静音、长元音或缓慢变化区间，下一阶段应开 dynamic/variable frame-rate 或 tokenization redesign 分支，而不是只扩大 context window。
- Mamba 仍只是候选上下文模型；本轮没有新增 Mamba 依赖或上下文建模代码。

## 仍损坏或未验证

- 本轮只做 web-level / primary-source-level 调研；未下载论文 PDF、未 clone 外部 repo、未运行外部 demo。
- Ultra Low-Bitrate Speech Coding、LMCodec、Single-Codec、TFC、CodecSlime 的官方代码或可复现实验设置仍需后续确认。
- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- Harness：`./scripts/harness-check.sh` 通过，0 warnings。
- 静态检查：`git diff --check` 通过。
- CLI sanity：训练脚本 help 在 `audiocodec` conda 环境通过。
- 单测：25 tests OK。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

等待用户审查 focused refresh 是否对齐。若确认，下一步进入 Commit 4：实现 representation export、long/full utterance manifest metadata 和 result schema，支撑 E0-E2 go/no-go diagnostics；不直接实现 Mamba codec 或 dynamic frame-rate redesign。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
