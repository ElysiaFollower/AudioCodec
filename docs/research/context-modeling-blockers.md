Owner: ely
Status: active
Last reviewed: 2026-05-05

# 研究收集阻塞与待补

## Blocked

| 项 | 阻塞类型 | 证据 | 下一步 |
| --- | --- | --- | --- |
| LMCodec official implementation | no official code found in first pass | 已确认 Google Research paper/project page，但首轮没有定位到官方代码库 | 搜索作者主页、Google Research GitHub 和非官方复现；若无代码，后续只作为 paper-level prior 设计参考 |
| Convolutional Transformer official implementation | no official code found in first pass | 已确认 Google Research paper page，但首轮没有定位到官方代码库 | 后续只提取 architecture idea；实现时用本仓库自己的 matched TCN/Transformer 模块 |
| neural speech codec entropy coding survey completeness | incomplete coverage | 本轮只覆盖 EnCodec / LMCodec 两个强相关项 | 下一轮继续搜集 arithmetic coding、autoregressive prior、recurrent prior 和 codebook entropy 估计论文 |
| Mamba-in-codec direct evidence | insufficient direct evidence | 第一轮只确认 Mamba/Mamba-2 是 model family reference | 后续检索 Mamba audio codec / speech tokenizer 专门论文；没有强证据前不得设为主路线 |

## Skipped intentionally

| 项 | 原因 |
| --- | --- |
| local PDF downloads | Commit 2 目标是建立 intake 主表，不建立本地文献库 |
| external repo clones | Commit 2 不评估依赖、license、训练配置或可运行性 |
| benchmark runs | Commit 2 不实现代码，也不跑外部模型或训练 |

## Commit 3 必须补齐的问题

- 结果表字段：明确 `nominal_bitrate`、`estimated_entropy_bitrate`、`quality_metrics`、`tokens_per_second`、`latency`、`memory` 的来源。
- 最小实验矩阵：先选哪些插入层级，哪些只保留为后续。
- 数据路径：明确本地 macOS sanity、Linux A100 smoke 和正式 speech benchmark 使用的数据。
- export 需求：是否先实现 RVQ code export，用于 code-prior entropy 诊断。
