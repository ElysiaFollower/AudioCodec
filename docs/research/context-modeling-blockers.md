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
| Ultra Low-Bitrate Speech Coding official implementation | no official code found in focused pass | 已确认 Google Research paper page，但未定位官方代码库 | 先作为 paper-level 证据；若后续实现 Transformer embedding baseline，需要自行复现或找到非官方实现 |
| Full utterance as deployable codec path | conceptual / deployment blocker | AudioLM/SoundStorm 支持 offline/生成式 long context；LongCat official repo 仍限制输入小于 30 秒并要求长音频切段 | Full utterance 只作为 offline upper bound；部署路径必须转成 causal/chunked long context |
| Long-context tokenizer redesign vs fixed baseline | scope blocker | TiCodec/SNAC/WavTokenizer/LongCat 都通过改变 tokenization/frame-rate/semantic-acoustic split 利用长程信息 | 第一阶段保持 SEANet/RVQ baseline；只有 go/no-go diagnostics 显示强 long-range收益后再开 tokenization redesign 任务 |
| Dynamic / variable frame-rate route vs context insertion route | scope blocker | Temporally Flexible Coding 和 CodecSlime 直接处理 fixed-frame-rate neural speech codec 的 temporal redundancy，但路线是 variable/dynamic frame allocation | 第一阶段仍先做 fixed-frame baseline 上的 context/predictability diagnostics；若冗余集中在 steady-state segments，再开 frame-rate redesign 任务 |

## Skipped intentionally

| 项 | 原因 |
| --- | --- |
| local PDF downloads | Commit 2 目标是建立 intake 主表，不建立本地文献库 |
| external repo clones | Commit 2 不评估依赖、license、训练配置或可运行性 |
| benchmark runs | Commit 2 不实现代码，也不跑外部模型或训练 |
| focused-pass PDF/repo acquisition | 本轮只补 long-range 方向判断，不建立外部资产库 |

## 后续必须补齐的问题

- code-prior baseline 的 token ordering：stage-major、time-major、coarse-to-fine conditioning 哪个先做。
- 长片段 / full utterance manifest 构建规则。
- go/no-go 阈值：long/full context 相比 local 至少带来多少 bits-per-code 或 predictability improvement 才进入 codec context 训练。
- steady-state segment redundancy 诊断：若收益主要来自静音、长元音或缓慢变化区间，应考虑 CodecSlime / TFC 类 dynamic frame-rate 分支，而不是只扩大 context window。

## Long-range focused pass 后的新增待补

- Ultra Low-Bitrate Speech Coding、LMCodec、Single-Codec、TFC、CodecSlime 是否有官方或可靠复现代码。
- TFC / CodecSlime 与当前 fixed-frame baseline 的公平比较口径：什么时候只是相关工作，什么时候应升级为后续实验分支。
