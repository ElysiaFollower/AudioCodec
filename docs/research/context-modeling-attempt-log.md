Owner: ely
Status: active
Last reviewed: 2026-05-05

# 研究收集日志

## 2026-05-05

本轮只做 web-level primary source verification 和结构化摘录，不下载论文 PDF、不 clone 外部仓库。

| 候选 | paper / project | code | 本地 acquisition 状态 | 备注 |
| --- | --- | --- | --- | --- |
| SoundStream | success: Google Research page | skipped_intentionally | skipped_intentionally | 首轮只需要确定 codec/RVQ baseline 地位；未下载 PDF |
| EnCodec | success: arXiv | success: official Meta GitHub | skipped_intentionally | code-prior / entropy 方向核心材料 |
| DAC | success: arXiv | success: official Descript GitHub | skipped_intentionally | 可作为 RVQGAN 工程参考 |
| AudioDec | success: arXiv | success: official Meta GitHub | skipped_intentionally | 可参考 streaming / latency reporting |
| LMCodec | success: Google Research page | blocked: no official code found in first pass | skipped_intentionally | 需要后续确认是否有非官方复现可参考 |
| Convolutional Transformer | success: Google Research page | blocked: no official code found in first pass | skipped_intentionally | early/latent context 证据，代码复用不确定 |
| BigCodec | success: arXiv | success: GitHub located | skipped_intentionally | 需要后续确认训练配置、数据和 license |
| SpeechTokenizer | success: arXiv | success: GitHub located | skipped_intentionally | speech tokenizer / token efficiency 参考 |
| Mimi / Moshi | success: arXiv | success: official Kyutai GitHub | skipped_intentionally | tokenizer / streaming / speech LM 参考 |
| Mamba / Mamba-2 | success: arXiv | success: official state-spaces GitHub | skipped_intentionally | model family reference, not codec evidence |

## 查询范围

- neural audio codec / RVQ baseline: SoundStream, EnCodec, DAC, AudioDec.
- speech codec / low bitrate / context: LMCodec, Convolutional Transformer, BigCodec.
- codec token / speech LM: SpeechTokenizer, Mimi/Moshi.
- sequence model family: Mamba, Mamba-2.

## 本轮有意跳过

- 不下载 PDF：当前 commit 目标是建立研究收集主表，不做本地文献库。
- 不 clone 外部 repo：当前 commit 不评估可运行性、license 或依赖冲突。
- 不跑外部模型 demo：当前 commit 不比较音质或速度。

## 2026-05-05 long-range focused refresh

主题重定后，补充检索 long-range redundancy、low-bitrate Transformer codec、codec-token language modeling、time-invariant / multi-scale / low-frame-rate tokenizer。

| 候选 | paper / project | code | 本地 acquisition 状态 | 备注 |
| --- | --- | --- | --- | --- |
| Ultra Low-Bitrate Speech Coding with Pretrained Transformers | success: Google Research page | blocked: no official code found in this pass | skipped_intentionally | 直接指出 conv/recurrent effective receptive field 限制 compression efficiency |
| AudioLM | success: Google project page and arXiv | skipped_intentionally | skipped_intentionally | 证明 token sequence 能承载 long-term structure，但目标是 generation |
| SoundStorm | success: arXiv | skipped_intentionally | skipped_intentionally | 30 秒 codec-token generation，支持 long token modeling feasibility |
| Stable Codec / TAAE | success: project page and paper | success: official Stability AI GitHub | skipped_intentionally | Transformer codec scaling；记录 FlashAttention / no CPU inference 工程限制 |
| TiCodec | success: arXiv | located via paper page, not cloned | skipped_intentionally | time-invariant utterance code 是长程不变信息的直接相关方向 |
| Single-Codec | success: arXiv | not searched in this pass | skipped_intentionally | time-invariant embedding + phonetic sequence，和长程不变信息相关 |
| SNAC | success: arXiv | success: official GitHub located | skipped_intentionally | multi-scale temporal resolution RVQ，属于 tokenization redesign |
| WavTokenizer | success: arXiv | success: official GitHub located | skipped_intentionally | low token rate + extended contextual windows，面向 audio LM |
| LongCat-Audio-Codec | success: project page and arXiv | success: official GitHub located | skipped_intentionally | 低帧率 semantic/acoustic tokenizer；记录 <30s input limitation |
| Temporally Flexible Coding | success: arXiv | not searched in this pass | skipped_intentionally | variable frame rate，直接针对 fixed-frame speech codec 的 time-varying information density |
| CodecSlime | success: arXiv | not searched in this pass | skipped_intentionally | dynamic frame rate plugin，直接处理 fixed-frame-rate codec 的 temporal redundancy |

本轮仍不下载 PDF、不 clone repo、不跑 demo；目标是更新研究方向和 go/no-go 依据。
