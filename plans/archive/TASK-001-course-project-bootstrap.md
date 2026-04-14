Owner: ely
Status: archived
Last reviewed: 2026-04-14

# TASK-001: 课程项目基线企划与执行计划

## Archive Note

课程 baseline 方案、文档骨架和执行路径已经完成并沉淀到仓库，不再作为当前活动计划维护。

## 目标

在 1 到 2 天内完成一个 `speech neural codec` 的最小闭环，包含：

- 训练一个 `Encoder-Decoder + RVQ` 模型
- 导出重建音频和离散 token
- 与 `MP3` 做对比实验
- 形成可用于 Presentation 的表格、图和结论

## 背景

当前任务是课程项目，不是正式科研。

因此优先级应为：

1. 跑通完整链路
2. 结果可解释
3. 对比实验完整
4. 在此基础上再谈小创新

## 约束

- 时间窗口短，只有 1 到 2 天
- 音频方向实践经验有限
- 不宜引入高复杂度模块或难调的训练机制
- 最终需要和传统压缩方法形成清晰对比

## 验收标准

满足以下条件即可关闭本任务：

1. 模型能在 speech 数据子集上输出可听的重建结果。
2. 能统计 neural codec 的等效码率或压缩率。
3. 能完成一组与 `MP3` 的对比实验。
4. 能产出至少一页有说服力的实验结果页。

## Out of Scope

- 追求通用音频能力
- 追求 SOTA 指标
- 直接做 Mamba 主干实验
- 做复杂主观听评体系

## 推荐方案

### 模型

- `1D Conv Encoder`
- `RVQ bottleneck`
- `1D Conv Decoder`
- `L1 + multi-scale STFT + commitment loss`
- 默认冻结配置：`stride=[5,4,4,2]`、`latent_dim=128`、`codebook_size=256`、`num_quantizers=8`

### 数据

- 仅使用 `16 kHz` 单声道 speech
- 默认数据集：`LibriSpeech dev-clean`
- 默认切分：`train 60 min / val 10 min / test 10 min`
- 若时间特别紧：先缩为 `train 30 min / val 10 min / test 10 min`

### Baseline

- `ffmpeg` 编码得到的 `MP3`
- 建议默认测试 `8 kbps`、`16 kbps` 两档

### 指标

- 压缩率 / 文件大小 / 等效码率
- `SI-SNR`
- 可选：`PESQ`、`STOI`
- 频谱图与音频样例

## 默认训练预算

### Smoke test

- `10` 条训练音频
- `500 steps`
- 目标：确认模型能过拟合一个极小集合，排除数据流、维度和量化模块错误

### Main run

- 音频裁剪长度：`2.0 s`
- 优先 batch size：`16`
- 若显存不足：batch size 改 `8`，或裁剪长度改 `1.0 s`
- 默认训练步数：`5000 steps`
- 若验证损失还在下降：继续到 `10000 steps`

### 默认优化器

- `AdamW`
- learning rate `2e-4`
- weight decay `1e-4`
- mixed precision 开启

### 结果检查频率

- 每 `500` steps 保存 checkpoint
- 每 `500` 或 `1000` steps 导出一组重建音频
- 不只看 loss，必须人工听一次样例

## 1 到 2 天执行节奏

### Day 1 上午

1. 配环境：`PyTorch`、`torchaudio`、`ffmpeg`
2. 选数据：整理少量 speech 音频并统一到 `16 kHz mono`
3. 写最小数据管线和模型骨架
4. 先做 tiny overfit，确认前向、反向、重建都正常

### Day 1 下午

1. 接入 `RVQ`
2. 加 `multi-scale STFT loss`
3. 在 `LibriSpeech dev-clean` 的默认训练子集上开始跑第一版模型
4. 导出中间重建音频，人工听一轮

### Day 1 晚上

1. 固定一版可用配置继续训练
2. 同时准备 `MP3` baseline 脚本
3. 开始记实验日志，不要等到最后再补

### Day 2 上午

1. 导出 neural codec 结果
2. 跑 `MP3` 对比
3. 计算指标，画频谱图
4. 若时间允许，增加一个小消融

### Day 2 下午

1. 整理表格与图片
2. 撰写 Presentation 叙事
3. 总结传统 codec 与 neural codec 的差异

## 核心风险与应对

### 风险 1：模型不收敛或重建很差

应对：

- 先降低压缩强度，不要一开始就追极低码率
- 先验证能否过拟合极小数据集
- 优先使用 `multi-scale STFT loss`

### 风险 2：RVQ 实现出 bug

应对：

- 优先使用成熟实现或最小可用实现
- 先单独检查 encoder 输出维度和 codebook 输入维度

### 风险 3：训练太慢

应对：

- 缩小数据集和模型宽度
- 缩短音频片段长度
- 将 main run 从 `10000 steps` 降回 `5000 steps`
- 若本地无可用 GPU，优先切到云端或 Colab

### 风险 4：指标库安装不顺

应对：

- 不把 `PESQ`、`STOI` 作为硬依赖
- 保底保留 `SI-SNR + 频谱图 + 听感样例`

## Presentation 建议页结构

1. 选题背景：传统压缩 vs learned compression
2. 方法：Encoder-Decoder + RVQ
3. 实验设置：数据、码率、指标
4. 结果：与 `MP3` 的对比表格和频谱图
5. 分析：优势、局限、失败案例
6. Future work：Mamba 或更强时序模型

## 现在最值得做的一件事

如果马上进入实现，第一优先级不是找“最先进模型”，而是：

`先把一个最小 neural codec 跑通，并确保能稳定输出可听样例`

这是后续一切实验、对比和讲述的前提。
