项目管理指南：

一句话总结需求和目标： “利用VAE+RVQ的主干技术(可能还可以引入其他简单有效的技术)来实现音频的编解码器， 仅针对speech领域， 并最终在benchmark上和mp3压缩比较压缩率和保真率”

文件基本结构：
```
README.md
AGENTS.md
docs/
  architecture/   # 系统结构、边界、核心模块
  adr/            # 架构决策记录
  how-to/         # 操作手册、发布、排障、迁移
  reference/      # API、配置、目录说明、脚本说明
  explanation/    # 设计动机、取舍、历史背景
  runbooks/       # 线上/运维/值班处理流程
  overview.md     # 项目基本介绍和核心目标
plans/
  active/         # 正在做的任务 spec / plan
  archive/        # 已完成但暂时保留的执行计划
.github/
  ISSUE_TEMPLATE/
  pull_request_template.md
  CODEOWNERS
  workflows/
evals/            # 如果你在做 AI 功能，放评测样本与脚本
```

**核心原则：文档按“寿命”分层，不按“事件”堆积。**

具体机制建议：
```
每个任务先开一个 Issue，模板里强制填：目标、背景、约束、验收标准、Out of scope。GitHub issue forms 本来就是为“收集结构化信息”设计的。
只有当任务超过半天、跨多个文件、涉及设计取舍时，才创建 plans/active/TASK-123-xxx.md。
PR 模板里强制回答三件事：
1）改了什么；
2）怎么验证；
3）是否需要更新 docs / ADR / changelog。GitHub 的 issue/PR templates 就是干这个的。
任务关闭时，明确只保留这些长期产物之一：
- 代码
- 测试
- ADR
- 用户/运维文档
- release note / changelog
- 什么都不保留. 任务计划本身如果没有长期价值，就进 archive/ 或直接删。
每个永久文档加最少元数据：Owner、Status、Last reviewed
```