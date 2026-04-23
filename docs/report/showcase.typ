#import "theme.typ": report_theme, theorem, definition, proposition, remark, figure_box

#show: report_theme.with(
  title: [面向正式技术报告的 Typst 主题样板],
  author: [Ely],
  date: [2026-04-20],
  abstract_text: [
    本文档用于展示一套面向单栏技术报告的 Typst 主题。它优先服务于 A4 打印输出，同时尽量保留屏幕阅读的舒适性。示例覆盖中文正文、English terms、数学公式、列表、代码、表格、图、脚注、引用与定理环境，以便在实现阶段进行直接视觉校正。#footnote[脚注也在这里一并展示。]
  ],
)

= 设计目标

这套主题的核心目标不是模拟网页，而是提供一种更接近技术报告的阅读结构。它强调节制、稳定与连续阅读能力，尽量避免暖色块、圆角卡片和过度强调所带来的界面感。对于中文技术写作来说，报告中的 English terminology、file path、inline identifiers 与数学符号都应当自然地嵌入段落，而不应破坏正文节奏。

在阅读体验上，这份主题有两个明确偏好。第一，它宁可让版面稍微紧一点，也不愿意把段落和标题拉得松散而失去信息密度。第二，它宁可让色彩退到背景，也不愿意用过于明显的 accent 去提示结构。对正式报告来说，秩序比风格更重要。

== 标题与正文节奏

技术报告中的一级标题应当有清楚的结构作用，但不应像幻灯片那样夸张；二级和三级标题则应更偏功能性，主要通过字重、字号和上下留白来区分层级。正文的行长需要足够克制，否则会让长句和中英混排都变得疲劳。

下面这段话专门用于观察中英混排效果。对于大多数 machine learning systems 报告而言，throughput、latency、token budget、failure mode 与 offline evaluation 这些术语几乎不可避免。如果拉丁字母与中文正文的重量和节奏不一致，阅读时就会出现明显的“异物感”。一个合格的主题至少应保证这些术语在段落里是顺滑的，而不是突出到像贴上去的标签。

=== 列表

- 主题应优先保证正文可读性。
- 标题层级应克制，不依赖颜色制造结构。
- 代码块、引用块、图表与定理环境应形成统一的 block language。
- 默认输出需要适合黑白打印。

+ 有序结构也需要保持紧凑。
+ 第二项用于观察 marker 与段落的相对位置。
+ 第三项用于观察跨行后的缩进是否稳定。

== 定理、定义与说明环境

#definition(number: [1], title: [状态一致性])[
若一个排版系统在标题、正文、图表、代码与引用之间使用同一套尺度与间距逻辑，则称该系统具有*状态一致性*。这种一致性并不会自动创造风格，但会显著降低阅读摩擦。
]

#theorem(kind: "Theorem", number: [2], title: [阅读稳定性])[
设一个单栏报告主题满足以下条件：正文行长受控、段间节奏稳定、标题层级克制、代码与表格不喧宾夺主。则在相同的信息密度下，该主题通常比默认办公文档样式更适合长时间阅读。
]

#proposition(number: [3], title: [打印兼容性])[
若版面结构不依赖颜色，并且块级元素主要依靠边线、留白和编号组织，则该主题在黑白打印环境中通常仍能保持较好的结构辨识度。
]

#remark(number: [4], title: [实现策略])[
对于 Typst 实现，真正值得仔细调节的是字号、leading、caption gap 与 block spacing，而不是去追逐网页式的微交互细节。
]

== 引用与脚注

#quote[
技术报告的视觉语言应当尽可能中立。真正让人持续阅读下去的，不是某个局部设计看起来多有趣，而是整份文档在二十页以后依然没有显著增加理解负担。
]

上面的引用块有意保持朴素。它不应像 callout 或 note box，也不应拥有太强的背景面板感。正式报告中的引用更接近文献性材料，而不是作者正在与读者进行 UI 式互动。

== 公式与数学

对于含有形式化定义的报告，数学环境的气质应尽量接近传统技术文档，而不是使用“被设计过头”的视觉样式。下面给出一个简短例子：

$
  L(theta) = sum_(i=1)^n ell(f_theta(x_i), y_i) + lambda norm(theta)_2^2
$

当一个模型需要在有限资源下优化吞吐量与正确率时，我们也可以写出一个非常常见的约束问题：

$
  max_(pi in Pi) E_(x ~ D)[u(pi, x)] quad
  s.t. quad c_("latency")(pi) <= tau and c_("memory")(pi) <= kappa
$

== 代码

报告中的代码块更像证据或说明材料，而不是编辑器截图。因此，它们应避免粗重阴影、鲜艳背景与过分华丽的语法高亮。

```typ
#let choose-plan(mode, budget) = {
  let limits = (
    safe: (120ms, 2048),
    fast: (80ms, 1536),
  )
  let target = limits.at(mode)
  budget.latency <= target.at(0)
    and budget.memory <= target.at(1)
}
```

行内代码例如 `accept(plan, budget)`、`report/theme.typ` 与 `latency_ms` 需要能嵌入中文句子，而不显得像一个色块按钮。

== 表格与图

下表用于观察正式报告风格下的表格边界、数字对齐感与 caption 节奏。

#figure(
  kind: table,
  supplement: [Table],
  caption: [不同版式设置下的阅读与打印表现比较。],
  table(
    columns: (2fr, 1fr, 1fr, 2fr),
    inset: (x: 6pt, y: 5pt),
    stroke: (x: none, y: 0.35pt + rgb("#b8b8b8")),
    align: (left, center, center, left),
    table.header(
      [方案],
      [屏幕阅读],
      [打印友好],
      [说明],
    ),
    [网页风长文主题], [高], [中], [舒适但常带有明显界面感],
    [默认办公文档], [低], [中], [结构通常足够清楚但节奏较粗糙],
    [本主题方向], [高], [高], [优先维持正式报告所需的秩序与连续性],
  ),
)

图的样式同样应尽量平静。它的任务是帮助说明，而不是替代内容本身。

#figure_box(
  caption: [占位图：示意性的系统性能对比图，用于观察图注、边界与留白。],
  block(
    width: 105mm,
    height: 48mm,
    stroke: 0.45pt + rgb("#888888"),
    fill: rgb("#fafafa"),
    inset: 12pt,
    radius: 0pt,
  )[
    #align(center)[
      #set text(font: ("PingFang SC", "Helvetica"), size: 9.4pt, fill: rgb("#666666"))
      示例图形区域
    ]
  ],
)

== 结论

一套好的技术报告主题不需要向读者证明自己“设计得很努力”。它只需要稳定地处理以下问题：页边距是否得体，正文是否耐读，标题是否清楚，表格是否正式，代码是否克制，图注是否统一，以及混合语言的段落是否自然。只要这些问题被认真解决，整体气质通常就会自然靠近专业技术文档，而不是停留在好看的 markdown 皮肤层面。
