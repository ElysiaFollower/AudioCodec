#let page_paper = "a4"
#let page_margin = (
  top: 25mm,
  bottom: 29mm,
  inside: 25mm,
  outside: 25mm,
)

#let body_fonts = ("Libertinus Serif", "Songti SC", "STSong")
#let sans_fonts = ("PingFang SC", "Helvetica")
#let mono_fonts = ("Menlo", "Monaco")

#let text_color = rgb("#181818")
#let muted_color = rgb("#5c5c5c")
#let line_color = rgb("#7c7c7c")
#let light_line_color = rgb("#b8b8b8")
#let surface_color = rgb("#f6f6f4")
#let code_surface_color = rgb("#fbfbfa")

#let base_size = 10.8pt
#let caption_size = 9pt
#let footnote_size = 9pt
#let code_size = 9.5pt

#let title_text(body) = text(
  font: body_fonts,
  size: 21pt,
  weight: "semibold",
  fill: text_color,
  lang: "zh",
)[#body]

#let section_heading(level, body) = {
  let size = if level == 1 {
    17pt
  } else if level == 2 {
    14pt
  } else if level == 3 {
    12pt
  } else {
    base_size
  }

  let weight = if level <= 3 { "semibold" } else { "bold" }

  text(
    font: body_fonts,
    size: size,
    weight: weight,
    fill: text_color,
    lang: "zh",
  )[#body]
}

#let theorem(kind: "Theorem", number: auto, title: none, body) = {
  let label = if title == none {
    [#kind #number]
  } else {
    [#kind #number (#title)]
  }

  block(
    above: 1.15em,
    below: 1.05em,
    inset: (left: 11pt, right: 0pt, top: 1pt, bottom: 1pt),
    stroke: (left: 1.2pt + line_color),
    radius: 0pt,
    breakable: false,
  )[
    #text(
      font: sans_fonts,
      size: 9.1pt,
      weight: "semibold",
      fill: muted_color,
    )[#label]

    #v(0.45em)
    #body
  ]
}

#let definition(number: auto, title: none, body) = theorem(
  kind: "Definition",
  number: number,
  title: title,
)[#body]

#let proposition(number: auto, title: none, body) = theorem(
  kind: "Proposition",
  number: number,
  title: title,
)[#body]

#let remark(number: auto, title: none, body) = theorem(
  kind: "Remark",
  number: number,
  title: title,
)[#body]

#let report_quote(body) = block(
  above: 1.0em,
  below: 1.0em,
  inset: (left: 12pt, right: 0pt, top: 1pt, bottom: 1pt),
  stroke: (left: 1pt + line_color),
  radius: 0pt,
  breakable: true,
)[
  #set text(size: 10.2pt, fill: muted_color)
  #body
]

#let figure_box(body, caption: none, kind: "Figure") = figure(
  body,
  kind: kind,
  caption: caption,
  supplement: [#kind],
)

#let report_theme(
  title: none,
  author: none,
  date: none,
  abstract_text: none,
  body,
) = {
  set page(
    paper: page_paper,
    margin: page_margin,
    numbering: "1",
    number-align: center + bottom,
  )

  set text(
    font: body_fonts,
    size: base_size,
    lang: "zh",
    fill: text_color,
  )

  set par(
    justify: true,
    leading: 0.5em,
    spacing: 0.58em,
    first-line-indent: 0em,
  )

  set heading(numbering: "1.1.1")

  show heading: it => block(
    above: if it.level == 1 { 1.95em } else if it.level == 2 { 1.6em } else if it.level == 3 { 1.3em } else { 1.1em },
    below: if it.level == 1 { 0.72em } else if it.level == 2 { 0.6em } else { 0.48em },
    breakable: false,
  )[
    #section_heading(it.level, it)
  ]

  show emph: set text(style: "italic")
  show strong: set text(font: sans_fonts, weight: "bold")

  show link: it => underline(offset: 1.8pt, stroke: 0.45pt + muted_color)[
    #text(fill: text_color)[#it.body]
  ]

  show quote: it => report_quote(it.body)

  show raw.where(block: false): it => text(
    font: mono_fonts,
    size: 0.94em,
    fill: text_color,
  )[#it]

  show raw.where(block: true): it => block(
    above: 1.0em,
    below: 1.0em,
    width: 100%,
    inset: (x: 12pt, y: 9pt),
    stroke: (
      top: 0.55pt + line_color,
      bottom: 0.35pt + light_line_color,
    ),
    fill: code_surface_color,
    radius: 0pt,
    breakable: false,
  )[
    #if it.lang != "" and it.lang != "text" [
      #align(right)[
        #text(
          font: sans_fonts,
          size: 7.8pt,
          weight: "medium",
          tracking: 0.08em,
          fill: muted_color,
        )[#upper(it.lang)]
      ]
      #v(0.28em)
    ]
    #set text(font: mono_fonts, size: code_size, lang: "en")
    #it
  ]

  show figure: set block(above: 1.15em, below: 1.15em)
  show figure.caption: set text(font: body_fonts, size: caption_size, fill: muted_color)
  show table: set text(size: 10pt)
  show footnote.entry: set text(size: footnote_size, fill: muted_color)

  if title != none {
    align(center)[#title_text(title)]

    v(0.95em)

    align(center)[
      #text(font: sans_fonts, size: 9.5pt, fill: muted_color)[
        #if author != none { author }
        #if author != none and date != none { [ · ] }
        #if date != none { date }
      ]
    ]

    if abstract_text != none {
      v(1.65em)
      block(
        inset: (x: 12pt, y: 11pt),
        stroke: (
          top: 0.6pt + line_color,
          bottom: 0.6pt + light_line_color,
        ),
        radius: 0pt,
      )[
        #text(
          font: sans_fonts,
          size: 8.8pt,
          weight: "semibold",
          tracking: 0.08em,
          fill: muted_color,
        )[ABSTRACT]

        #v(0.4em)
        #set par(justify: true, leading: 0.46em, spacing: 0.42em)
        #set text(size: 9.7pt)
        #abstract_text
      ]
    }

    v(1.35em)
  }

  body
}
