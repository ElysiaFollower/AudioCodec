from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[4]
SUMMARY = ROOT / "evals/outputs/scored-test/summary.csv"
OUT = Path(__file__).resolve().parent / "benchmark_full_table.svg"


ORDER = [
    "neural-12k",
    "neural-8k",
    "neural-4k",
    "neural-2k",
    "opus-default",
    "opus-16k",
    "opus-12k",
    "opus-8k",
    "opus-4k",
    "opus-2k",
    "mp3-default",
    "mp3-16k",
    "mp3-12k",
    "mp3-8k",
    "mp3-4k",
    "mp3-2k",
    "aac-default",
    "aac-16k",
    "aac-12k",
    "aac-8k",
    "aac-4k",
    "aac-2k",
    "flac",
]


def fmt(num: str, digits: int = 2) -> str:
    return f"{float(num):.{digits}f}"


def main() -> None:
    rows = list(csv.DictReader(SUMMARY.open()))
    by_label = {row["codec_label"]: row for row in rows}
    ordered = [by_label[label] for label in ORDER if label in by_label]

    width = 1260
    row_h = 34
    header_h = 44
    top_pad = 52
    bottom_pad = 22
    height = top_pad + header_h + row_h * len(ordered) + bottom_pad

    col_x = [34, 260, 420, 590, 760, 930]
    headers = [
        ("Codec", col_x[0]),
        ("Actual kbps", col_x[1]),
        ("Ratio x PCM16", col_x[2]),
        ("STOI", col_x[3]),
        ("LSD", col_x[4]),
        ("MS-STFT", col_x[5]),
    ]

    parts: list[str] = []
    parts.append(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f5f9ff"/>
    </linearGradient>
    <linearGradient id="head" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#eef5ff"/>
      <stop offset="100%" stop-color="#f7fbff"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="18" flood-color="#2758a4" flood-opacity="0.10"/>
    </filter>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#f6f9fe"/>
  <rect x="14" y="14" rx="24" ry="24" width="{width-28}" height="{height-28}" fill="url(#bg)" stroke="rgba(55,102,185,0.10)" filter="url(#shadow)"/>
  <text x="34" y="34" font-family="Segoe UI, PingFang SC, sans-serif" font-size="15" font-weight="700" fill="#2f6feb">Benchmark Summary</text>
  <rect x="24" y="{top_pad}" width="{width-48}" height="{header_h}" rx="14" fill="url(#head)" stroke="#d8e6fb"/>
"""
    )

    for title, x in headers:
        parts.append(
            f'<text x="{x}" y="{top_pad + 28}" font-family="Segoe UI, PingFang SC, sans-serif" font-size="12" font-weight="700" letter-spacing="0.08em" fill="#516786">{escape(title.upper())}</text>'
        )

    y0 = top_pad + header_h
    for idx, row in enumerate(ordered):
        y = y0 + idx * row_h
        fill = "#ffffff" if idx % 2 == 0 else "#f8fbff"
        stroke = "#e6eefb"
        label = row["codec_label"]
        is_neural = label.startswith("neural")
        if is_neural:
            fill = "#eef5ff"
            stroke = "#cfe0ff"

        parts.append(f'<rect x="24" y="{y}" width="{width-48}" height="{row_h}" fill="{fill}" stroke="{stroke}"/>')
        parts.append(
            f'<text x="{col_x[0]}" y="{y + 22}" font-family="Segoe UI, PingFang SC, sans-serif" font-size="14" font-weight="{"700" if is_neural else "500"}" fill="{"#2356c8" if is_neural else "#12233d"}">{escape(label)}</text>'
        )
        values = [
            fmt(row["actual_bitrate_kbps"]),
            fmt(row["compression_ratio_vs_pcm16"]),
            fmt(row["stoi"], 3),
            fmt(row["log_spectral_distance"]),
            fmt(row["multi_scale_stft"], 3),
        ]
        for value, x in zip(values, col_x[1:]):
            parts.append(
                f'<text x="{x}" y="{y + 22}" font-family="Segoe UI, PingFang SC, sans-serif" font-size="14" fill="#223655">{escape(value)}</text>'
            )

    parts.append("</svg>")
    OUT.write_text("\n".join(parts))


if __name__ == "__main__":
    main()
