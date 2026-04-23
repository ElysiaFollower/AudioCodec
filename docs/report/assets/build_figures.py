from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = ROOT / "evals" / "outputs" / "scored-test" / "summary.csv"
ASSET_DIR = Path(__file__).resolve().parent


COLORS = {
    "neural_codec": "#1f1f1f",
    "opus": "#1f77b4",
    "mp3": "#ff7f0e",
    "aac": "#2ca02c",
    "flac": "#7f7f7f",
}

LABELS = {
    "neural_codec": "Neural codec",
    "opus": "Opus",
    "mp3": "MP3",
    "aac": "AAC",
    "flac": "FLAC",
}


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_PATH.open() as handle:
        return list(csv.DictReader(handle))


def split_modes(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    sweeps: dict[str, list[dict[str, str]]] = {}
    defaults: list[dict[str, str]] = []
    for row in rows:
        codec = row["codec_name"]
        if codec == "flac" or row["codec_label"].endswith("-default"):
            defaults.append(row)
            continue
        sweeps.setdefault(codec, []).append(row)
    for codec_rows in sweeps.values():
        codec_rows.sort(key=lambda row: float(row["actual_bitrate_kbps"]))
    return sweeps, defaults


def render_plot(metric_key: str, ylabel: str, output_name: str) -> None:
    rows = load_rows()
    sweeps, defaults = split_modes(rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for codec, codec_rows in sweeps.items():
        xs = [float(row["actual_bitrate_kbps"]) for row in codec_rows]
        ys = [float(row[metric_key]) for row in codec_rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=5.5,
            color=COLORS[codec],
            label=LABELS[codec],
        )

    for row in defaults:
        codec = row["codec_name"]
        x = float(row["actual_bitrate_kbps"])
        y = float(row[metric_key])
        label = f"{LABELS[codec]} default" if codec != "flac" else LABELS[codec]
        ax.scatter(
            [x],
            [y],
            marker="x" if codec != "flac" else "s",
            s=52,
            linewidths=1.6,
            color=COLORS[codec],
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Actual bitrate (kbps)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1.6, 180)
    ax.legend(frameon=True, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / output_name, format="svg")
    plt.close(fig)


def main() -> None:
    render_plot("stoi", "STOI ↑", "rd_stoi.svg")
    render_plot("log_spectral_distance", "Log spectral distance ↓", "rd_lsd.svg")
    render_plot("multi_scale_stft", "Multi-scale STFT ↓", "rd_msstft.svg")


if __name__ == "__main__":
    main()
