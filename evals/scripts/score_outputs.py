#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import load_audio, read_jsonl, write_jsonl
from _metrics import compute_log_spectral_distance, compute_multi_scale_stft, compute_si_sdr_db, compute_stoi_or_none


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one or more benchmark output manifests.")
    parser.add_argument("--run-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["codec_label"], []).append(row)

    numeric_fields = [
        "compressed_bytes",
        "actual_bitrate_kbps",
        "compression_ratio_vs_pcm16",
        "si_sdr_db",
        "log_spectral_distance",
        "multi_scale_stft",
        "stoi",
    ]
    summary: list[dict] = []
    for codec_label, items in sorted(grouped.items()):
        row = {
            "codec_label": codec_label,
            "codec_family": items[0]["codec_family"],
            "codec_name": items[0]["codec_name"],
            "count": len(items),
        }
        for field in numeric_fields:
            values = [item[field] for item in items if item.get(field) is not None]
            row[field] = statistics.fmean(values) if values else None
        summary.append(row)
    return summary


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "codec_label",
        "codec_family",
        "codec_name",
        "count",
        "compressed_bytes",
        "actual_bitrate_kbps",
        "compression_ratio_vs_pcm16",
        "si_sdr_db",
        "log_spectral_distance",
        "multi_scale_stft",
        "stoi",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    per_file_path = output_dir / "per_file_metrics.jsonl"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    scored_rows: list[dict] = []
    for manifest_path in args.run_manifest:
        for row in read_jsonl(manifest_path):
            reference = load_audio(row["source_path"], sample_rate=row["sample_rate"], channels=row["channels"])
            degraded = load_audio(
                row["reconstruction_path"],
                sample_rate=row["sample_rate"],
                channels=row["channels"],
            )
            scored_rows.append(
                {
                    **row,
                    "compression_ratio_vs_pcm16": row["pcm16_bytes"] / row["compressed_bytes"],
                    "si_sdr_db": compute_si_sdr_db(reference, degraded),
                    "log_spectral_distance": compute_log_spectral_distance(reference, degraded),
                    "multi_scale_stft": compute_multi_scale_stft(reference, degraded),
                    "stoi": compute_stoi_or_none(reference, degraded, sample_rate=row["sample_rate"]),
                }
            )

    summary_rows = summarize_rows(scored_rows)
    write_jsonl(per_file_path, scored_rows)
    write_summary_csv(summary_csv_path, summary_rows)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"Wrote metrics to {output_dir}")


if __name__ == "__main__":
    main()
