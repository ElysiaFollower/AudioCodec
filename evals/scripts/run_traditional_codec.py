#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import bytes_to_kbps, read_jsonl, run_ffmpeg, write_jsonl


LOSSY_CODEC_OPTIONS = {
    "mp3": {
        "extension": ".mp3",
        "encoder_args": lambda kbps: ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"],
    },
    "opus": {
        "extension": ".opus",
        "encoder_args": lambda kbps: ["-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip"],
    },
    "aac": {
        "extension": ".m4a",
        "encoder_args": lambda kbps: ["-c:a", "aac", "-b:a", f"{kbps}k"],
    },
}

LOSSLESS_CODEC_OPTIONS = {
    "flac": {
        "extension": ".flac",
        "encoder_args": lambda _kbps: ["-c:a", "flac"],
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a traditional codec over a benchmark manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--codec", choices=("mp3", "opus", "aac", "flac"), required=True)
    parser.add_argument("--bitrate-kbps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec-label", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    output_dir = args.output_dir.resolve()
    compressed_dir = output_dir / "compressed"
    recon_dir = output_dir / "reconstructions"
    manifest_path = output_dir / "manifest.jsonl"
    run_path = output_dir / "run.json"
    compressed_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    if args.codec in LOSSY_CODEC_OPTIONS and args.bitrate_kbps is None:
        raise ValueError(f"--bitrate-kbps is required for codec {args.codec}.")
    if args.codec in LOSSLESS_CODEC_OPTIONS and args.bitrate_kbps is not None:
        raise ValueError(f"--bitrate-kbps must be omitted for lossless codec {args.codec}.")

    specs = LOSSY_CODEC_OPTIONS.get(args.codec) or LOSSLESS_CODEC_OPTIONS[args.codec]
    exported_rows: list[dict] = []

    for row in rows:
        source_path = Path(row["source_path"]).expanduser().resolve()
        compressed_path = compressed_dir / f"{row['id']}{specs['extension']}"
        recon_path = recon_dir / f"{row['id']}.wav"

        run_ffmpeg(
            [
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-map_metadata",
                "-1",
                "-ac",
                "1",
                "-ar",
                str(row["sample_rate"]),
                *specs["encoder_args"](args.bitrate_kbps),
                str(compressed_path),
            ]
        )
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(compressed_path),
                "-vn",
                "-map_metadata",
                "-1",
                "-ac",
                "1",
                "-ar",
                str(row["sample_rate"]),
                str(recon_path),
            ]
        )

        compressed_bytes = compressed_path.stat().st_size
        exported_rows.append(
            {
                "id": row["id"],
                "codec_family": "traditional",
                "codec_name": args.codec,
                "codec_label": args.codec_label,
                "source_path": str(source_path),
                "compressed_path": str(compressed_path),
                "reconstruction_path": str(recon_path),
                "duration_seconds": row["duration_seconds"],
                "num_samples": row["num_samples"],
                "sample_rate": row["sample_rate"],
                "channels": row["channels"],
                "pcm16_bytes": row["pcm16_bytes"],
                "compressed_bytes": compressed_bytes,
                "actual_bitrate_kbps": bytes_to_kbps(compressed_bytes, row["duration_seconds"]),
                "target_bitrate_kbps": args.bitrate_kbps,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_path, exported_rows)
    run_path.write_text(
        json.dumps(
            {
                "codec_family": "traditional",
                "codec_name": args.codec,
                "codec_label": args.codec_label,
                "target_bitrate_kbps": args.bitrate_kbps,
                "items": len(exported_rows),
            },
            indent=2,
        )
    )
    print(f"Wrote traditional codec run to {output_dir}")


if __name__ == "__main__":
    main()
