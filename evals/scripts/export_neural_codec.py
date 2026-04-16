#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import bytes_to_kbps, load_audio, read_jsonl, resolve_device, rvq_payload_bytes, save_audio, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audiocodec.config import CodecExperimentConfig, load_experiment_config
from audiocodec.models.codec import build_codec_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export neural codec reconstructions for a benchmark manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec-label", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-codes", action="store_true")
    return parser.parse_args()


def _load_config_from_checkpoint(checkpoint: dict, fallback_path: Path | None) -> CodecExperimentConfig:
    payload = checkpoint.get("config")
    if isinstance(payload, dict):
        return CodecExperimentConfig.from_dict(payload)
    if fallback_path is None:
        raise ValueError("Checkpoint does not contain a config payload. Pass --config.")
    return load_experiment_config(fallback_path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    recon_dir = output_dir / "reconstructions"
    codes_dir = output_dir / "codes"
    manifest_path = output_dir / "manifest.jsonl"
    run_path = output_dir / "run.json"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = _load_config_from_checkpoint(checkpoint, args.config)
    model = build_codec_model(config)
    model.load_state_dict(checkpoint["model"])
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    rows = read_jsonl(args.manifest)
    bits_per_code = math.ceil(math.log2(config.quantizer.codebook_size))
    checkpoint_step = int(checkpoint.get("step", -1))
    exported_rows: list[dict] = []

    for row in rows:
        source_path = Path(row["source_path"]).expanduser().resolve()
        waveform = load_audio(
            source_path,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        num_samples = int(waveform.shape[-1])
        duration_seconds = num_samples / config.audio.sample_rate

        with torch.no_grad():
            encoded = model.encode(waveform.unsqueeze(0).to(device))
            reconstruction = model.decode(encoded.codes, length=num_samples).cpu()[0]
            codes = encoded.codes.detach().cpu()

        num_frames = int(codes.shape[-1])
        payload_bytes = rvq_payload_bytes(
            num_frames=num_frames,
            num_quantizers=config.quantizer.num_quantizers,
            codebook_size=config.quantizer.codebook_size,
        )
        recon_path = recon_dir / f"{row['id']}.wav"
        save_audio(recon_path, reconstruction, config.audio.sample_rate)

        codes_path = None
        if args.save_codes:
            codes_path = codes_dir / f"{row['id']}.pt"
            codes_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(codes, codes_path)

        exported_rows.append(
            {
                "id": row["id"],
                "codec_family": "neural",
                "codec_name": "neural_codec",
                "codec_label": args.codec_label,
                "source_path": str(source_path),
                "reconstruction_path": str(recon_path),
                "codes_path": str(codes_path) if codes_path is not None else None,
                "duration_seconds": duration_seconds,
                "num_samples": num_samples,
                "sample_rate": config.audio.sample_rate,
                "channels": config.audio.channels,
                "pcm16_bytes": row["pcm16_bytes"],
                "compressed_bytes": payload_bytes,
                "actual_bitrate_kbps": bytes_to_kbps(payload_bytes, duration_seconds),
                "payload_bytes": payload_bytes,
                "payload_bits": payload_bytes * 8,
                "num_frames": num_frames,
                "bits_per_code": bits_per_code,
                "num_quantizers": config.quantizer.num_quantizers,
                "codebook_size": config.quantizer.codebook_size,
                "checkpoint_path": str(args.checkpoint.resolve()),
                "checkpoint_step": checkpoint_step,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_path, exported_rows)
    run_path.write_text(
        json.dumps(
            {
                "codec_family": "neural",
                "codec_label": args.codec_label,
                "checkpoint_path": str(args.checkpoint.resolve()),
                "checkpoint_step": checkpoint_step,
                "items": len(exported_rows),
            },
            indent=2,
        )
    )
    print(f"Wrote neural benchmark run to {output_dir}")


if __name__ == "__main__":
    main()
