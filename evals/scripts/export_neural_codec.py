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

from _common import (
    bytes_to_kbps,
    load_audio,
    pcm16_bytes,
    read_jsonl,
    resolve_device,
    rvq_payload_bytes,
    save_audio,
    write_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audiocodec.config import CodecExperimentConfig, load_experiment_config
from audiocodec.models.codec import build_codec_model


CONTEXT_EXPORT_SCHEMA_VERSION = "context-export-v1"
CONTEXT_SCOPES = ("none", "local", "medium", "long", "full_utterance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export neural codec reconstructions for a benchmark manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec-label", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-codes", action="store_true")
    parser.add_argument(
        "--save-representations",
        action="store_true",
        help="Save latent, quantized, and RVQ code tensors for context-modeling diagnostics.",
    )
    parser.add_argument("--context-scope", choices=CONTEXT_SCOPES, default="none")
    parser.add_argument("--context-window-seconds", type=float, default=None)
    parser.add_argument("--context-window-frames", type=int, default=None)
    parser.add_argument("--clip-scope", type=str, default=None)
    parser.add_argument("--is-full-utterance", action="store_true")
    return parser.parse_args()


def _load_config_from_checkpoint(checkpoint: dict, fallback_path: Path | None) -> CodecExperimentConfig:
    payload = checkpoint.get("config")
    if isinstance(payload, dict):
        return CodecExperimentConfig.from_dict(payload)
    if fallback_path is None:
        raise ValueError("Checkpoint does not contain a config payload. Pass --config.")
    return load_experiment_config(fallback_path)


def _rvq_payload_bits(num_frames: int, num_quantizers: int, bits_per_code: int) -> int:
    return int(num_frames) * int(num_quantizers) * int(bits_per_code)


def _nominal_bitrate_kbps(frame_rate: int, num_quantizers: int, bits_per_code: int) -> float:
    return float(frame_rate * num_quantizers * bits_per_code) / 1000.0


def _resolve_context_window_frames(
    *,
    context_window_seconds: float | None,
    context_window_frames: int | None,
    frame_rate: int,
) -> int | None:
    if context_window_frames is not None:
        if context_window_frames < 0:
            raise ValueError("context_window_frames must be non-negative.")
        return context_window_frames
    if context_window_seconds is None:
        return None
    if context_window_seconds < 0:
        raise ValueError("context_window_seconds must be non-negative.")
    return int(round(context_window_seconds * frame_rate))


def _resolve_context_window_seconds(
    *,
    context_window_seconds: float | None,
    context_window_frames: int | None,
    frame_rate: int,
) -> float | None:
    if context_window_seconds is not None:
        if context_window_seconds < 0:
            raise ValueError("context_window_seconds must be non-negative.")
        return context_window_seconds
    if context_window_frames is None:
        return None
    if context_window_frames < 0:
        raise ValueError("context_window_frames must be non-negative.")
    return float(context_window_frames) / float(frame_rate)


def _representation_paths(representations_dir: Path, item_id: str) -> dict[str, Path]:
    return {
        "codes": representations_dir / f"{item_id}.codes.pt",
        "latent": representations_dir / f"{item_id}.latent.pt",
        "quantized": representations_dir / f"{item_id}.quantized.pt",
    }


def _save_representations(
    representations_dir: Path,
    item_id: str,
    *,
    codes: torch.Tensor,
    latent: torch.Tensor,
    quantized: torch.Tensor,
) -> dict[str, Path]:
    paths = _representation_paths(representations_dir, item_id)
    representations_dir.mkdir(parents=True, exist_ok=True)
    torch.save(codes, paths["codes"])
    torch.save(latent, paths["latent"])
    torch.save(quantized, paths["quantized"])
    return paths


def _build_manifest_row(
    *,
    source_row: dict,
    source_path: Path,
    reconstruction_path: Path,
    codes_path: Path | None,
    latent_path: Path | None,
    quantized_path: Path | None,
    duration_seconds: float,
    num_samples: int,
    sample_rate: int,
    channels: int,
    num_frames: int,
    frame_rate: int,
    hop_length: int,
    latent_dim: int,
    num_quantizers: int,
    codebook_size: int,
    bits_per_code: int,
    payload_bytes: int,
    codec_label: str,
    checkpoint_path: Path,
    checkpoint_step: int,
    config_path: Path | None,
    context_scope: str,
    context_window_seconds: float | None,
    context_window_frames: int | None,
    clip_scope: str,
    is_full_utterance: bool,
) -> dict:
    exact_payload_bits = _rvq_payload_bits(
        num_frames=num_frames,
        num_quantizers=num_quantizers,
        bits_per_code=bits_per_code,
    )
    return {
        "id": source_row["id"],
        "schema_version": CONTEXT_EXPORT_SCHEMA_VERSION,
        "codec_family": "neural",
        "codec_name": "neural_codec",
        "codec_label": codec_label,
        "source_path": str(source_path),
        "reconstruction_path": str(reconstruction_path),
        "codes_path": str(codes_path) if codes_path is not None else None,
        "latent_path": str(latent_path) if latent_path is not None else None,
        "quantized_path": str(quantized_path) if quantized_path is not None else None,
        "duration_seconds": duration_seconds,
        "num_samples": num_samples,
        "sample_rate": sample_rate,
        "channels": channels,
        "pcm16_bytes": source_row.get("pcm16_bytes", pcm16_bytes(num_samples, channels=channels)),
        "compressed_bytes": payload_bytes,
        "actual_bitrate_kbps": bytes_to_kbps(payload_bytes, duration_seconds),
        "payload_bytes": payload_bytes,
        "payload_bits": payload_bytes * 8,
        "rvq_payload_bytes": payload_bytes,
        "rvq_payload_bits": exact_payload_bits,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "hop_length": hop_length,
        "latent_dim": latent_dim,
        "bits_per_code": bits_per_code,
        "num_quantizers": num_quantizers,
        "codebook_size": codebook_size,
        "nominal_bitrate_kbps": _nominal_bitrate_kbps(
            frame_rate=frame_rate,
            num_quantizers=num_quantizers,
            bits_per_code=bits_per_code,
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "context_scope": context_scope,
        "context_window_seconds": context_window_seconds,
        "context_window_frames": context_window_frames,
        "clip_scope": clip_scope,
        "clip_duration_seconds": duration_seconds,
        "is_full_utterance": is_full_utterance,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    recon_dir = output_dir / "reconstructions"
    codes_dir = output_dir / "codes"
    representations_dir = output_dir / "representations"
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
    context_window_frames = _resolve_context_window_frames(
        context_window_seconds=args.context_window_seconds,
        context_window_frames=args.context_window_frames,
        frame_rate=model.frame_rate,
    )
    context_window_seconds = _resolve_context_window_seconds(
        context_window_seconds=args.context_window_seconds,
        context_window_frames=context_window_frames,
        frame_rate=model.frame_rate,
    )
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
            output = model(waveform.unsqueeze(0).to(device))
            reconstruction = output.reconstruction.detach().cpu()[0]
            codes = output.codes.detach().cpu()
            latent = output.latent.detach().cpu()
            quantized = output.quantized.detach().cpu()

        num_frames = int(codes.shape[-1])
        payload_bytes = rvq_payload_bytes(
            num_frames=num_frames,
            num_quantizers=config.quantizer.num_quantizers,
            codebook_size=config.quantizer.codebook_size,
        )
        recon_path = recon_dir / f"{row['id']}.wav"
        save_audio(recon_path, reconstruction, config.audio.sample_rate)

        codes_path = None
        latent_path = None
        quantized_path = None
        if args.save_representations:
            representation_paths = _save_representations(
                representations_dir,
                row["id"],
                codes=codes,
                latent=latent,
                quantized=quantized,
            )
            codes_path = representation_paths["codes"]
            latent_path = representation_paths["latent"]
            quantized_path = representation_paths["quantized"]
        if args.save_codes:
            legacy_codes_path = codes_dir / f"{row['id']}.pt"
            legacy_codes_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(codes, legacy_codes_path)
            if codes_path is None:
                codes_path = legacy_codes_path

        is_full_utterance = bool(row.get("is_full_utterance", args.is_full_utterance))
        clip_scope = row.get(
            "clip_scope",
            args.clip_scope or ("full_utterance" if is_full_utterance else "manifest_item"),
        )
        exported_rows.append(
            _build_manifest_row(
                source_row=row,
                source_path=source_path,
                reconstruction_path=recon_path,
                codes_path=codes_path,
                latent_path=latent_path,
                quantized_path=quantized_path,
                duration_seconds=duration_seconds,
                num_samples=num_samples,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
                num_frames=num_frames,
                frame_rate=model.frame_rate,
                hop_length=model.hop_length,
                latent_dim=int(latent.shape[1]),
                num_quantizers=config.quantizer.num_quantizers,
                codebook_size=config.quantizer.codebook_size,
                bits_per_code=bits_per_code,
                payload_bytes=payload_bytes,
                codec_label=args.codec_label,
                checkpoint_path=args.checkpoint.resolve(),
                checkpoint_step=checkpoint_step,
                config_path=args.config,
                context_scope=args.context_scope,
                context_window_seconds=context_window_seconds,
                context_window_frames=context_window_frames,
                clip_scope=clip_scope,
                is_full_utterance=is_full_utterance,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_path, exported_rows)
    run_path.write_text(
        json.dumps(
            {
                "schema_version": CONTEXT_EXPORT_SCHEMA_VERSION,
                "codec_family": "neural",
                "codec_label": args.codec_label,
                "checkpoint_path": str(args.checkpoint.resolve()),
                "checkpoint_step": checkpoint_step,
                "config_path": str(args.config.resolve()) if args.config is not None else None,
                "sample_rate": config.audio.sample_rate,
                "channels": config.audio.channels,
                "frame_rate": model.frame_rate,
                "hop_length": model.hop_length,
                "latent_dim": config.model.latent_dim,
                "num_quantizers": config.quantizer.num_quantizers,
                "codebook_size": config.quantizer.codebook_size,
                "bits_per_code": bits_per_code,
                "nominal_bitrate_kbps": _nominal_bitrate_kbps(
                    frame_rate=model.frame_rate,
                    num_quantizers=config.quantizer.num_quantizers,
                    bits_per_code=bits_per_code,
                ),
                "save_codes": args.save_codes,
                "save_representations": args.save_representations,
                "representation_kinds": ["codes", "latent", "quantized"] if args.save_representations else [],
                "context_scope": args.context_scope,
                "context_window_seconds": context_window_seconds,
                "context_window_frames": context_window_frames,
                "clip_scope": args.clip_scope,
                "is_full_utterance": args.is_full_utterance,
                "items": len(exported_rows),
            },
            indent=2,
        )
    )
    print(f"Wrote neural benchmark run to {output_dir}")


if __name__ == "__main__":
    main()
