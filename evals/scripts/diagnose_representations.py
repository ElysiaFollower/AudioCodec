#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import ensure_directory, read_jsonl, write_jsonl


DIAGNOSTICS_SCHEMA_VERSION = "representation-diagnostics-v1"
DEFAULT_SCOPE_SECONDS = {
    "local": 1.0,
    "medium": 5.0,
    "long": 30.0,
    "full_utterance": None,
}
REPRESENTATIONS = ("latent", "quantized", "codes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen representation redundancy diagnostics.")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--local-window-seconds", type=float, default=DEFAULT_SCOPE_SECONDS["local"])
    parser.add_argument("--medium-window-seconds", type=float, default=DEFAULT_SCOPE_SECONDS["medium"])
    parser.add_argument("--long-window-seconds", type=float, default=DEFAULT_SCOPE_SECONDS["long"])
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def _window_frames(scope: str, frame_rate: int, seconds: float | None) -> int | None:
    if scope == "full_utterance":
        return None
    if seconds is None:
        raise ValueError(f"{scope} requires a window length in seconds.")
    if seconds <= 0:
        raise ValueError("window seconds must be positive.")
    return max(1, int(round(seconds * frame_rate)))


def _scope_specs(args: argparse.Namespace) -> list[dict]:
    return [
        {
            "context_scope": "local",
            "context_window_seconds": args.local_window_seconds,
            "context_window_frames": None,
        },
        {
            "context_scope": "medium",
            "context_window_seconds": args.medium_window_seconds,
            "context_window_frames": None,
        },
        {
            "context_scope": "long",
            "context_window_seconds": args.long_window_seconds,
            "context_window_frames": None,
        },
        {
            "context_scope": "full_utterance",
            "context_window_seconds": None,
            "context_window_frames": None,
        },
    ]


def _relative_path_from_manifest(manifest_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path} did not contain a torch.Tensor.")
    return tensor.detach().cpu()


def _normalize_sequence(tensor: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "codes":
        if tensor.ndim != 3:
            raise ValueError("codes tensor must have shape [batch, stages, frames].")
        return tensor.long()
    if tensor.ndim != 3:
        raise ValueError(f"{representation} tensor must have shape [batch, channels, frames].")
    return tensor.float()


def _effective_window_frames(window_frames: int | None, num_frames: int) -> int:
    if num_frames <= 1:
        return 0
    if window_frames is None:
        return num_frames - 1
    return min(max(1, window_frames), num_frames - 1)


def _past_window_mean_metrics(sequence: torch.Tensor, window_frames: int | None) -> dict:
    batch, channels, num_frames = sequence.shape
    effective_window = _effective_window_frames(window_frames, num_frames)
    if effective_window <= 0:
        return {
            "metric_family": "continuous_past_window_mean",
            "num_prediction_frames": 0,
            "effective_window_frames": effective_window,
            "mse": None,
            "variance": None,
            "normalized_mse": None,
            "predictability_score": None,
        }

    targets = []
    predictions = []
    for index in range(1, num_frames):
        start = 0 if window_frames is None else max(0, index - effective_window)
        predictions.append(sequence[:, :, start:index].mean(dim=-1))
        targets.append(sequence[:, :, index])
    target = torch.stack(targets, dim=-1)
    prediction = torch.stack(predictions, dim=-1)
    mse = torch.mean((target - prediction) ** 2).item()
    variance = torch.var(target, unbiased=False).item()
    normalized_mse = mse / variance if variance > 0 else 0.0 if mse == 0 else None
    predictability = 1.0 - normalized_mse if normalized_mse is not None else None
    return {
        "metric_family": "continuous_past_window_mean",
        "num_prediction_frames": int(num_frames - 1),
        "effective_window_frames": int(effective_window),
        "mse": mse,
        "variance": variance,
        "normalized_mse": normalized_mse,
        "predictability_score": predictability,
        "batch_size": int(batch),
        "channels": int(channels),
    }


def _marginal_entropy_bits(codes: torch.Tensor) -> float:
    values = codes.reshape(-1)
    if values.numel() == 0:
        return 0.0
    _, counts = torch.unique(values, return_counts=True)
    probabilities = counts.float() / float(values.numel())
    entropy = -(probabilities * torch.log2(probabilities)).sum()
    return float(entropy.item())


def _code_reuse_metrics(codes: torch.Tensor, window_frames: int | None) -> dict:
    batch, stages, num_frames = codes.shape
    effective_window = _effective_window_frames(window_frames, num_frames)
    if effective_window <= 0:
        return {
            "metric_family": "code_window_reuse",
            "num_prediction_frames": 0,
            "effective_window_frames": effective_window,
            "window_reuse_rate": None,
            "previous_frame_match_rate": None,
            "marginal_entropy_bits_per_code": _marginal_entropy_bits(codes),
        }

    previous_matches = []
    window_hits = []
    for index in range(1, num_frames):
        current = codes[:, :, index]
        previous = codes[:, :, index - 1]
        previous_matches.append(current.eq(previous))

        start = 0 if window_frames is None else max(0, index - effective_window)
        history = codes[:, :, start:index]
        window_hits.append(history.eq(current.unsqueeze(-1)).any(dim=-1))

    previous_match_rate = torch.stack(previous_matches, dim=-1).float().mean().item()
    window_reuse_rate = torch.stack(window_hits, dim=-1).float().mean().item()
    distinct_per_stage = []
    for stage in range(stages):
        distinct_per_stage.append(float(torch.unique(codes[:, stage, :]).numel()))
    return {
        "metric_family": "code_window_reuse",
        "num_prediction_frames": int(num_frames - 1),
        "effective_window_frames": int(effective_window),
        "window_reuse_rate": window_reuse_rate,
        "previous_frame_match_rate": previous_match_rate,
        "marginal_entropy_bits_per_code": _marginal_entropy_bits(codes),
        "distinct_codes_per_stage_mean": sum(distinct_per_stage) / len(distinct_per_stage),
        "batch_size": int(batch),
        "num_quantizers": int(stages),
    }


def _representation_path(row: dict, manifest_path: Path, representation: str) -> Path | None:
    key = f"{representation}_path"
    return _relative_path_from_manifest(manifest_path, row.get(key))


def _diagnose_tensor(
    *,
    row: dict,
    representation: str,
    tensor: torch.Tensor,
    representation_path: Path,
    scope: dict,
) -> dict:
    frame_rate = int(row["frame_rate"])
    num_frames = int(tensor.shape[-1])
    scope_name = scope["context_scope"]
    requested_seconds = scope["context_window_seconds"]
    requested_frames = _window_frames(scope_name, frame_rate, requested_seconds)
    if representation == "codes":
        metrics = _code_reuse_metrics(tensor, requested_frames)
    else:
        metrics = _past_window_mean_metrics(tensor, requested_frames)
    effective_seconds = (
        metrics["effective_window_frames"] / frame_rate if metrics.get("effective_window_frames") is not None else None
    )
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "id": row["id"],
        "representation": representation,
        "representation_path": str(representation_path),
        "shape": list(tensor.shape),
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "context_scope": scope_name,
        "context_window_seconds": requested_seconds,
        "context_window_frames": requested_frames,
        "effective_window_seconds": effective_seconds,
        **metrics,
    }


def _metric_value(row: dict) -> float | None:
    if row["representation"] == "codes":
        return row.get("window_reuse_rate")
    return row.get("predictability_score")


def _summarize(rows: list[dict], improvement_threshold: float) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["representation"], row["context_scope"]), []).append(row)

    aggregates: list[dict] = []
    mean_by_key: dict[tuple[str, str], float] = {}
    for (representation, scope), items in sorted(grouped.items()):
        values = [value for item in items if (value := _metric_value(item)) is not None]
        mean_value = sum(values) / len(values) if values else None
        if mean_value is not None:
            mean_by_key[(representation, scope)] = mean_value
        aggregates.append(
            {
                "representation": representation,
                "context_scope": scope,
                "items": len(items),
                "metric": "window_reuse_rate" if representation == "codes" else "predictability_score",
                "mean_value": mean_value,
            }
        )

    gates = []
    for representation in sorted({row["representation"] for row in rows}):
        local_value = mean_by_key.get((representation, "local"))
        candidates = {
            scope: mean_by_key.get((representation, scope))
            for scope in ("long", "full_utterance")
            if mean_by_key.get((representation, scope)) is not None
        }
        best_scope = None
        best_value = None
        if candidates:
            best_scope, best_value = max(candidates.items(), key=lambda item: item[1])
        if local_value is None or best_value is None:
            improvement = None
            passed = False
        else:
            denominator = abs(local_value) if abs(local_value) > 1e-12 else 1.0
            improvement = (best_value - local_value) / denominator
            passed = improvement >= improvement_threshold
        gates.append(
            {
                "representation": representation,
                "local_value": local_value,
                "best_long_or_full_scope": best_scope,
                "best_long_or_full_value": best_value,
                "relative_improvement": improvement,
                "threshold": improvement_threshold,
                "passed": passed,
            }
        )

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "items": len(rows),
        "aggregates": aggregates,
        "gate_recommendations": gates,
    }


def run_diagnostics(
    *,
    export_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    representations: list[str],
    scope_specs: list[dict],
    max_items: int | None = None,
    improvement_threshold: float = 0.05,
) -> tuple[list[dict], dict]:
    rows = read_jsonl(manifest_path)
    selected = rows[:max_items] if max_items is not None else rows
    diagnostics: list[dict] = []
    for row in selected:
        for representation in representations:
            path = _representation_path(row, manifest_path, representation)
            if path is None:
                continue
            tensor = _normalize_sequence(_load_tensor(path), representation)
            for scope in scope_specs:
                diagnostics.append(
                    _diagnose_tensor(
                        row=row,
                        representation=representation,
                        tensor=tensor,
                        representation_path=path,
                        scope=scope,
                    )
                )

    ensure_directory(output_dir)
    write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
    summary = _summarize(diagnostics, improvement_threshold=improvement_threshold)
    summary["export_dir"] = str(export_dir.resolve())
    summary["manifest_path"] = str(manifest_path.resolve())
    summary["representations"] = representations
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    return diagnostics, summary


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else export_dir / "manifest.jsonl"
    output_dir = args.output_dir.resolve() if args.output_dir is not None else export_dir / "diagnostics"
    diagnostics, _ = run_diagnostics(
        export_dir=export_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        representations=list(args.representations),
        scope_specs=_scope_specs(args),
        max_items=args.max_items,
    )
    print(f"Wrote {len(diagnostics)} diagnostic rows to {output_dir}")


if __name__ == "__main__":
    main()
