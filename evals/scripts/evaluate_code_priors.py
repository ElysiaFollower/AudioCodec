#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import ensure_directory, read_jsonl, write_jsonl


CODE_PRIOR_SCHEMA_VERSION = "code-prior-entropy-v1"
TOKEN_ORDERING = "time_major_frame_stage_coarse_to_fine"
PRIOR_CHOICES = ("unigram", "previous_frame")


@dataclass(frozen=True)
class CodeSequence:
    row: dict
    path: Path
    codes: torch.Tensor


@dataclass(frozen=True)
class PriorMetadata:
    frame_rate: int
    num_quantizers: int
    codebook_size: int
    bits_per_code: int
    nominal_bitrate_kbps: float


@dataclass(frozen=True)
class UnigramModel:
    probabilities: torch.Tensor
    smoothing: float


@dataclass(frozen=True)
class PreviousFrameModel:
    conditional_counts: torch.Tensor
    context_counts: torch.Tensor
    fallback_unigram: UnigramModel
    smoothing: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen RVQ code-prior entropy baselines.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Phase 1 neural export directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Training manifest. Defaults to export-dir/manifest.jsonl.")
    parser.add_argument(
        "--eval-export-dir",
        type=Path,
        default=None,
        help="Optional separate Phase 1 export directory for held-out evaluation.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Optional held-out eval manifest. Defaults to eval-export-dir/manifest.jsonl when provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--priors", nargs="+", choices=PRIOR_CHOICES, default=list(PRIOR_CHOICES))
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    return parser.parse_args()


def _relative_path_from_manifest(manifest_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _load_codes(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path} did not contain a torch.Tensor.")
    if tensor.ndim != 3:
        raise ValueError(f"{path} must have shape [batch, quantizers, frames].")
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{path} has no frames.")
    return tensor.detach().cpu().long()


def _load_sequences(manifest_path: Path, max_items: int | None = None) -> list[CodeSequence]:
    rows = read_jsonl(manifest_path)
    selected = rows[:max_items] if max_items is not None else rows
    sequences: list[CodeSequence] = []
    for row in selected:
        path = _relative_path_from_manifest(manifest_path, row.get("codes_path"))
        if path is None:
            continue
        sequences.append(CodeSequence(row=row, path=path, codes=_load_codes(path)))
    if not sequences:
        raise ValueError(f"No codes_path entries found in {manifest_path}.")
    return sequences


def _first_present_int(rows: list[dict], key: str) -> int | None:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def _max_code_value(sequences: list[CodeSequence]) -> int:
    return max(int(sequence.codes.max().item()) for sequence in sequences)


def _metadata(
    *,
    train_sequences: list[CodeSequence],
    eval_sequences: list[CodeSequence],
    codebook_size_override: int | None,
) -> PriorMetadata:
    all_sequences = [*train_sequences, *eval_sequences]
    rows = [sequence.row for sequence in all_sequences]
    frame_rates = {int(row["frame_rate"]) for row in rows if row.get("frame_rate") is not None}
    if len(frame_rates) != 1:
        raise ValueError(f"Expected exactly one frame_rate across manifests, got {sorted(frame_rates)}.")
    frame_rate = frame_rates.pop()

    quantizers = {int(sequence.codes.shape[1]) for sequence in all_sequences}
    if len(quantizers) != 1:
        raise ValueError(f"Expected one num_quantizers across code tensors, got {sorted(quantizers)}.")
    num_quantizers = quantizers.pop()

    codebook_size = codebook_size_override or _first_present_int(rows, "codebook_size")
    if codebook_size is None:
        codebook_size = _max_code_value(all_sequences) + 1
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than 1.")

    observed_max = _max_code_value(all_sequences)
    if observed_max >= codebook_size:
        raise ValueError(f"Observed code {observed_max} but codebook_size is {codebook_size}.")

    bits_per_code = _first_present_int(rows, "bits_per_code") or math.ceil(math.log2(codebook_size))
    nominal_bitrate = float(frame_rate * num_quantizers * bits_per_code) / 1000.0
    return PriorMetadata(
        frame_rate=frame_rate,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        bits_per_code=bits_per_code,
        nominal_bitrate_kbps=nominal_bitrate,
    )


def _stage_tokens(sequences: list[CodeSequence], stage: int) -> torch.Tensor:
    parts = [sequence.codes[:, stage, :].reshape(-1) for sequence in sequences]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)


def _fit_unigram(
    sequences: list[CodeSequence],
    *,
    num_quantizers: int,
    codebook_size: int,
    smoothing: float,
) -> UnigramModel:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    counts = torch.zeros((num_quantizers, codebook_size), dtype=torch.float64)
    for stage in range(num_quantizers):
        tokens = _stage_tokens(sequences, stage)
        counts[stage] = torch.bincount(tokens, minlength=codebook_size).to(torch.float64)
    probabilities = (counts + smoothing) / (counts.sum(dim=1, keepdim=True) + smoothing * codebook_size)
    return UnigramModel(probabilities=probabilities, smoothing=smoothing)


def _fit_previous_frame(
    sequences: list[CodeSequence],
    *,
    num_quantizers: int,
    codebook_size: int,
    smoothing: float,
) -> PreviousFrameModel:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    conditional_counts = torch.zeros((num_quantizers, codebook_size, codebook_size), dtype=torch.float64)
    for sequence in sequences:
        for stage in range(num_quantizers):
            stage_codes = sequence.codes[:, stage, :]
            if stage_codes.shape[-1] <= 1:
                continue
            contexts = stage_codes[:, :-1].reshape(-1)
            targets = stage_codes[:, 1:].reshape(-1)
            pair_ids = contexts * codebook_size + targets
            pair_counts = torch.bincount(pair_ids, minlength=codebook_size * codebook_size).to(torch.float64)
            conditional_counts[stage] += pair_counts.view(codebook_size, codebook_size)
    context_counts = conditional_counts.sum(dim=-1)
    fallback = _fit_unigram(
        sequences,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        smoothing=smoothing,
    )
    return PreviousFrameModel(
        conditional_counts=conditional_counts,
        context_counts=context_counts,
        fallback_unigram=fallback,
        smoothing=smoothing,
    )


def _nll_bits_from_probabilities(probabilities: torch.Tensor, tokens: torch.Tensor) -> float:
    if tokens.numel() == 0:
        return 0.0
    token_probs = probabilities[tokens]
    return float((-torch.log2(token_probs)).sum().item())


def _unigram_stage_row(
    *,
    model: UnigramModel,
    sequences: list[CodeSequence],
    metadata: PriorMetadata,
    stage: int,
) -> dict:
    tokens = _stage_tokens(sequences, stage)
    total_nll = _nll_bits_from_probabilities(model.probabilities[stage], tokens)
    bits_per_code = total_nll / float(tokens.numel()) if tokens.numel() else None
    return {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "prior_family": "unigram",
        "prior_name": "unigram_per_stage",
        "stage": stage,
        "conditioning": "stage_unigram",
        "context_scope": "none",
        "context_window_frames": 0,
        "context_window_seconds": 0.0,
        "token_ordering": TOKEN_ORDERING,
        "num_tokens": int(tokens.numel()),
        "num_initial_tokens": int(tokens.numel()),
        "num_transition_tokens": 0,
        "total_nll_bits": total_nll,
        "stage_bits_per_code": bits_per_code,
        "frame_rate": metadata.frame_rate,
        "codebook_size": metadata.codebook_size,
        "bits_per_code": metadata.bits_per_code,
    }


def _previous_frame_stage_row(
    *,
    model: PreviousFrameModel,
    sequences: list[CodeSequence],
    metadata: PriorMetadata,
    stage: int,
) -> dict:
    initial_nll = 0.0
    transition_nll = 0.0
    initial_tokens = 0
    transition_tokens = 0
    for sequence in sequences:
        stage_codes = sequence.codes[:, stage, :]
        initial = stage_codes[:, 0].reshape(-1)
        initial_nll += _nll_bits_from_probabilities(model.fallback_unigram.probabilities[stage], initial)
        initial_tokens += int(initial.numel())
        if stage_codes.shape[-1] <= 1:
            continue
        contexts = stage_codes[:, :-1].reshape(-1)
        targets = stage_codes[:, 1:].reshape(-1)
        counts = model.conditional_counts[stage, contexts, targets]
        denominators = model.context_counts[stage, contexts] + model.smoothing * metadata.codebook_size
        probabilities = (counts + model.smoothing) / denominators
        transition_nll += float((-torch.log2(probabilities)).sum().item())
        transition_tokens += int(targets.numel())

    total_tokens = initial_tokens + transition_tokens
    total_nll = initial_nll + transition_nll
    bits_per_code = total_nll / float(total_tokens) if total_tokens else None
    return {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "prior_family": "previous_frame",
        "prior_name": "previous_frame_markov_per_stage",
        "stage": stage,
        "conditioning": "same_stage_previous_frame_with_unigram_first_frame",
        "context_scope": "local",
        "context_window_frames": 1,
        "context_window_seconds": 1.0 / float(metadata.frame_rate),
        "token_ordering": TOKEN_ORDERING,
        "num_tokens": total_tokens,
        "num_initial_tokens": initial_tokens,
        "num_transition_tokens": transition_tokens,
        "total_nll_bits": total_nll,
        "stage_bits_per_code": bits_per_code,
        "frame_rate": metadata.frame_rate,
        "codebook_size": metadata.codebook_size,
        "bits_per_code": metadata.bits_per_code,
    }


def _aggregate_prior(rows: list[dict], metadata: PriorMetadata, unigram_bitrate: float | None) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate empty prior rows.")
    total_nll = sum(float(row["total_nll_bits"]) for row in rows)
    total_tokens = sum(int(row["num_tokens"]) for row in rows)
    stage_bits = [row["stage_bits_per_code"] for row in sorted(rows, key=lambda item: int(item["stage"]))]
    estimated_bitrate = float(metadata.frame_rate * sum(float(value) for value in stage_bits if value is not None)) / 1000.0
    entropy_savings = 1.0 - estimated_bitrate / metadata.nominal_bitrate_kbps
    if unigram_bitrate is None or estimated_bitrate == unigram_bitrate:
        relative_improvement = 0.0
    else:
        relative_improvement = 1.0 - estimated_bitrate / unigram_bitrate
    first = rows[0]
    return {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "prior_family": first["prior_family"],
        "prior_name": first["prior_name"],
        "conditioning": first["conditioning"],
        "context_scope": first["context_scope"],
        "context_window_frames": first["context_window_frames"],
        "context_window_seconds": first["context_window_seconds"],
        "token_ordering": TOKEN_ORDERING,
        "num_tokens": total_tokens,
        "total_nll_bits": total_nll,
        "bits_per_code": total_nll / float(total_tokens) if total_tokens else None,
        "stage_bits_per_code": stage_bits,
        "estimated_entropy_bitrate_kbps": estimated_bitrate,
        "nominal_bitrate_kbps": metadata.nominal_bitrate_kbps,
        "entropy_savings_ratio": entropy_savings,
        "relative_improvement_vs_unigram": relative_improvement,
    }


def evaluate_priors(
    *,
    train_sequences: list[CodeSequence],
    eval_sequences: list[CodeSequence],
    metadata: PriorMetadata,
    priors: list[str],
    smoothing: float,
) -> tuple[list[dict], list[dict]]:
    stage_rows: list[dict] = []
    summary_rows: list[dict] = []
    unigram_model = _fit_unigram(
        train_sequences,
        num_quantizers=metadata.num_quantizers,
        codebook_size=metadata.codebook_size,
        smoothing=smoothing,
    )
    unigram_summary_bitrate: float | None = None

    if "unigram" in priors:
        rows = [
            _unigram_stage_row(model=unigram_model, sequences=eval_sequences, metadata=metadata, stage=stage)
            for stage in range(metadata.num_quantizers)
        ]
        stage_rows.extend(rows)
        summary = _aggregate_prior(rows, metadata, unigram_bitrate=None)
        unigram_summary_bitrate = summary["estimated_entropy_bitrate_kbps"]
        summary_rows.append(summary)

    if "previous_frame" in priors:
        model = _fit_previous_frame(
            train_sequences,
            num_quantizers=metadata.num_quantizers,
            codebook_size=metadata.codebook_size,
            smoothing=smoothing,
        )
        rows = [
            _previous_frame_stage_row(model=model, sequences=eval_sequences, metadata=metadata, stage=stage)
            for stage in range(metadata.num_quantizers)
        ]
        stage_rows.extend(rows)
        if unigram_summary_bitrate is None:
            unigram_rows = [
                _unigram_stage_row(model=unigram_model, sequences=eval_sequences, metadata=metadata, stage=stage)
                for stage in range(metadata.num_quantizers)
            ]
            unigram_summary_bitrate = _aggregate_prior(unigram_rows, metadata, unigram_bitrate=None)[
                "estimated_entropy_bitrate_kbps"
            ]
        summary_rows.append(_aggregate_prior(rows, metadata, unigram_bitrate=unigram_summary_bitrate))

    return stage_rows, summary_rows


def _with_dataset_split(rows: list[dict], dataset_split: str) -> list[dict]:
    return [{**row, "dataset_split": dataset_split} for row in rows]


def run_code_prior_evaluation(
    *,
    export_dir: Path,
    manifest_path: Path,
    eval_export_dir: Path | None,
    eval_manifest_path: Path | None,
    output_dir: Path,
    priors: list[str],
    codebook_size: int | None,
    smoothing: float,
    max_train_items: int | None = None,
    max_eval_items: int | None = None,
) -> tuple[list[dict], dict]:
    train_sequences = _load_sequences(manifest_path, max_items=max_train_items)
    resolved_eval_manifest = eval_manifest_path
    if resolved_eval_manifest is None and eval_export_dir is not None:
        resolved_eval_manifest = eval_export_dir / "manifest.jsonl"
    eval_sequences = (
        _load_sequences(resolved_eval_manifest, max_items=max_eval_items)
        if resolved_eval_manifest is not None
        else train_sequences
    )
    metadata = _metadata(
        train_sequences=train_sequences,
        eval_sequences=eval_sequences,
        codebook_size_override=codebook_size,
    )
    train_stage_rows, _ = evaluate_priors(
        train_sequences=train_sequences,
        eval_sequences=train_sequences,
        metadata=metadata,
        priors=priors,
        smoothing=smoothing,
    )
    eval_stage_rows, prior_summaries = evaluate_priors(
        train_sequences=train_sequences,
        eval_sequences=eval_sequences,
        metadata=metadata,
        priors=priors,
        smoothing=smoothing,
    )

    ensure_directory(output_dir)
    write_jsonl(output_dir / "train_metrics.jsonl", _with_dataset_split(train_stage_rows, "train"))
    evaluation_split = "heldout" if resolved_eval_manifest is not None else "self_eval"
    write_jsonl(output_dir / "val_metrics.jsonl", _with_dataset_split(eval_stage_rows, evaluation_split))
    config = {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "export_dir": str(export_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "eval_export_dir": str(eval_export_dir.resolve()) if eval_export_dir is not None else None,
        "eval_manifest_path": str(resolved_eval_manifest.resolve()) if resolved_eval_manifest is not None else None,
        "evaluation_split": evaluation_split,
        "priors": priors,
        "smoothing": smoothing,
        "token_ordering": TOKEN_ORDERING,
        "max_train_items": max_train_items,
        "max_eval_items": max_eval_items,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True))
    summary = {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "frame_rate": metadata.frame_rate,
        "num_quantizers": metadata.num_quantizers,
        "codebook_size": metadata.codebook_size,
        "bits_per_code": metadata.bits_per_code,
        "nominal_bitrate_kbps": metadata.nominal_bitrate_kbps,
        "token_ordering": TOKEN_ORDERING,
        "evaluation_split": config["evaluation_split"],
        "train_items": len(train_sequences),
        "eval_items": len(eval_sequences),
        "priors": prior_summaries,
        "blocked_priors": [
            {
                "prior_family": "local_tcn",
                "reason": "not produced by analytic evaluator; use train_code_prior.py for trained local TCN",
            },
            {
                "prior_family": "long_transformer",
                "reason": "not produced by analytic evaluator; use train_code_prior.py for trained long Transformer",
            },
            {
                "prior_family": "mamba",
                "reason": "Mamba/SSM dependency is not fixed and is not a Phase 3 blocker",
            },
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    return eval_stage_rows, summary


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else export_dir / "manifest.jsonl"
    eval_export_dir = args.eval_export_dir.resolve() if args.eval_export_dir is not None else None
    eval_manifest_path = args.eval_manifest.resolve() if args.eval_manifest is not None else None
    output_dir = args.output_dir.resolve() if args.output_dir is not None else export_dir / "code_priors"
    stage_rows, summary = run_code_prior_evaluation(
        export_dir=export_dir,
        manifest_path=manifest_path,
        eval_export_dir=eval_export_dir,
        eval_manifest_path=eval_manifest_path,
        output_dir=output_dir,
        priors=list(args.priors),
        codebook_size=args.codebook_size,
        smoothing=args.smoothing,
        max_train_items=args.max_train_items,
        max_eval_items=args.max_eval_items,
    )
    print(
        f"Wrote {len(stage_rows)} code-prior metric rows "
        f"for {len(summary['priors'])} priors to {output_dir}"
    )


if __name__ == "__main__":
    main()
