#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import ensure_directory, write_jsonl


CONTEXT_RESULTS_SCHEMA_VERSION = "context-results-v1"
DEFAULT_GATE_THRESHOLD = 0.05


RESULT_FIELDS = [
    "schema_version",
    "stage",
    "result_type",
    "source_file",
    "representation",
    "prior_family",
    "prior_name",
    "context_scope",
    "context_window_frames",
    "context_window_seconds",
    "metric_name",
    "metric_value",
    "bits_per_code",
    "stage_bits_per_code",
    "estimated_entropy_bitrate_kbps",
    "nominal_bitrate_kbps",
    "entropy_savings_ratio",
    "relative_improvement_vs_local_or_unigram",
    "gate_reference",
    "gate_threshold",
    "gate_passed",
    "token_ordering",
    "dataset_split",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect context modeling diagnostics and prior results.")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-summary", type=Path, default=None)
    parser.add_argument("--analytic-prior-summary", type=Path, default=None)
    parser.add_argument("--prior-root", type=Path, default=None)
    parser.add_argument("--trained-prior-summary", action="append", type=Path, default=[])
    parser.add_argument("--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True)


def _base_row(*, stage: str, result_type: str, source_file: Path) -> dict:
    return {
        "schema_version": CONTEXT_RESULTS_SCHEMA_VERSION,
        "stage": stage,
        "result_type": result_type,
        "source_file": str(source_file),
        "representation": None,
        "prior_family": None,
        "prior_name": None,
        "context_scope": None,
        "context_window_frames": None,
        "context_window_seconds": None,
        "metric_name": None,
        "metric_value": None,
        "bits_per_code": None,
        "stage_bits_per_code": None,
        "estimated_entropy_bitrate_kbps": None,
        "nominal_bitrate_kbps": None,
        "entropy_savings_ratio": None,
        "relative_improvement_vs_local_or_unigram": None,
        "gate_reference": None,
        "gate_threshold": None,
        "gate_passed": None,
        "token_ordering": None,
        "dataset_split": None,
        "notes": None,
    }


def _relative_improvement(value: float | None, baseline: float | None, *, higher_is_better: bool) -> float | None:
    if value is None or baseline is None:
        return None
    denominator = abs(baseline) if abs(baseline) > 1e-12 else 1.0
    if higher_is_better:
        return (value - baseline) / denominator
    return (baseline - value) / denominator


def _collect_diagnostics(summary_path: Path, threshold: float) -> tuple[list[dict], dict | None, list[str]]:
    summary = _load_json(summary_path)
    if summary is None:
        return [], None, [f"missing diagnostics summary: {summary_path}"]

    local_by_representation = {
        item["representation"]: item.get("mean_value")
        for item in summary.get("aggregates", [])
        if item.get("context_scope") == "local"
    }
    gate_by_representation = {
        item["representation"]: item for item in summary.get("gate_recommendations", [])
    }

    rows: list[dict] = []
    for item in summary.get("aggregates", []):
        representation = item.get("representation")
        scope = item.get("context_scope")
        value = item.get("mean_value")
        gate = gate_by_representation.get(representation, {})
        relative = _relative_improvement(
            value,
            local_by_representation.get(representation),
            higher_is_better=True,
        )
        row = _base_row(stage="phase2_diagnostics", result_type="representation_diagnostic", source_file=summary_path)
        row.update(
            {
                "representation": representation,
                "context_scope": scope,
                "metric_name": item.get("metric"),
                "metric_value": value,
                "relative_improvement_vs_local_or_unigram": relative,
                "gate_reference": "local",
                "gate_threshold": threshold,
                "gate_passed": bool(gate.get("passed")) if scope == gate.get("best_long_or_full_scope") else False,
                "notes": f"items={item.get('items')}",
            }
        )
        rows.append(row)
    return rows, summary, []


def _prior_summary_rows(summary: dict, source_path: Path) -> list[dict]:
    rows = []
    for item in summary.get("priors", []):
        row = _base_row(stage="phase3_code_prior", result_type="analytic_prior", source_file=source_path)
        row.update(
            {
                "prior_family": item.get("prior_family"),
                "prior_name": item.get("prior_name"),
                "context_scope": item.get("context_scope"),
                "context_window_frames": item.get("context_window_frames"),
                "context_window_seconds": item.get("context_window_seconds"),
                "metric_name": "estimated_entropy_bitrate_kbps",
                "metric_value": item.get("estimated_entropy_bitrate_kbps"),
                "bits_per_code": item.get("bits_per_code"),
                "stage_bits_per_code": _json_or_none(item.get("stage_bits_per_code")),
                "estimated_entropy_bitrate_kbps": item.get("estimated_entropy_bitrate_kbps"),
                "nominal_bitrate_kbps": item.get("nominal_bitrate_kbps"),
                "entropy_savings_ratio": item.get("entropy_savings_ratio"),
                "relative_improvement_vs_local_or_unigram": item.get("relative_improvement_vs_unigram"),
                "gate_reference": "unigram",
                "token_ordering": item.get("token_ordering") or summary.get("token_ordering"),
                "dataset_split": summary.get("evaluation_split"),
            }
        )
        rows.append(row)
    return rows


def _trained_summary_row(summary: dict, source_path: Path) -> dict:
    final_val = summary.get("final_val", {})
    row = _base_row(stage="phase3_code_prior", result_type="trained_prior", source_file=source_path)
    row.update(
        {
            "prior_family": summary.get("prior_family"),
            "prior_name": summary.get("prior_name"),
            "context_scope": summary.get("context_scope"),
            "context_window_frames": summary.get("context_window_frames"),
            "context_window_seconds": summary.get("context_window_seconds"),
            "metric_name": "estimated_entropy_bitrate_kbps",
            "metric_value": final_val.get("estimated_entropy_bitrate_kbps"),
            "bits_per_code": final_val.get("bits_per_code"),
            "stage_bits_per_code": _json_or_none(final_val.get("stage_bits_per_code")),
            "estimated_entropy_bitrate_kbps": final_val.get("estimated_entropy_bitrate_kbps"),
            "nominal_bitrate_kbps": final_val.get("nominal_bitrate_kbps") or summary.get("nominal_bitrate_kbps"),
            "entropy_savings_ratio": final_val.get("entropy_savings_ratio"),
            "token_ordering": summary.get("token_ordering"),
            "dataset_split": final_val.get("dataset_split"),
            "notes": f"checkpoint_path={summary.get('checkpoint_path')}",
        }
    )
    return row


def _collect_priors(
    analytic_summary_path: Path,
    trained_summary_paths: list[Path],
    threshold: float,
) -> tuple[list[dict], dict | None, list[dict], list[str]]:
    warnings: list[str] = []
    analytic_summary = _load_json(analytic_summary_path)
    rows: list[dict] = []
    if analytic_summary is None:
        warnings.append(f"missing analytic prior summary: {analytic_summary_path}")
    else:
        rows.extend(_prior_summary_rows(analytic_summary, analytic_summary_path))

    trained_summaries = []
    for path in trained_summary_paths:
        summary = _load_json(path)
        if summary is None:
            warnings.append(f"missing trained prior summary: {path}")
            continue
        trained_summaries.append(summary)
        rows.append(_trained_summary_row(summary, path))

    unigram_bitrate = next(
        (
            row.get("estimated_entropy_bitrate_kbps")
            for row in rows
            if row.get("prior_family") == "unigram" and row.get("estimated_entropy_bitrate_kbps") is not None
        ),
        None,
    )
    local_candidates = [
        row
        for row in rows
        if row.get("context_scope") in {"local", "none"}
        and row.get("estimated_entropy_bitrate_kbps") is not None
        and row.get("prior_family") != "unigram"
    ]
    best_local = min(local_candidates, key=lambda row: float(row["estimated_entropy_bitrate_kbps"]), default=None)
    best_local_bitrate = best_local.get("estimated_entropy_bitrate_kbps") if best_local else unigram_bitrate
    best_local_name = best_local.get("prior_family") if best_local else "unigram"

    for row in rows:
        bitrate = row.get("estimated_entropy_bitrate_kbps")
        if row.get("prior_family") == "unigram":
            row["relative_improvement_vs_local_or_unigram"] = 0.0
            row["gate_reference"] = "self"
            row["gate_threshold"] = threshold
            row["gate_passed"] = False
            continue
        if row.get("context_scope") in {"long", "full_utterance"}:
            reference = best_local_bitrate
            reference_name = f"best_local:{best_local_name}"
        else:
            reference = unigram_bitrate
            reference_name = "unigram"
        relative = _relative_improvement(bitrate, reference, higher_is_better=False)
        row["relative_improvement_vs_local_or_unigram"] = relative
        row["gate_reference"] = reference_name
        row["gate_threshold"] = threshold
        row["gate_passed"] = bool(
            row.get("context_scope") in {"long", "full_utterance"}
            and relative is not None
            and relative >= threshold
        )

    return rows, analytic_summary, trained_summaries, warnings


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_directory(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def _go_no_go(rows: list[dict], threshold: float) -> dict:
    diagnostic_gates = [
        row for row in rows if row.get("stage") == "phase2_diagnostics" and row.get("gate_passed") is True
    ]
    long_prior_rows = [
        row
        for row in rows
        if row.get("stage") == "phase3_code_prior"
        and row.get("context_scope") in {"long", "full_utterance"}
        and row.get("estimated_entropy_bitrate_kbps") is not None
    ]
    best_long_prior = min(
        long_prior_rows,
        key=lambda row: float(row["estimated_entropy_bitrate_kbps"]),
        default=None,
    )
    prior_gate_passed = bool(best_long_prior and best_long_prior.get("gate_passed"))
    return {
        "gate_threshold": threshold,
        "diagnostics_gate_passed": bool(diagnostic_gates),
        "diagnostics_passed_rows": len(diagnostic_gates),
        "best_long_prior_family": best_long_prior.get("prior_family") if best_long_prior else None,
        "best_long_prior_bitrate_kbps": best_long_prior.get("estimated_entropy_bitrate_kbps") if best_long_prior else None,
        "prior_gate_passed": prior_gate_passed,
        "go_to_codec_context_training": bool(diagnostic_gates and prior_gate_passed),
    }


def collect_context_results(
    *,
    export_dir: Path,
    output_dir: Path,
    diagnostics_summary_path: Path,
    analytic_prior_summary_path: Path,
    trained_prior_summary_paths: list[Path],
    gate_threshold: float,
) -> tuple[list[dict], dict]:
    diagnostics_rows, diagnostics_summary, diagnostics_warnings = _collect_diagnostics(
        diagnostics_summary_path,
        gate_threshold,
    )
    prior_rows, analytic_summary, trained_summaries, prior_warnings = _collect_priors(
        analytic_prior_summary_path,
        trained_prior_summary_paths,
        gate_threshold,
    )
    rows = [*diagnostics_rows, *prior_rows]
    ensure_directory(output_dir)
    write_jsonl(output_dir / "results.jsonl", rows)
    _write_csv(output_dir / "summary.csv", rows)
    summary = {
        "schema_version": CONTEXT_RESULTS_SCHEMA_VERSION,
        "export_dir": str(export_dir.resolve()),
        "results_path": str((output_dir / "results.jsonl").resolve()),
        "summary_csv_path": str((output_dir / "summary.csv").resolve()),
        "rows": len(rows),
        "diagnostics_rows": len(diagnostics_rows),
        "prior_rows": len(prior_rows),
        "diagnostics_summary_path": str(diagnostics_summary_path),
        "analytic_prior_summary_path": str(analytic_prior_summary_path),
        "trained_prior_summary_paths": [str(path) for path in trained_prior_summary_paths],
        "warnings": [*diagnostics_warnings, *prior_warnings],
        "go_no_go": _go_no_go(rows, gate_threshold),
        "source_schemas": {
            "diagnostics": diagnostics_summary.get("schema_version") if diagnostics_summary else None,
            "analytic_prior": analytic_summary.get("schema_version") if analytic_summary else None,
            "trained_priors": [summary.get("schema_version") for summary in trained_summaries],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    return rows, summary


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else export_dir / "context_results"
    diagnostics_summary_path = (
        args.diagnostics_summary.resolve()
        if args.diagnostics_summary is not None
        else export_dir / "diagnostics" / "summary.json"
    )
    analytic_prior_summary_path = (
        args.analytic_prior_summary.resolve()
        if args.analytic_prior_summary is not None
        else export_dir / "code_priors" / "summary.json"
    )
    prior_root = args.prior_root.resolve() if args.prior_root is not None else export_dir.parent / "priors"
    trained_prior_summary_paths = (
        [path.resolve() for path in args.trained_prior_summary]
        if args.trained_prior_summary
        else [
            prior_root / "local-tcn" / "summary.json",
            prior_root / "long-transformer" / "summary.json",
        ]
    )
    rows, _ = collect_context_results(
        export_dir=export_dir,
        output_dir=output_dir,
        diagnostics_summary_path=diagnostics_summary_path,
        analytic_prior_summary_path=analytic_prior_summary_path,
        trained_prior_summary_paths=trained_prior_summary_paths,
        gate_threshold=args.gate_threshold,
    )
    print(f"Wrote {len(rows)} context result rows to {output_dir}")


if __name__ == "__main__":
    main()
