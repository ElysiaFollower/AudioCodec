#!/usr/bin/env bash
# Build a lightweight, downloadable bundle from context-modeling pipeline outputs.

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

output_root="evals/outputs/context-modeling"
python_bin=${PYTHON_BIN:-python}
export_dir=""
results_dir=""
prior_root=""
bundle_root=""
bundle_dir=""
run_id=""
audio_pairs=3
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  scripts/pack-context-results.sh [options]

Common options:
  --output-root DIR                Pipeline output root, default: evals/outputs/context-modeling.
  --export-dir DIR                 Export dir, default: output-root/neural-4k-export.
  --results-dir DIR                Results dir, default: output-root/results.
  --prior-root DIR                 Trained prior root, default: output-root/priors.
  --bundle-root DIR                Bundle root, default: output-root/download-bundles.
  --bundle-dir DIR                 Exact bundle dir. Overrides --bundle-root and --run-id.
  --run-id ID                      Bundle subdir name when --bundle-dir is not set.
  --audio-pairs N                  Copy up to N source/reconstruction pairs, default: 3.
  --skip-audio-pairs               Equivalent to --audio-pairs 0.
  --dry-run                        Print planned copies without creating files.

The bundle uses a whitelist and intentionally excludes checkpoints, representation
tensors, bulk reconstructions, compressed audio, and other heavy artifacts.
EOF
}

require_value() {
  if [ -z "${2:-}" ]; then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-root)
      require_value "$1" "${2:-}"
      output_root=$2
      shift 2
      ;;
    --export-dir)
      require_value "$1" "${2:-}"
      export_dir=$2
      shift 2
      ;;
    --results-dir)
      require_value "$1" "${2:-}"
      results_dir=$2
      shift 2
      ;;
    --prior-root)
      require_value "$1" "${2:-}"
      prior_root=$2
      shift 2
      ;;
    --bundle-root)
      require_value "$1" "${2:-}"
      bundle_root=$2
      shift 2
      ;;
    --bundle-dir)
      require_value "$1" "${2:-}"
      bundle_dir=$2
      shift 2
      ;;
    --run-id)
      require_value "$1" "${2:-}"
      run_id=$2
      shift 2
      ;;
    --audio-pairs)
      require_value "$1" "${2:-}"
      audio_pairs=$2
      shift 2
      ;;
    --skip-audio-pairs)
      audio_pairs=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$audio_pairs" in
  ""|*[!0-9]*)
    printf 'Error: --audio-pairs must be a non-negative integer.\n' >&2
    exit 2
    ;;
esac

if [ -z "$export_dir" ]; then
  export_dir="$output_root/neural-4k-export"
fi
if [ -z "$results_dir" ]; then
  results_dir="$output_root/results"
fi
if [ -z "$prior_root" ]; then
  prior_root="$output_root/priors"
fi
if [ -z "$bundle_root" ]; then
  bundle_root="$output_root/download-bundles"
fi
if [ -z "$bundle_dir" ]; then
  if [ -z "$run_id" ]; then
    run_id=$(date -u +%Y%m%dT%H%M%SZ)
  fi
  bundle_dir="$bundle_root/$run_id"
fi

export_name=$(basename "$export_dir")
results_name=$(basename "$results_dir")
prior_name=$(basename "$prior_root")
manifest_file="$bundle_dir/BUNDLE_MANIFEST.txt"
copied=0
missing=0

init_manifest() {
  if [ "$dry_run" -eq 1 ]; then
    return
  fi
  mkdir -p "$bundle_dir"
  {
    printf 'context_results_bundle\n'
    printf 'created_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit: '
    git rev-parse --verify HEAD 2>/dev/null || printf 'unknown\n'
    printf 'output_root: %s\n' "$output_root"
    printf 'export_dir: %s\n' "$export_dir"
    printf 'results_dir: %s\n' "$results_dir"
    printf 'prior_root: %s\n' "$prior_root"
    printf 'audio_pairs: %s\n' "$audio_pairs"
    printf 'excludes: checkpoint.pt, *.pt tensors, bulk reconstructions, compressed codec outputs\n'
    printf '\n[copied]\n'
  } > "$manifest_file"
}

copy_if_exists() {
  src=$1
  dest_rel=$2
  dest="$bundle_dir/$dest_rel"
  if [ -f "$src" ]; then
    if [ "$dry_run" -eq 1 ]; then
      printf '[dry-run] copy %s -> %s\n' "$src" "$dest"
    else
      mkdir -p "$(dirname "$dest")"
      cp "$src" "$dest"
      printf '%s -> %s\n' "$src" "$dest_rel" >> "$manifest_file"
    fi
    copied=$((copied + 1))
  else
    if [ "$dry_run" -eq 1 ]; then
      printf '[dry-run] missing %s\n' "$src"
    else
      printf '%s\n' "$src" >> "$bundle_dir/.missing.tmp"
    fi
    missing=$((missing + 1))
  fi
}

copy_prior_files() {
  prior_dir=$1
  prior_dest=$2
  copy_if_exists "$prior_dir/summary.json" "$prior_dest/summary.json"
  copy_if_exists "$prior_dir/config.json" "$prior_dest/config.json"
  copy_if_exists "$prior_dir/train_metrics.jsonl" "$prior_dest/train_metrics.jsonl"
  copy_if_exists "$prior_dir/val_metrics.jsonl" "$prior_dest/val_metrics.jsonl"
}

copy_audio_pairs() {
  if [ "$audio_pairs" -eq 0 ]; then
    return
  fi
  manifest_path="$export_dir/manifest.jsonl"
  if [ ! -f "$manifest_path" ]; then
    if [ "$dry_run" -eq 1 ]; then
      printf '[dry-run] audio pairs skipped; missing %s\n' "$manifest_path"
    else
      printf 'audio pairs skipped; missing %s\n' "$manifest_path" >> "$bundle_dir/.missing.tmp"
    fi
    missing=$((missing + 1))
    return
  fi

  audio_plan=$(mktemp "${TMPDIR:-/tmp}/context-audio-pairs.XXXXXX")
  if ! "$python_bin" - "$manifest_path" "$audio_pairs" > "$audio_plan" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
limit = int(sys.argv[2])


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def safe_item_dir(raw_id: object, index: int) -> str:
    text = str(raw_id or f"item-{index:03d}")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not text:
        text = f"item-{index:03d}"
    return f"{index:03d}-{text[:80]}"


emitted = 0
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if emitted >= limit:
            break
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        source_value = row.get("source_path")
        reconstruction_value = row.get("reconstruction_path")
        if not source_value or not reconstruction_value:
            continue
        emitted += 1
        item_dir = safe_item_dir(row.get("id"), emitted)
        source_path = resolve_path(str(source_value))
        reconstruction_path = resolve_path(str(reconstruction_value))
        source_suffix = source_path.suffix or ".audio"
        reconstruction_suffix = reconstruction_path.suffix or ".wav"
        print(
            "\t".join(
                [
                    str(source_path),
                    f"audio_pairs/{item_dir}/source{source_suffix}",
                    str(reconstruction_path),
                    f"audio_pairs/{item_dir}/reconstruction{reconstruction_suffix}",
                ]
            )
        )
PY
  then
    rm -f "$audio_plan"
    printf 'Error: failed to parse audio pairs from %s\n' "$manifest_path" >&2
    exit 1
  fi

  while IFS=$'\t' read -r source_path source_dest reconstruction_path reconstruction_dest; do
    copy_if_exists "$source_path" "$source_dest"
    copy_if_exists "$reconstruction_path" "$reconstruction_dest"
  done < "$audio_plan"
  rm -f "$audio_plan"
}

init_manifest

copy_if_exists "$export_dir/manifest.jsonl" "$export_name/manifest.jsonl"
copy_if_exists "$export_dir/run.json" "$export_name/run.json"
copy_if_exists "$export_dir/diagnostics/summary.json" "$export_name/diagnostics/summary.json"
copy_if_exists "$export_dir/diagnostics/diagnostics.jsonl" "$export_name/diagnostics/diagnostics.jsonl"
copy_prior_files "$export_dir/code_priors" "$export_name/code_priors"

copy_if_exists "$results_dir/results.jsonl" "$results_name/results.jsonl"
copy_if_exists "$results_dir/summary.csv" "$results_name/summary.csv"
copy_if_exists "$results_dir/summary.json" "$results_name/summary.json"

copy_prior_files "$prior_root/local-tcn" "$prior_name/local-tcn"
copy_prior_files "$prior_root/long-transformer" "$prior_name/long-transformer"
copy_audio_pairs

if [ "$dry_run" -eq 1 ]; then
  printf '[dry-run] bundle dir: %s\n' "$bundle_dir"
  printf '[dry-run] copied: %s, missing: %s\n' "$copied" "$missing"
else
  if [ -f "$bundle_dir/.missing.tmp" ]; then
    {
      printf '\n[missing]\n'
      cat "$bundle_dir/.missing.tmp"
    } >> "$manifest_file"
    rm "$bundle_dir/.missing.tmp"
  fi
  printf '\nsummary: copied=%s missing=%s\n' "$copied" "$missing" >> "$manifest_file"
  printf 'Wrote lightweight context results bundle to %s (copied=%s, missing=%s)\n' \
    "$bundle_dir" "$copied" "$missing"
fi
