#!/usr/bin/env bash
# Build a lightweight, downloadable bundle from context-modeling pipeline outputs.

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

output_root="evals/outputs/context-modeling"
export_dir=""
results_dir=""
prior_root=""
bundle_root=""
bundle_dir=""
run_id=""
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
  --dry-run                        Print planned copies without creating files.

The bundle uses a whitelist and intentionally excludes checkpoints, representation
tensors, reconstructions, compressed audio, and other heavy artifacts.
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
    printf 'excludes: checkpoint.pt, *.pt tensors, wav/audio reconstructions, compressed codec outputs\n'
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
