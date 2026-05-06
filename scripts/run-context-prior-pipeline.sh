#!/usr/bin/env bash
# Run the fixed-frame long-range redundancy pipeline:
# export representations -> frozen diagnostics -> analytic code priors -> trained code priors.

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python}
manifest=""
checkpoint=""
config=""
codec_label="neural-4k"
output_root="evals/outputs/context-modeling"
export_dir=""
device="auto"
context_scope="full_utterance"
clip_scope=""
is_full_utterance=1
max_items=""
train_steps=1000
train_batch_size=8
train_sequence_length=512
train_eval_every=100
train_device="auto"
run_export=1
run_diagnostics=1
run_analytic_prior=1
run_trained_priors=1
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run-context-prior-pipeline.sh --manifest PATH --checkpoint PATH [options]

Required unless --skip-export is used:
  --manifest PATH                  Input benchmark manifest.
  --checkpoint PATH                Neural codec checkpoint for representation export.

Common options:
  --config PATH                    Optional codec config override for export.
  --codec-label LABEL              Codec label, default: neural-4k.
  --output-root DIR                Root output dir, default: evals/outputs/context-modeling.
  --export-dir DIR                 Override export dir, default: output-root/codec-label-export.
  --device DEVICE                  Export device, default: auto.
  --context-scope SCOPE            none/local/medium/long/full_utterance, default: full_utterance.
  --clip-scope SCOPE               Optional clip scope metadata.
  --not-full-utterance             Do not pass --is-full-utterance to export.
  --max-items N                    Limit diagnostics and prior item count.

Training prior options:
  --train-steps N                  Default: 1000.
  --train-batch-size N             Default: 8.
  --train-sequence-length N        Default: 512.
  --train-eval-every N             Default: 100.
  --train-device DEVICE            Default: auto.

Stage switches:
  --skip-export
  --skip-diagnostics
  --skip-analytic-prior
  --skip-trained-priors
  --dry-run                        Print commands without running them.

Environment:
  PYTHON_BIN                       Python executable, default: python.
EOF
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  if [ "$dry_run" -eq 1 ]; then
    quote_cmd "$@"
  else
    "$@"
  fi
}

require_value() {
  if [ -z "${2:-}" ]; then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest)
      require_value "$1" "${2:-}"
      manifest=$2
      shift 2
      ;;
    --checkpoint)
      require_value "$1" "${2:-}"
      checkpoint=$2
      shift 2
      ;;
    --config)
      require_value "$1" "${2:-}"
      config=$2
      shift 2
      ;;
    --codec-label)
      require_value "$1" "${2:-}"
      codec_label=$2
      shift 2
      ;;
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
    --device)
      require_value "$1" "${2:-}"
      device=$2
      shift 2
      ;;
    --context-scope)
      require_value "$1" "${2:-}"
      context_scope=$2
      shift 2
      ;;
    --clip-scope)
      require_value "$1" "${2:-}"
      clip_scope=$2
      shift 2
      ;;
    --not-full-utterance)
      is_full_utterance=0
      shift
      ;;
    --max-items)
      require_value "$1" "${2:-}"
      max_items=$2
      shift 2
      ;;
    --train-steps)
      require_value "$1" "${2:-}"
      train_steps=$2
      shift 2
      ;;
    --train-batch-size)
      require_value "$1" "${2:-}"
      train_batch_size=$2
      shift 2
      ;;
    --train-sequence-length)
      require_value "$1" "${2:-}"
      train_sequence_length=$2
      shift 2
      ;;
    --train-eval-every)
      require_value "$1" "${2:-}"
      train_eval_every=$2
      shift 2
      ;;
    --train-device)
      require_value "$1" "${2:-}"
      train_device=$2
      shift 2
      ;;
    --skip-export)
      run_export=0
      shift
      ;;
    --skip-diagnostics)
      run_diagnostics=0
      shift
      ;;
    --skip-analytic-prior)
      run_analytic_prior=0
      shift
      ;;
    --skip-trained-priors)
      run_trained_priors=0
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

if [ -z "$export_dir" ]; then
  export_dir="$output_root/${codec_label}-export"
fi

if [ "$run_export" -eq 1 ]; then
  if [ -z "$manifest" ] || [ -z "$checkpoint" ]; then
    printf 'Error: --manifest and --checkpoint are required unless --skip-export is used.\n' >&2
    usage >&2
    exit 2
  fi
else
  if [ -z "$manifest" ]; then
    manifest="$export_dir/manifest.jsonl"
  fi
fi

export PYTHONPATH="${PYTHONPATH:-src}"

export_cmd=(
  "$python_bin" "evals/scripts/export_neural_codec.py"
  --manifest "$manifest"
  --checkpoint "$checkpoint"
  --output-dir "$export_dir"
  --codec-label "$codec_label"
  --device "$device"
  --save-representations
  --context-scope "$context_scope"
)
if [ -n "$config" ]; then
  export_cmd+=(--config "$config")
fi
if [ -n "$clip_scope" ]; then
  export_cmd+=(--clip-scope "$clip_scope")
fi
if [ "$is_full_utterance" -eq 1 ]; then
  export_cmd+=(--is-full-utterance)
fi

diagnostics_cmd=(
  "$python_bin" "evals/scripts/diagnose_representations.py"
  --export-dir "$export_dir"
  --representations latent quantized codes
)
if [ -n "$max_items" ]; then
  diagnostics_cmd+=(--max-items "$max_items")
fi

analytic_prior_cmd=(
  "$python_bin" "evals/scripts/evaluate_code_priors.py"
  --export-dir "$export_dir"
  --priors unigram previous_frame
)
if [ -n "$max_items" ]; then
  analytic_prior_cmd+=(--max-train-items "$max_items" --max-eval-items "$max_items")
fi

local_tcn_cmd=(
  "$python_bin" "evals/scripts/train_code_prior.py"
  --export-dir "$export_dir"
  --prior local_tcn
  --output-dir "$output_root/priors/local-tcn"
  --steps "$train_steps"
  --batch-size "$train_batch_size"
  --sequence-length "$train_sequence_length"
  --eval-every "$train_eval_every"
  --device "$train_device"
)
long_transformer_cmd=(
  "$python_bin" "evals/scripts/train_code_prior.py"
  --export-dir "$export_dir"
  --prior long_transformer
  --output-dir "$output_root/priors/long-transformer"
  --steps "$train_steps"
  --batch-size "$train_batch_size"
  --sequence-length "$train_sequence_length"
  --eval-every "$train_eval_every"
  --device "$train_device"
)
if [ -n "$max_items" ]; then
  local_tcn_cmd+=(--max-train-items "$max_items" --max-eval-items "$max_items")
  long_transformer_cmd+=(--max-train-items "$max_items" --max-eval-items "$max_items")
fi

if [ "$run_export" -eq 1 ]; then
  run_cmd "${export_cmd[@]}"
fi
if [ "$run_diagnostics" -eq 1 ]; then
  run_cmd "${diagnostics_cmd[@]}"
fi
if [ "$run_analytic_prior" -eq 1 ]; then
  run_cmd "${analytic_prior_cmd[@]}"
fi
if [ "$run_trained_priors" -eq 1 ]; then
  run_cmd "${local_tcn_cmd[@]}"
  run_cmd "${long_transformer_cmd[@]}"
fi
