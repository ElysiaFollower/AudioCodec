#!/usr/bin/env bash
# Run the self-contained fixed-frame long-range redundancy pipeline:
# build manifest -> train/reuse baseline codec -> export representations ->
# frozen diagnostics -> analytic code priors -> trained code priors -> result bundle.

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python}
manifest=""
checkpoint=""
config="configs/ablation-adversarial-msstft-balanced-4kbps.json"
codec_label="neural-4k"
output_root="evals/outputs/context-modeling"
export_dir=""
device="auto"
dataset_root=""
manifest_split="test"
manifest_limit=""
skip_manifest_build=0
codec_output_dir=""
codec_steps=""
codec_device="auto"
codec_smoke_test=0
codec_limit_train_examples=""
codec_resume_from=""
skip_codec_training=0
force_codec_training=0
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
run_collect_results=1
run_pack_results=1
bundle_root=""
bundle_dir=""
bundle_run_id=""
audio_pairs=""
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run-context-prior-pipeline.sh [options]

Default behavior is self-contained for this branch: build a manifest from the
4kbps config, train a fixed-frame baseline codec when needed, export
representations, run diagnostics/priors, collect results, then pack a
lightweight downloadable bundle.

Common options:
  --config PATH                    Codec config, default: configs/ablation-adversarial-msstft-balanced-4kbps.json.
  --dataset-root PATH              Override dataset root for codec training and manifest building.
  --manifest PATH                  Existing input manifest. If omitted, build output-root/manifests/SPLIT.jsonl.
  --manifest-split SPLIT           train/val/test split for generated manifest, default: test.
  --manifest-limit N               Limit generated manifest rows.
  --checkpoint PATH                Existing codec checkpoint. If omitted, use codec-output-dir/checkpoints/best.pt.
  --codec-label LABEL              Codec label, default: neural-4k.
  --output-root DIR                Root output dir, default: evals/outputs/context-modeling.
  --export-dir DIR                 Override export dir, default: output-root/codec-label-export.
  --device DEVICE                  Export device, default: auto.
  --context-scope SCOPE            none/local/medium/long/full_utterance, default: full_utterance.
  --clip-scope SCOPE               Optional clip scope metadata.
  --not-full-utterance             Do not pass --is-full-utterance to export.
  --max-items N                    Limit diagnostics and prior item count.

Codec training options:
  --codec-output-dir DIR           Baseline codec output dir, default: output-root/codec-baseline.
  --codec-steps N                  Override codec training steps. If omitted, use config main_steps.
  --codec-device DEVICE            Codec training device, default: auto.
  --codec-smoke-test               Run codec training in smoke-test mode.
  --limit-train-examples N         Limit codec training examples.
  --resume-codec-from PATH         Resume codec training from checkpoint.
  --skip-codec-training            Require checkpoint to already exist; do not train codec.
  --force-codec-training           Run codec training even if the default checkpoint already exists.
  --skip-manifest-build            Require manifest to already exist; do not build it.

Training prior options:
  --train-steps N                  Default: 1000.
  --train-batch-size N             Default: 8.
  --train-sequence-length N        Default: 512.
  --train-eval-every N             Default: 100.
  --train-device DEVICE            Default: auto.

Result bundle options:
  --bundle-root DIR                Bundle root, default: output-root/download-bundles.
  --bundle-dir DIR                 Exact bundle dir for lightweight downloadable results.
  --bundle-run-id ID               Bundle subdir name when --bundle-dir is not set.
  --audio-pairs N                  Copy up to N source/reconstruction pairs into bundle, default: 3.

Stage switches:
  --skip-export
  --skip-diagnostics
  --skip-analytic-prior
  --skip-trained-priors
  --skip-collect-results
  --skip-pack-results
  --skip-audio-pairs
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
    --manifest-split)
      require_value "$1" "${2:-}"
      manifest_split=$2
      shift 2
      ;;
    --manifest-limit)
      require_value "$1" "${2:-}"
      manifest_limit=$2
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
    --dataset-root)
      require_value "$1" "${2:-}"
      dataset_root=$2
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
    --codec-output-dir)
      require_value "$1" "${2:-}"
      codec_output_dir=$2
      shift 2
      ;;
    --codec-steps)
      require_value "$1" "${2:-}"
      codec_steps=$2
      shift 2
      ;;
    --codec-device)
      require_value "$1" "${2:-}"
      codec_device=$2
      shift 2
      ;;
    --codec-smoke-test)
      codec_smoke_test=1
      shift
      ;;
    --limit-train-examples)
      require_value "$1" "${2:-}"
      codec_limit_train_examples=$2
      shift 2
      ;;
    --resume-codec-from)
      require_value "$1" "${2:-}"
      codec_resume_from=$2
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
    --bundle-run-id)
      require_value "$1" "${2:-}"
      bundle_run_id=$2
      shift 2
      ;;
    --audio-pairs)
      require_value "$1" "${2:-}"
      audio_pairs=$2
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
    --skip-collect-results)
      run_collect_results=0
      shift
      ;;
    --skip-pack-results)
      run_pack_results=0
      shift
      ;;
    --skip-audio-pairs)
      audio_pairs=0
      shift
      ;;
    --skip-manifest-build)
      skip_manifest_build=1
      shift
      ;;
    --skip-codec-training)
      skip_codec_training=1
      shift
      ;;
    --force-codec-training)
      force_codec_training=1
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
if [ -z "$codec_output_dir" ]; then
  codec_output_dir="$output_root/codec-baseline"
fi

run_manifest_build=0
run_codec_training=0

case "$manifest_split" in
  train|val|test) ;;
  *)
    printf 'Error: --manifest-split must be one of train, val, test.\n' >&2
    exit 2
    ;;
esac

if [ "$run_export" -eq 1 ]; then
  if [ -z "$manifest" ]; then
    manifest="$output_root/manifests/${manifest_split}.jsonl"
    if [ "$skip_manifest_build" -eq 0 ]; then
      run_manifest_build=1
    fi
  fi
  if [ -z "$checkpoint" ]; then
    checkpoint="$codec_output_dir/checkpoints/best.pt"
    if [ "$skip_codec_training" -eq 0 ]; then
      run_codec_training=1
    fi
  fi
else
  if [ -z "$manifest" ]; then
    manifest="$export_dir/manifest.jsonl"
  fi
fi

export PYTHONPATH="${PYTHONPATH:-src}"

is_placeholder_path() {
  case "$1" in
    /path/to/*|path/to/*|"/path/to"*|"")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

require_existing_file() {
  label=$1
  path=$2
  if is_placeholder_path "$path"; then
    printf 'Error: %s is still a placeholder path: %s\n' "$label" "$path" >&2
    exit 2
  fi
  if [ ! -f "$path" ]; then
    printf 'Error: %s does not exist: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

if [ "$dry_run" -eq 0 ]; then
  require_existing_file "--config" "$config"
  if [ "$run_export" -eq 1 ]; then
    if [ "$run_manifest_build" -eq 0 ]; then
      require_existing_file "--manifest" "$manifest"
    fi
    if [ "$run_codec_training" -eq 0 ]; then
      require_existing_file "--checkpoint" "$checkpoint"
    fi
    if [ -n "$dataset_root" ] && [ ! -d "$dataset_root" ]; then
      printf 'Error: --dataset-root does not exist: %s\n' "$dataset_root" >&2
      exit 2
    fi
    if [ -n "$codec_resume_from" ]; then
      require_existing_file "--resume-codec-from" "$codec_resume_from"
    fi
  fi
fi

manifest_cmd=(
  "$python_bin" "evals/scripts/build_manifest.py"
  --config "$config"
  --split "$manifest_split"
  --output "$manifest"
)
if [ -n "$dataset_root" ]; then
  manifest_cmd+=(--dataset-root "$dataset_root")
fi
if [ -n "$manifest_limit" ]; then
  manifest_cmd+=(--limit "$manifest_limit")
fi

codec_train_cmd=(
  "$python_bin" "scripts/train_codec.py"
  --config "$config"
  --output-dir "$codec_output_dir"
  --device "$codec_device"
)
if [ -n "$dataset_root" ]; then
  codec_train_cmd+=(--dataset-root "$dataset_root")
fi
if [ -n "$codec_steps" ]; then
  codec_train_cmd+=(--steps "$codec_steps")
fi
if [ "$codec_smoke_test" -eq 1 ]; then
  codec_train_cmd+=(--smoke-test)
fi
if [ -n "$codec_limit_train_examples" ]; then
  codec_train_cmd+=(--limit-train-examples "$codec_limit_train_examples")
fi
if [ -n "$codec_resume_from" ]; then
  codec_train_cmd+=(--resume-from "$codec_resume_from")
fi

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
collect_results_cmd=(
  "$python_bin" "evals/scripts/collect_context_results.py"
  --export-dir "$export_dir"
  --output-dir "$output_root/results"
  --prior-root "$output_root/priors"
)
pack_results_cmd=(
  "scripts/pack-context-results.sh"
  --output-root "$output_root"
  --export-dir "$export_dir"
  --results-dir "$output_root/results"
  --prior-root "$output_root/priors"
)
if [ -n "$bundle_root" ]; then
  pack_results_cmd+=(--bundle-root "$bundle_root")
fi
if [ -n "$bundle_dir" ]; then
  pack_results_cmd+=(--bundle-dir "$bundle_dir")
fi
if [ -n "$bundle_run_id" ]; then
  pack_results_cmd+=(--run-id "$bundle_run_id")
fi
if [ -n "$audio_pairs" ]; then
  pack_results_cmd+=(--audio-pairs "$audio_pairs")
fi
if [ -n "$max_items" ]; then
  local_tcn_cmd+=(--max-train-items "$max_items" --max-eval-items "$max_items")
  long_transformer_cmd+=(--max-train-items "$max_items" --max-eval-items "$max_items")
fi

if [ "$run_manifest_build" -eq 1 ]; then
  run_cmd "${manifest_cmd[@]}"
fi
if [ "$run_codec_training" -eq 1 ]; then
  if [ "$dry_run" -eq 0 ] && [ "$force_codec_training" -eq 0 ] && [ -f "$checkpoint" ]; then
    printf 'Using existing self-contained codec checkpoint: %s\n' "$checkpoint"
  else
    run_cmd "${codec_train_cmd[@]}"
  fi
  if [ "$dry_run" -eq 0 ]; then
    require_existing_file "codec training output checkpoint" "$checkpoint"
  fi
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
if [ "$run_collect_results" -eq 1 ]; then
  run_cmd "${collect_results_cmd[@]}"
fi
if [ "$run_pack_results" -eq 1 ]; then
  run_cmd "${pack_results_cmd[@]}"
fi
