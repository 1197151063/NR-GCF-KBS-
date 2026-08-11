#!/usr/bin/env bash
set -euo pipefail

# Focused 2 x 2 x 2 diagnostic for the failed reference-SSM/CrossNorm run.
# Only initialization, all-layer L2 decay, and message dropout vary.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.6_ssm_crossnorm_factorial}"
gpu_id="${GPU_ID:-0}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done

run_case() {
  local init_method="$1"
  local decay="$2"
  local dropout="$3"
  local init_weight
  if [[ "$init_method" == "normal" ]]; then
    init_weight=0.01
  else
    init_weight=1.0
  fi
  local case_name="init_${init_method}_decay_${decay}_dropout_${dropout}"
  local case_root="${output_root%/}/${case_name}"
  local completed="$case_root/comparison_summary.json"

  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed $case_name"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start $case_name"
  DATASET=yelp2018 \
  NOISE_MODE=degree_preserving_replace \
  NOISE_RATIOS=0 \
  SEEDS=2020 \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$case_root" \
  TRAIN_EPOCHS=10 \
  TRAIN_PATIENCE=10 \
  TRAIN_BATCH_SIZE=2048 \
  TRAIN_LR=0.001 \
  TRAIN_INIT_METHOD="$init_method" \
  TRAIN_INIT_WEIGHT="$init_weight" \
  TRAIN_DECAY="$decay" \
  TRAINING_OBJECTIVE=ssm \
  SSM_TAU=0.1 \
  OBJECTIVE_MESSAGE_DROPOUT="$dropout" \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE=original_always \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA=1.0 \
  STOP_AFTER_FILTER=0 \
  SUMMARY_ONLY=1 \
  RUN_PILOT_ANALYSIS=0 \
  KEEP_EDGE_LABELS=0 \
  KEEP_GENERATED_TRAIN=0 \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
  DRY_RUN="$dry_run" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"

  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$case_root" --output "$completed"
  fi
  echo "Done $case_name"
}

echo "SSM + always-on CrossNorm focused factorial"
echo "  initialization: Xavier Uniform(gain=1), Normal(std=0.01)"
echo "  decay:          0.0001, 0.1"
echo "  dropout:        0, 0.1"
echo "  fixed:          Yelp2018, seed=2020, epochs=10, lr=0.001"
echo "  output:         $output_root"
echo "  planned runs:   8"

for init_method in xavier_uniform normal; do
  for decay in 0.0001 0.1; do
    for dropout in 0 0.1; do
      run_case "$init_method" "$decay" "$dropout"
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"
python3 "$script_dir/analyze_ssm_crossnorm_factorial.py" \
  --root "$output_root" \
  --output "$output_root/factorial_analysis.json" \
  --markdown "$output_root/factorial_analysis.md"

echo "Completed: $output_root"
echo "Table:     $output_root/factorial_analysis.md"
