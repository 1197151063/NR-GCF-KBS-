#!/usr/bin/env bash
set -euo pipefail

# Two-stage Yelp2018 SSM/CrossNorm grid:
# 1. rank every lr x tau x decay combination on clean data;
# 2. validate the top clean configurations at 0.2 replacement noise.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.8_ssm_yelp_grid}"
gpu_id="${GPU_ID:-0}"
seed="${SEED:-2020}"
learning_rates="${LEARNING_RATES:-0.00001 0.0001 0.0005 0.001}"
temperatures="${TEMPERATURES:-0.07 0.09 0.10 0.12}"
decays="${DECAYS:-0.000001 0.00001 0.0001}"
top_k="${TOP_K:-6}"
phase="${PHASE:-all}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ "$phase" != "all" && "$phase" != "clean" && "$phase" != "noisy" ]]; then
  echo "PHASE must be all, clean, or noisy." >&2
  exit 2
fi
if ! [[ "$gpu_id" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ && "$top_k" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_ID and SEED must be non-negative integers; TOP_K must be positive." >&2
  exit 2
fi

read -r -a lr_values <<<"$learning_rates"
read -r -a tau_values <<<"$temperatures"
read -r -a decay_values <<<"$decays"
combination_count=$(( ${#lr_values[@]} * ${#tau_values[@]} * ${#decay_values[@]} ))
if (( top_k > combination_count )); then
  echo "TOP_K cannot exceed the number of clean combinations." >&2
  exit 2
fi

token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_case() {
  local ratio="$1"
  local lr="$2"
  local tau="$3"
  local decay="$4"
  local stage="$5"
  local case_name="lr_$(token "$lr")_tau_$(token "$tau")_decay_$(token "$decay")"
  local case_root="${output_root%/}/${stage}/${case_name}"
  local completed="$case_root/comparison_summary.json"

  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed stage=$stage $case_name"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start stage=$stage ratio=$ratio lr=$lr tau=$tau decay=$decay"
  DATASET=yelp2018 \
  NOISE_MODE=degree_preserving_replace \
  NOISE_RATIOS="$ratio" \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$case_root" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}" \
  TRAIN_PATIENCE="${TRAIN_PATIENCE:-20}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}" \
  TRAIN_LR="$lr" \
  TRAIN_INIT_METHOD=xavier_uniform \
  TRAIN_INIT_WEIGHT=1.0 \
  TRAIN_DECAY="$decay" \
  TRAINING_OBJECTIVE=ssm \
  SSM_TAU="$tau" \
  OBJECTIVE_MESSAGE_DROPOUT="${OBJECTIVE_MESSAGE_DROPOUT:-0.1}" \
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
  echo "Done stage=$stage $case_name"
}

analyze_grid() {
  python3 "$script_dir/analyze_ssm_yelp_hyperparameter_grid.py" \
    --root "$output_root" \
    --learning-rates "$learning_rates" \
    --temperatures "$temperatures" \
    --decays "$decays" \
    --seed "$seed" \
    --top-k "$top_k" \
    --selected "$output_root/selected_combinations.tsv" \
    --output "$output_root/hyperparameter_grid.json" \
    --markdown "$output_root/hyperparameter_grid.md"
}

echo "Yelp2018 SSM + always-on CrossNorm hyperparameter grid"
echo "  learning rates:     $learning_rates"
echo "  temperatures:       $temperatures"
echo "  decays:             $decays"
echo "  clean combinations: $combination_count"
echo "  noisy top-k:        $top_k"
echo "  maximum runs:       $((combination_count + top_k))"
echo "  epochs/patience:    ${TRAIN_EPOCHS:-100}/${TRAIN_PATIENCE:-20}"
echo "  seed/GPU:           $seed/$gpu_id"
echo "  phase:              $phase"
echo "  output:             $output_root"

if [[ "$phase" == "all" || "$phase" == "clean" ]]; then
  for lr in "${lr_values[@]}"; do
    for tau in "${tau_values[@]}"; do
      for decay in "${decay_values[@]}"; do
        run_case 0 "$lr" "$tau" "$decay" clean_grid
      done
    done
  done
fi

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run planned $combination_count clean runs and $top_k noisy validations."
  exit 0
fi

analyze_grid
if [[ "$phase" == "clean" ]]; then
  echo "Clean phase completed. Selected combinations: $output_root/selected_combinations.tsv"
  exit 0
fi

while IFS=$'\t' read -r lr tau decay; do
  [[ -n "$lr" ]] || continue
  run_case 0.2 "$lr" "$tau" "$decay" noisy_validation
done < "$output_root/selected_combinations.tsv"

analyze_grid
echo "Completed: $output_root"
echo "Ranking:   $output_root/hyperparameter_grid.md"
echo "JSON:      $output_root/hyperparameter_grid.json"
