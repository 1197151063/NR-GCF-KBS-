#!/usr/bin/env bash
set -euo pipefail

# Focused clean LastFM SSM grid on the ordinary LightGCN backbone.
# Only learning rate and temperature vary; filtering and CrossNorm are off.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v4.5_lastfm_ssm_lr_tau}"
learning_rates="${LEARNING_RATES:-0.0001 0.0005 0.001}"
temperatures="${TEMPERATURES:-0.05 0.1 0.2 0.5}"
seed="${SEED:-2026}"
gpu_id="${GPU_ID:-0}"
decay="${TRAIN_DECAY:-0.0001}"
message_dropout="${OBJECTIVE_MESSAGE_DROPOUT:-0.0}"
batch_size="${TRAIN_BATCH_SIZE:-2048}"
train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done
if ! [[ "$gpu_id" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID and SEED must be non-negative integers." >&2
  exit 2
fi
if [[ ! -f "$script_dir/../data/lastfm/train.txt" || ! -f "$script_dir/../data/lastfm/test.txt" ]]; then
  echo "Missing converted LastFM train.txt or test.txt." >&2
  exit 2
fi
read -r -a lr_values <<<"$learning_rates"
read -r -a tau_values <<<"$temperatures"
python3 - "$learning_rates" "$temperatures" <<'PY'
import math
import sys
for label, raw_values in (("learning rate", sys.argv[1]), ("temperature", sys.argv[2])):
    values = [float(raw) for raw in raw_values.split()]
    if not values or len(values) != len(set(values)):
        raise SystemExit("%s values must be non-empty and unique" % label)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise SystemExit("%s values must be finite and positive" % label)
PY

token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_case() {
  local learning_rate="$1"
  local temperature="$2"
  local case_name="lr_$(token "$learning_rate")_tau_$(token "$temperature")"
  local case_root="${output_root%/}/$case_name"
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed lr=$learning_rate tau=$temperature"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start lr=$learning_rate tau=$temperature"
  DATASET=lastfm \
  NOISE_MODE=degree_preserving_replace \
  NOISE_RATIOS=0 \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$case_root" \
  TRAIN_EPOCHS="$train_epochs" \
  TRAIN_PATIENCE="$train_patience" \
  TRAIN_BATCH_SIZE="$batch_size" \
  TRAIN_LR="$learning_rate" \
  TRAIN_INIT_METHOD=xavier_uniform \
  TRAIN_INIT_WEIGHT=1.0 \
  TRAIN_DECAY="$decay" \
  TRAINING_OBJECTIVE=ssm \
  SSM_TAU="$temperature" \
  OBJECTIVE_MESSAGE_DROPOUT="$message_dropout" \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE=none \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA=0 \
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
  echo "Done lr=$learning_rate tau=$temperature"
}

combination_count=$((${#lr_values[@]} * ${#tau_values[@]}))
echo "LastFM clean SSM LightGCN lr/tau grid"
echo "  learning rates:    $learning_rates"
echo "  temperatures:      $temperatures"
echo "  fixed decay:       $decay"
echo "  dropout/batch:     $message_dropout/$batch_size"
echo "  epochs/patience:   $train_epochs/$train_patience"
echo "  filtering:         none"
echo "  modulation:        none (ordinary LightGCN)"
echo "  seed/GPU:          $seed/$gpu_id"
echo "  planned runs:      $combination_count"
echo "  output:            $output_root"

for learning_rate in "${lr_values[@]}"; do
  for temperature in "${tau_values[@]}"; do
    run_case "$learning_rate" "$temperature"
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"

analysis_command=(
  python3 "$script_dir/analyze_lastfm_ssm_lr_tau_grid.py"
  --input "$output_root/all_runs.json"
  --learning-rates
)
analysis_command+=("${lr_values[@]}")
analysis_command+=(--temperatures)
analysis_command+=("${tau_values[@]}")
analysis_command+=(
  --seed "$seed"
  --decay "$decay"
  --message-dropout "$message_dropout"
  --batch-size "$batch_size"
  --output "$output_root/lastfm_ssm_lr_tau_grid.json"
  --markdown "$output_root/lastfm_ssm_lr_tau_grid.md"
)
"${analysis_command[@]}"

echo "LastFM SSM grid completed: $output_root"
echo "  table: $output_root/lastfm_ssm_lr_tau_grid.md"
echo "  JSON:  $output_root/lastfm_ssm_lr_tau_grid.json"
echo "  runs:  $output_root/all_runs.json"
