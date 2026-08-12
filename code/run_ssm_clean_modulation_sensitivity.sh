#!/usr/bin/env bash
set -euo pipefail

# Clean Yelp2018 sensitivity for
# H_next=(1-mu)*propagated_H+mu*CrossNorm(propagated_H).
# Every arm uses the same blend_always implementation, including mu=0 and mu=1.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v4.0_ssm_clean_modulation}"
weights="${MODULATION_WEIGHTS:-0.0 0.2 0.4 0.6 0.8 1.0}"
seeds="${SEEDS:-2020}"
gpu_id="${GPU_ID:-0}"
learning_rate="${TRAIN_LR:-0.0001}"
temperature="${SSM_TAU:-0.14}"
decay="${TRAIN_DECAY:-0.0001}"
message_dropout="${OBJECTIVE_MESSAGE_DROPOUT:-0.1}"
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
if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer." >&2
  exit 2
fi

read -r -a weight_values <<<"$weights"
read -r -a seed_values <<<"$seeds"
if (( ${#weight_values[@]} == 0 || ${#seed_values[@]} == 0 )); then
  echo "MODULATION_WEIGHTS and SEEDS cannot be empty." >&2
  exit 2
fi
python3 - "$weights" <<'PY'
import math
import sys
for raw in sys.argv[1].split():
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit("Invalid modulation weight: %s" % raw)
PY

token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_index=0
total_runs=$(( ${#weight_values[@]} * ${#seed_values[@]} ))
echo "Yelp2018 clean SSM modulation-weight sensitivity"
echo "  modulation weights: $weights"
echo "  seeds:              $seeds"
echo "  SSM lr/tau/decay:   $learning_rate/$temperature/$decay"
echo "  dropout/batch:      $message_dropout/$batch_size"
echo "  epochs/patience:    $train_epochs/$train_patience"
echo "  edge filtering:     none"
echo "  modulation mode:    blend_always"
echo "  planned runs:       $total_runs"
echo "  output:             $output_root"

for weight in "${weight_values[@]}"; do
  weight_tag="$(token "$weight")"
  for seed in "${seed_values[@]}"; do
    run_index=$((run_index + 1))
    if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
      echo "Invalid integer seed: $seed" >&2
      exit 2
    fi
    case_root="${output_root%/}/mu_${weight_tag}/seed_${seed}"
    completed="$case_root/comparison_summary.json"
    if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
      echo "[$run_index/$total_runs] skip completed: mu=$weight seed=$seed"
      continue
    fi
    if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
      echo "Existing incomplete directory: $case_root" >&2
      echo "Move it aside or choose a new OUTPUT_ROOT." >&2
      exit 1
    fi

    echo "[$run_index/$total_runs] start: mu=$weight seed=$seed"
    DATASET=yelp2018 \
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
    REPRESENTATION_MODULATION_MODE=blend_always \
    REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
    REPRESENTATION_MODULATION_LAMBDA="$weight" \
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
    echo "[$run_index/$total_runs] done: mu=$weight seed=$seed"
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"

analysis_command=(
  python3 "$script_dir/analyze_ssm_clean_modulation_sensitivity.py"
  --input "$output_root/all_runs.json"
  --dataset yelp2018
  --weights
)
analysis_command+=("${weight_values[@]}")
analysis_command+=(--seeds)
analysis_command+=("${seed_values[@]}")
analysis_command+=(
  --learning-rate "$learning_rate"
  --temperature "$temperature"
  --decay "$decay"
  --message-dropout "$message_dropout"
  --batch-size "$batch_size"
  --output "$output_root/ssm_clean_modulation_sensitivity.json"
  --markdown "$output_root/ssm_clean_modulation_sensitivity.md"
)
"${analysis_command[@]}"

echo "SSM clean modulation sensitivity completed: $output_root"
echo "  table: $output_root/ssm_clean_modulation_sensitivity.md"
echo "  JSON:  $output_root/ssm_clean_modulation_sensitivity.json"
echo "  runs:  $output_root/all_runs.json"
