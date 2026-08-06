#!/usr/bin/env bash
set -euo pipefail

# Sequential sensitivity study for the active structure--momentum fusion:
#
#   risk(e) = lambda * (1 - structure_rank(e))
#             + (1 - lambda) * momentum_rank(e)
#
# Experimental matrix (default):
#   noise ratio: 0, 0.2
#   lambda:      0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0
#   seed:        2026
#
# This is deliberately sequential.  GPU_ID selects one GPU, and every run is
# completed before the next configuration starts.  Here lambda maps to
# RELIABILITY_STRUCTURE_WEIGHT.  The legacy --lambda_ argument is ignored by
# original_always direct CrossNorm and remains fixed at 1.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset="${DATASET:-yelp2018}"
noise_ratios="${NOISE_RATIOS:-0 0.2}"
lambda_values="${LAMBDA_VALUES:-0.00 0.20 0.40 0.60 0.80 0.90 0.95 1.00}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/hyperparameter_sensitivity/fusion_weight}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
train_lr="${TRAIN_LR:-0.0005}"
train_init_weight="${TRAIN_INIT_WEIGHT:-0.01}"
adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-2}"
adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-4}"

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

read -r -a ratio_array <<<"$noise_ratios"
read -r -a lambda_array <<<"$lambda_values"
read -r -a seed_array <<<"$seeds"
if [[ ${#ratio_array[@]} -eq 0 || ${#lambda_array[@]} -eq 0 || \
      ${#seed_array[@]} -eq 0 ]]; then
  echo "NOISE_RATIOS, LAMBDA_VALUES, and SEEDS cannot be empty." >&2
  exit 2
fi

python3 - "$noise_ratios" "$lambda_values" <<'PY'
import math
import sys

for label, text, upper in (
    ("noise ratio", sys.argv[1], None),
    ("lambda", sys.argv[2], 1.0),
):
    for raw in text.split():
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if not math.isfinite(value) or value < 0:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if upper is not None and value > upper:
            raise SystemExit("Invalid %s: %s" % (label, raw))
PY

ratio_tag() {
  python3 - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(format(value, ".12g").replace("-", "m").replace(".", "p"))
PY
}

lambda_tag() {
  ratio_tag "$1"
}

total_runs=$((${#ratio_array[@]} * ${#lambda_array[@]} * ${#seed_array[@]}))
run_index=0

echo "NR-GCF structure--momentum fusion-weight sensitivity"
echo "  dataset:          $dataset"
echo "  noise ratios:     $noise_ratios"
echo "  active lambdas:   $lambda_values"
echo "  seeds:            $seeds"
echo "  GPU:              $gpu_id"
echo "  execution:        sequential"
echo "  total runs:       $total_runs"
echo "  output:           $output_root"
echo "  selection metric: ${SELECTION_METRIC:-best_recall_at_20}"
echo "  lambda semantics: edge-reliability structure weight"
echo "  legacy --lambda_: fixed to 1 and ignored by original_always"

for ratio in "${ratio_array[@]}"; do
  rtag="$(ratio_tag "$ratio")"
  for active_lambda in "${lambda_array[@]}"; do
    ltag="$(lambda_tag "$active_lambda")"
    for seed in "${seed_array[@]}"; do
      run_index=$((run_index + 1))
      if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid integer seed: $seed" >&2
        exit 2
      fi

      combo_root="${output_root%/}/noise_${rtag}/lambda_${ltag}"
      run_name="hard_structure_momentum_replace_noise_${rtag}_seed_${seed}_filter_adaptive_${adaptive_min_epoch}_${adaptive_max_epoch}_mod_original_always"
      run_dir="$combo_root/$dataset/$run_name"

      if [[ "$dry_run" != "1" && "$skip_completed" == "1" && \
            -f "$run_dir/edge_reliability/training_summary.json" ]]; then
        echo "[$run_index/$total_runs] skip completed: noise=$ratio lambda=$active_lambda seed=$seed"
        continue
      fi
      if [[ "$dry_run" != "1" && -e "$run_dir" ]]; then
        echo "Existing incomplete run directory: $run_dir" >&2
        echo "Move that directory aside or use a new OUTPUT_ROOT." >&2
        exit 1
      fi

      echo "[$run_index/$total_runs] start: noise=$ratio lambda=$active_lambda seed=$seed"
      DATASET="$dataset" \
      NOISE_MODE=degree_preserving_replace \
      REPLACEMENT_SELECTION=uniform \
      NOISE_RATIOS="$ratio" \
      SEEDS="$seed" \
      GPU_ID="$gpu_id" \
      OUTPUT_ROOT="$combo_root" \
      TRAIN_EPOCHS="$train_epochs" \
      TRAIN_PATIENCE="$train_patience" \
      TRAIN_LR="$train_lr" \
      TRAIN_INIT_WEIGHT="$train_init_weight" \
      NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
      STOP_AFTER_FILTER=0 \
      SUMMARY_ONLY=1 \
      RUN_PILOT_ANALYSIS=0 \
      KEEP_EDGE_LABELS=0 \
      KEEP_GENERATED_TRAIN=0 \
      STRUCTURAL_MODE=two_hop_minhash \
      TOPK="${TOPK:-10}" \
      CHUNK_SIZE="${CHUNK_SIZE:-8192}" \
      MIN_DEGREE="${MIN_DEGREE:-2}" \
      EDGE_FILTER_MODE=hard_structure_momentum \
      RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}" \
      RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}" \
      RELIABILITY_STRUCTURE_WEIGHT="$active_lambda" \
      RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
      RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
      RELIABILITY_FILTER_SCHEDULE=adaptive \
      RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
      RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
      RELIABILITY_ADAPTIVE_MIN_COVERAGE="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}" \
      RELIABILITY_ADAPTIVE_JACCARD="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}" \
      RELIABILITY_ADAPTIVE_STABLE_CHECKS="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-1}" \
      REPRESENTATION_MODULATION_MODE=original_always \
      REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
      REPRESENTATION_MODULATION_LAMBDA=1 \
      REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
      DRY_RUN="$dry_run" \
        bash "$script_dir/run_edge_diagnostics_grid.sh"
      echo "[$run_index/$total_runs] done: noise=$ratio lambda=$active_lambda seed=$seed"
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training or summary was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/lambda_grid_runs.json"

python3 "$script_dir/select_lambda_sensitivity.py" \
  --input "$output_root/lambda_grid_runs.json" \
  --output "$output_root/best_lambda_by_noise.json" \
  --markdown "$output_root/lambda_sensitivity_table.md" \
  --parameter fusion \
  --selection-metric "${SELECTION_METRIC:-best_recall_at_20}"

echo "Lambda sensitivity completed."
echo "  compact runs:     $output_root/lambda_grid_runs.json"
echo "  selected lambdas: $output_root/best_lambda_by_noise.json"
echo "  readable table:   $output_root/lambda_sensitivity_table.md"
