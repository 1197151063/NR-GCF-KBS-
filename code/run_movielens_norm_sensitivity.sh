#!/usr/bin/env bash
set -euo pipefail

# End-to-end MovieLens CrossNorm-strength sensitivity under the current best
# 1% hard-filter cap.  Only mu changes within each noise-ratio block:
#
#   x_next = mu * CrossNorm(propagated_x) + (1 - mu) * propagated_x
#
# mu=0 is ordinary propagation and mu=1 is the released direct CrossNorm
# operator.  Because CrossNorm is active from epoch one, mu legitimately also
# affects the warm-up momentum signal and the subsequently selected edge set.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

noise_ratios="${NOISE_RATIOS:-0 0.2}"
modulation_weights="${MODULATION_WEIGHTS:-0.00 0.20 0.40 0.60 0.80 1.00}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v2.9_ml_norm}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
train_lr="${TRAIN_LR:-0.0005}"
train_init_weight="${TRAIN_INIT_WEIGHT:-0.01}"
max_removal_ratio="${RELIABILITY_MAX_REMOVAL_RATIO:-0.01}"
structure_weight="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-5}"
adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-10}"
adaptive_stable_checks="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-2}"

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
read -r -a modulation_array <<<"$modulation_weights"
read -r -a seed_array <<<"$seeds"
if [[ ${#ratio_array[@]} -eq 0 || ${#modulation_array[@]} -eq 0 || \
      ${#seed_array[@]} -eq 0 ]]; then
  echo "NOISE_RATIOS, MODULATION_WEIGHTS, and SEEDS cannot be empty." >&2
  exit 2
fi

python3 - "$noise_ratios" "$modulation_weights" \
  "$max_removal_ratio" "$structure_weight" <<'PY'
import math
import sys

for label, text, upper in (
    ("noise ratio", sys.argv[1], None),
    ("modulation weight", sys.argv[2], 1.0),
    ("max removal ratio", sys.argv[3], 1.0),
    ("structure weight", sys.argv[4], 1.0),
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

number_tag() {
  python3 - "$1" <<'PY'
import sys
print(format(float(sys.argv[1]), ".12g").replace("-", "m").replace(".", "p"))
PY
}

total_runs=$((${#ratio_array[@]} * ${#modulation_array[@]} * ${#seed_array[@]}))
run_index=0

echo "MovieLens CrossNorm-strength sensitivity"
echo "  noise ratios:       $noise_ratios"
echo "  modulation weights: $modulation_weights"
echo "  max removal ratio:  $max_removal_ratio"
echo "  structure weight:   $structure_weight"
echo "  adaptive window:    ${adaptive_min_epoch}-${adaptive_max_epoch}"
echo "  stable checks:      $adaptive_stable_checks"
echo "  seeds:              $seeds"
echo "  GPU:                $gpu_id"
echo "  total runs:         $total_runs"
echo "  output:             $output_root"
echo "  endpoint semantics: mu=0 ordinary propagation; mu=1 direct CrossNorm"

for ratio in "${ratio_array[@]}"; do
  ratio_tag="$(number_tag "$ratio")"
  for modulation_weight in "${modulation_array[@]}"; do
    modulation_tag="$(number_tag "$modulation_weight")"
    for seed in "${seed_array[@]}"; do
      run_index=$((run_index + 1))
      if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid integer seed: $seed" >&2
        exit 2
      fi

      combo_root="${output_root%/}/noise_${ratio_tag}/norm_${modulation_tag}"
      run_name="hard_structure_momentum_replace_noise_${ratio_tag}_seed_${seed}_filter_adaptive_${adaptive_min_epoch}_${adaptive_max_epoch}_mod_blend_always"
      run_dir="$combo_root/ml-1m/$run_name"
      completed="$run_dir/edge_reliability/training_summary.json"

      if [[ "$dry_run" != "1" && "$skip_completed" == "1" && \
            -f "$completed" ]]; then
        echo "[$run_index/$total_runs] skip completed: noise=$ratio mu=$modulation_weight"
        continue
      fi
      if [[ "$dry_run" != "1" && -e "$run_dir" ]]; then
        echo "Existing incomplete run directory: $run_dir" >&2
        echo "Move it aside or choose a new OUTPUT_ROOT." >&2
        exit 1
      fi

      echo "[$run_index/$total_runs] start: noise=$ratio mu=$modulation_weight seed=$seed"
      DATASET=ml-1m \
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
      RELIABILITY_STRUCTURE_WEIGHT="$structure_weight" \
      RELIABILITY_MAX_REMOVAL_RATIO="$max_removal_ratio" \
      RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
      RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
      RELIABILITY_FILTER_SCHEDULE=adaptive \
      RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
      RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
      RELIABILITY_ADAPTIVE_MIN_COVERAGE="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}" \
      RELIABILITY_ADAPTIVE_JACCARD="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}" \
      RELIABILITY_ADAPTIVE_STABLE_CHECKS="$adaptive_stable_checks" \
      REPRESENTATION_MODULATION_MODE=blend_always \
      REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
      REPRESENTATION_MODULATION_LAMBDA="$modulation_weight" \
      REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
      NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
      DRY_RUN="$dry_run" \
        bash "$script_dir/run_edge_diagnostics_grid.sh"
      echo "[$run_index/$total_runs] done: noise=$ratio mu=$modulation_weight"
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training or files were generated."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/norm_grid_runs.json"

python3 "$script_dir/select_lambda_sensitivity.py" \
  --input "$output_root/norm_grid_runs.json" \
  --output "$output_root/best_norm_by_noise.json" \
  --markdown "$output_root/norm_sensitivity_table.md" \
  --parameter modulation \
  --selection-metric best_recall_at_20

echo "MovieLens norm sensitivity completed: $output_root"
echo "  compact runs: $output_root/norm_grid_runs.json"
echo "  selections:   $output_root/best_norm_by_noise.json"
echo "  table:        $output_root/norm_sensitivity_table.md"
