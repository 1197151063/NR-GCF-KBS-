#!/usr/bin/env bash
set -euo pipefail

# Sequential sensitivity study for the paper-style interpolation:
#
#   x_next = mu * CrossNorm(propagated_x) + (1 - mu) * propagated_x
#
# Run run_fusion_weight_sensitivity.sh first.  This script reads the selected
# structure--momentum fusion lambda for each noise ratio and keeps it fixed
# while sweeping the modulation weight mu in the explicit blend_always mode.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset="${DATASET:-yelp2018}"
noise_ratios="${NOISE_RATIOS:-0 0.2}"
modulation_weights="${MODULATION_WEIGHTS:-0.00 0.20 0.40 0.60 0.80 1.00}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
base_root="${HYPERPARAMETER_ROOT:-/root/autodl-tmp/outputs/hyperparameter_sensitivity}"
fusion_selection_json="${FUSION_SELECTION_JSON:-$base_root/fusion_weight/best_lambda_by_noise.json}"
output_root="${OUTPUT_ROOT:-$base_root/modulation_weight}"
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
if [[ ! -f "$fusion_selection_json" ]]; then
  echo "Fusion selection JSON not found: $fusion_selection_json" >&2
  echo "Run run_fusion_weight_sensitivity.sh first." >&2
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

python3 - "$noise_ratios" "$modulation_weights" <<'PY'
import math
import sys

for label, text in (
    ("noise ratio", sys.argv[1]),
    ("modulation weight", sys.argv[2]),
):
    for raw in text.split():
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit("Invalid %s: %s" % (label, raw))
PY

number_tag() {
  python3 - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(format(value, ".12g").replace("-", "m").replace(".", "p"))
PY
}

selected_fusion_weight() {
  python3 - "$fusion_selection_json" "$dataset" "$1" <<'PY'
import json
import math
import sys

path, dataset, raw_ratio = sys.argv[1:]
ratio = float(raw_ratio)
with open(path, encoding="utf-8") as stream:
    report = json.load(stream)
matches = [
    row for row in report.get("best_by_noise_ratio", [])
    if row.get("dataset") == dataset
    and math.isclose(float(row.get("requested_noise_ratio")), ratio,
                     rel_tol=0.0, abs_tol=1e-12)
]
if len(matches) != 1:
    raise SystemExit(
        "Expected one selected fusion lambda for dataset=%s ratio=%s; found %d"
        % (dataset, raw_ratio, len(matches))
    )
print(format(float(matches[0]["selected_lambda"]), ".12g"))
PY
}

total_runs=$((${#ratio_array[@]} * ${#modulation_array[@]} * ${#seed_array[@]}))
run_index=0

echo "NR-GCF propagation/CrossNorm modulation-weight sensitivity"
echo "  dataset:            $dataset"
echo "  noise ratios:       $noise_ratios"
echo "  modulation weights: $modulation_weights"
echo "  fusion selections:  $fusion_selection_json"
echo "  seeds:              $seeds"
echo "  GPU:                $gpu_id"
echo "  execution:          sequential"
echo "  total runs:         $total_runs"
echo "  output:             $output_root"

for ratio in "${ratio_array[@]}"; do
  rtag="$(number_tag "$ratio")"
  fusion_weight="$(selected_fusion_weight "$ratio")"
  echo "Noise $ratio uses selected structure--momentum lambda=$fusion_weight"
  for modulation_weight in "${modulation_array[@]}"; do
    mtag="$(number_tag "$modulation_weight")"
    for seed in "${seed_array[@]}"; do
      run_index=$((run_index + 1))
      if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid integer seed: $seed" >&2
        exit 2
      fi

      combo_root="${output_root%/}/noise_${rtag}/modulation_${mtag}"
      run_name="hard_structure_momentum_replace_noise_${rtag}_seed_${seed}_filter_adaptive_${adaptive_min_epoch}_${adaptive_max_epoch}_mod_blend_always"
      run_dir="$combo_root/$dataset/$run_name"

      if [[ "$dry_run" != "1" && "$skip_completed" == "1" && \
            -f "$run_dir/edge_reliability/training_summary.json" ]]; then
        echo "[$run_index/$total_runs] skip completed: noise=$ratio mu=$modulation_weight seed=$seed"
        continue
      fi
      if [[ "$dry_run" != "1" && -e "$run_dir" ]]; then
        echo "Existing incomplete run directory: $run_dir" >&2
        exit 1
      fi

      echo "[$run_index/$total_runs] start: noise=$ratio fusion=$fusion_weight mu=$modulation_weight seed=$seed"
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
      RELIABILITY_STRUCTURE_WEIGHT="$fusion_weight" \
      RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
      RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
      RELIABILITY_FILTER_SCHEDULE=adaptive \
      RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
      RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
      RELIABILITY_ADAPTIVE_MIN_COVERAGE="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}" \
      RELIABILITY_ADAPTIVE_JACCARD="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}" \
      RELIABILITY_ADAPTIVE_STABLE_CHECKS="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-1}" \
      REPRESENTATION_MODULATION_MODE=blend_always \
      REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
      REPRESENTATION_MODULATION_LAMBDA="$modulation_weight" \
      REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
      DRY_RUN="$dry_run" \
        bash "$script_dir/run_edge_diagnostics_grid.sh"
      echo "[$run_index/$total_runs] done: noise=$ratio mu=$modulation_weight seed=$seed"
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training or summary was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/modulation_grid_runs.json"

python3 "$script_dir/select_lambda_sensitivity.py" \
  --input "$output_root/modulation_grid_runs.json" \
  --output "$output_root/best_modulation_by_noise.json" \
  --markdown "$output_root/modulation_sensitivity_table.md" \
  --parameter modulation \
  --selection-metric "${SELECTION_METRIC:-best_recall_at_20}"

echo "Modulation sensitivity completed."
echo "  compact runs:     $output_root/modulation_grid_runs.json"
echo "  selected weights: $output_root/best_modulation_by_noise.json"
echo "  readable table:   $output_root/modulation_sensitivity_table.md"
