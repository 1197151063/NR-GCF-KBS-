#!/usr/bin/env bash
set -euo pipefail

# Compact LastFM confirmation suite:
#   A. fixed-epoch ranking ablation at noise 0/0.2,
#   B. one-seed 0--0.5 noise curve for the balanced fused ranking.
# The w=0.5 clean/0.2 runs from stage A are reused by the final analysis.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=lastfm
controlled_noise_ratios="${CONTROLLED_NOISE_RATIOS:-0 0.2}"
curve_extra_noise_ratios="${CURVE_EXTRA_NOISE_RATIOS:-0.1 0.3 0.4 0.5}"
all_noise_ratios="${ALL_NOISE_RATIOS:-0 0.1 0.2 0.3 0.4 0.5}"
ranking_weights="${RANKING_WEIGHTS:-0.00 0.50 0.95 1.00}"
selected_weight="${SELECTED_STRUCTURE_WEIGHT:-0.50}"
modulation_weight="${MODULATION_WEIGHT:-0.20}"
removal_cap="${REMOVAL_CAP:-0.04}"
filter_epoch="${FILTER_EPOCH:-10}"
seed="${SEED:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.2_lastfm_fixed_confirmation}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"
train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"

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
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$filter_epoch" =~ ^[0-9]+$ ]] || [[ "$filter_epoch" -lt 2 ]]; then
  echo "FILTER_EPOCH must be an integer >= 2." >&2
  exit 2
fi

python3 - "$controlled_noise_ratios" "$curve_extra_noise_ratios" \
  "$all_noise_ratios" "$ranking_weights" "$selected_weight" \
  "$modulation_weight" "$removal_cap" <<'PY'
import math
import sys

def values(label, text):
    result = []
    for raw in text.split():
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise SystemExit("%s must contain unique values" % label)
    return result

controlled = values("controlled noise ratio", sys.argv[1])
extra = values("curve extra noise ratio", sys.argv[2])
all_ratios = values("all noise ratio", sys.argv[3])
weights = values("ranking weight", sys.argv[4])
selected = values("selected structure weight", sys.argv[5])
values("modulation weight", sys.argv[6])
values("removal cap", sys.argv[7])
if len(selected) != 1 or selected[0] not in weights:
    raise SystemExit("SELECTED_STRUCTURE_WEIGHT must be one RANKING_WEIGHTS arm")
if set(controlled) & set(extra):
    raise SystemExit("Controlled and extra noise-ratio sets must be disjoint")
if sorted(controlled + extra) != sorted(all_ratios):
    raise SystemExit(
        "ALL_NOISE_RATIOS must equal CONTROLLED_NOISE_RATIOS plus "
        "CURVE_EXTRA_NOISE_RATIOS"
    )
PY

number_tag() {
  python3 - "$1" <<'PY'
import sys
print(format(float(sys.argv[1]), ".12g").replace("-", "m").replace(".", "p"))
PY
}

run_combo() {
  local label="$1"
  local combo_root="$2"
  local ratios="$3"
  local structure_weight="$4"
  local completed="$combo_root/comparison_summary.json"

  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed $label"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$combo_root" ]]; then
    echo "Existing incomplete directory for $label: $combo_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start $label"
  DATASET="$dataset" \
  NOISE_MODE=degree_preserving_replace \
  REPLACEMENT_SELECTION=uniform \
  NOISE_RATIOS="$ratios" \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$combo_root" \
  TRAIN_EPOCHS="$train_epochs" \
  TRAIN_PATIENCE="$train_patience" \
  TRAIN_LR="${TRAIN_LR:-0.0005}" \
  TRAIN_INIT_WEIGHT="${TRAIN_INIT_WEIGHT:-0.01}" \
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
  RELIABILITY_MAX_REMOVAL_RATIO="$removal_cap" \
  RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
  RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
  RELIABILITY_FILTER_SCHEDULE=fixed \
  RELIABILITY_FILTER_EPOCH="$filter_epoch" \
  REPRESENTATION_MODULATION_MODE=blend_always \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA="$modulation_weight" \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
  DRY_RUN="$dry_run" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"

  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$combo_root" --output "$completed"
  fi
  echo "Done $label"
}

controlled_count="$(wc -w <<<"$controlled_noise_ratios" | tr -d ' ')"
weight_count="$(wc -w <<<"$ranking_weights" | tr -d ' ')"
extra_count="$(wc -w <<<"$curve_extra_noise_ratios" | tr -d ' ')"
planned_runs=$((controlled_count * weight_count + extra_count))

echo "LastFM fixed-time confirmation"
echo "  controlled noise:  $controlled_noise_ratios"
echo "  extra curve noise: $curve_extra_noise_ratios"
echo "  ranking weights:   $ranking_weights"
echo "  selected weight:   $selected_weight"
echo "  modulation weight: $modulation_weight"
echo "  removal cap:       $removal_cap"
echo "  fixed filter epoch:$filter_epoch"
echo "  seed:              $seed"
echo "  output:            $output_root"
echo "  planned runs:      $planned_runs"

# A. Matched ranking ablation: the filter epoch is identical for every arm.
for structure_weight in $ranking_weights; do
  tag="$(number_tag "$structure_weight")"
  run_combo \
    "controlled ranking weight=$structure_weight" \
    "${output_root%/}/controlled/weight_${tag}" \
    "$controlled_noise_ratios" "$structure_weight"
done

# B. Extend only the selected fused arm; noise 0/0.2 are reused from stage A.
selected_tag="$(number_tag "$selected_weight")"
run_combo \
  "selected ranking extra noise curve" \
  "${output_root%/}/noise_curve_extra/weight_${selected_tag}" \
  "$curve_extra_noise_ratios" "$selected_weight"

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"

python3 "$script_dir/analyze_ranking_ablation.py" \
  --input "$output_root/all_runs.json" \
  --dataset "$dataset" \
  --removal-cap "$removal_cap" \
  --modulation-lambda "$modulation_weight" \
  --weights "$ranking_weights" \
  --noise-ratios "$controlled_noise_ratios" \
  --output "$output_root/fixed_ranking_ablation.json" \
  --markdown "$output_root/fixed_ranking_ablation.md"

python3 "$script_dir/analyze_noise_curve.py" \
  --input "$output_root/all_runs.json" \
  --dataset "$dataset" \
  --removal-cap "$removal_cap" \
  --modulation-lambda "$modulation_weight" \
  --structure-weight "$selected_weight" \
  --noise-ratios "$all_noise_ratios" \
  --seed "$seed" \
  --output "$output_root/noise_curve.json" \
  --markdown "$output_root/noise_curve.md"

echo "LastFM fixed-time confirmation completed: $output_root"
echo "  ranking table: $output_root/fixed_ranking_ablation.md"
echo "  noise curve:   $output_root/noise_curve.md"
echo "  all runs:      $output_root/all_runs.json"
