#!/usr/bin/env bash
set -euo pipefail

# Focused MovieLens pilot.  It changes only the maximum fraction removed by
# hard_structure_momentum while keeping the scoring rule fixed.  Run this
# before any fusion-weight sweep.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

noise_ratios="${NOISE_RATIOS:-0 0.2}"
removal_caps="${REMOVAL_CAPS:-0.01 0.02 0.03}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/movielens_removal_budget}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"
run_baseline="${RUN_BASELINE:-1}"

train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
structure_weight="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-5}"
adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-10}"
adaptive_stable_checks="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-2}"

for flag in dry_run skip_completed run_baseline; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done

python3 - "$noise_ratios" "$removal_caps" "$structure_weight" <<'PY'
import math
import sys

for label, text, upper in (
    ("noise ratio", sys.argv[1], None),
    ("removal cap", sys.argv[2], 1.0),
    ("structure weight", sys.argv[3], 1.0),
):
    for raw in text.split():
        value = float(raw)
        if not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
            raise SystemExit("Invalid %s: %s" % (label, raw))
PY

tag() {
  python3 - "$1" <<'PY'
import sys
print(format(float(sys.argv[1]), ".12g").replace("-", "m").replace(".", "p"))
PY
}

run_grid() {
  local mode="$1"
  local ratios="$2"
  local cap="$3"
  local root="$4"
  DATASET=ml-1m \
  NOISE_MODE=degree_preserving_replace \
  REPLACEMENT_SELECTION=uniform \
  NOISE_RATIOS="$ratios" \
  SEEDS="$seeds" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$root" \
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
  EDGE_FILTER_MODE="$mode" \
  RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}" \
  RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}" \
  RELIABILITY_STRUCTURE_WEIGHT="$structure_weight" \
  RELIABILITY_MAX_REMOVAL_RATIO="$cap" \
  RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
  RELIABILITY_FILTER_SCHEDULE="$([[ "$mode" == "hard_structure_momentum" ]] && echo adaptive || echo fixed)" \
  RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
  RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
  RELIABILITY_ADAPTIVE_MIN_COVERAGE="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}" \
  RELIABILITY_ADAPTIVE_JACCARD="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}" \
  RELIABILITY_ADAPTIVE_STABLE_CHECKS="$adaptive_stable_checks" \
  REPRESENTATION_MODULATION_MODE=blend_always \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA="${REPRESENTATION_MODULATION_LAMBDA:-0.40}" \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  DRY_RUN="$dry_run" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"
}

echo "MovieLens focused removal-budget sensitivity"
echo "  noise ratios:       $noise_ratios"
echo "  removal caps:       $removal_caps"
echo "  structure weight:   $structure_weight"
echo "  adaptive window:    ${adaptive_min_epoch}-${adaptive_max_epoch}"
echo "  stable checks:      $adaptive_stable_checks"
echo "  seed(s):            $seeds"
echo "  GPU:                $gpu_id"
echo "  output:             $output_root"

if [[ "$run_baseline" == "1" ]]; then
  baseline_root="${output_root%/}/baseline"
  baseline_done="$baseline_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$baseline_done" ]]; then
    echo "Skip completed no-filter baselines."
  else
    if [[ "$dry_run" != "1" && -e "$baseline_root" ]]; then
      echo "Existing incomplete baseline directory: $baseline_root" >&2
      exit 1
    fi
    run_grid none "$noise_ratios" 1.0 "$baseline_root"
    if [[ "$dry_run" != "1" ]]; then
      python3 "$script_dir/summarize_reliability_runs.py" \
        --root "$baseline_root" --output "$baseline_done"
    fi
  fi
fi

for cap in $removal_caps; do
  cap_tag="$(tag "$cap")"
  combo_root="${output_root%/}/cap_${cap_tag}"
  combo_done="$combo_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$combo_done" ]]; then
    echo "Skip completed cap=$cap."
    continue
  fi
  if [[ "$dry_run" != "1" && -e "$combo_root" ]]; then
    echo "Existing incomplete cap directory: $combo_root" >&2
    exit 1
  fi
  echo "Start cap=$cap"
  run_grid hard_structure_momentum "$noise_ratios" "$cap" "$combo_root"
  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$combo_root" --output "$combo_done"
  fi
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/comparison_summary.json"
python3 "$script_dir/analyze_movielens_removal_budget.py" \
  --input "$output_root/comparison_summary.json" \
  --output "$output_root/budget_selection.json" \
  --markdown "$output_root/budget_table.md" \
  --clean-tolerance "${CLEAN_RECALL_TOLERANCE:-0.002}"

echo "MovieLens removal-budget pilot completed: $output_root"
