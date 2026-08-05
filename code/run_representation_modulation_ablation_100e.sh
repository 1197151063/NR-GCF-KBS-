#!/usr/bin/env bash
set -euo pipefail

# Focused three-arm test for the connection between the validated T=20 hard
# structure-momentum filter and representation modulation.  The default is a
# single Yelp2018 20%-replacement run per arm; override NOISE_RATIOS only after
# this focused pilot establishes a winner.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-yelp2018}"
export NOISE_MODE="degree_preserving_replace"
export REPLACEMENT_SELECTION="uniform"
export NOISE_RATIOS="${NOISE_RATIOS:-0.2}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/nrgcf_representation_modulation_100e}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_PATIENCE="${TRAIN_PATIENCE:-1000}"
export STOP_AFTER_FILTER=0
export SUMMARY_ONLY=1
export RUN_PILOT_ANALYSIS=0
export KEEP_EDGE_LABELS="${KEEP_EDGE_LABELS:-0}"
export KEEP_GENERATED_TRAIN="${KEEP_GENERATED_TRAIN:-0}"
export EDGE_FILTER_MODE=hard_structure_momentum
export RELIABILITY_FILTER_EPOCH="${RELIABILITY_FILTER_EPOCH:-20}"
export RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}"
export RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}"
export RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}"
export RELIABILITY_STRUCTURE_WEIGHT="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
export RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}"
export REPRESENTATION_MODULATION_LAMBDA="${REPRESENTATION_MODULATION_LAMBDA:-1}"

modulation_modes="${MODULATION_MODES:-none paper_stage_two reliability_weighted_stage_two}"
for modulation_mode in $modulation_modes; do
  if [[ "$modulation_mode" == "none" || "$modulation_mode" == "legacy_always" ]]; then
    export REPRESENTATION_MODULATION_RAMP_EPOCHS=0
  else
    # The same transition is used for both stage-two variants so the only
    # difference between them is the reliability-weighted scale estimator.
    export REPRESENTATION_MODULATION_RAMP_EPOCHS="${STAGE_TWO_RAMP_EPOCHS:-5}"
  fi
  export REPRESENTATION_MODULATION_MODE="$modulation_mode"
  echo "Starting representation modulation arm: $modulation_mode"
  bash "$script_dir/run_edge_diagnostics_grid.sh"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/comparison_summary.json"
fi

echo "Representation modulation ablation completed: $OUTPUT_ROOT"
