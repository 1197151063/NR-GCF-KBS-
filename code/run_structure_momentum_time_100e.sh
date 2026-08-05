#!/usr/bin/env bash
set -euo pipefail

# Three-run filtering-time pilot for the structure-dominant hard method.
# The default experiment uses 20% uniform degree-preserving replacement,
# seed 2026, and filtering epochs 10/15/20.  Every run has 100 total epochs.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-yelp2018}"
export NOISE_MODE="degree_preserving_replace"
export REPLACEMENT_SELECTION="uniform"
export NOISE_RATIOS="${NOISE_RATIOS:-0.2}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/nrgcf_structure_momentum_time_100e}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_PATIENCE="${TRAIN_PATIENCE:-1000}"
export STOP_AFTER_FILTER=0
export SUMMARY_ONLY=1
export RUN_PILOT_ANALYSIS=0
export KEEP_EDGE_LABELS="${KEEP_EDGE_LABELS:-0}"
export KEEP_GENERATED_TRAIN="${KEEP_GENERATED_TRAIN:-0}"
export STRUCTURAL_MODE="two_hop_minhash"
export TOPK="${TOPK:-10}"
export CHUNK_SIZE="${CHUNK_SIZE:-8192}"
export MIN_DEGREE="${MIN_DEGREE:-2}"
export RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}"
export RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}"
export RELIABILITY_STRUCTURE_WEIGHT="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
export RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}"
export RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}"

filter_epochs="${FILTER_EPOCHS:-10 15 20}"
for filter_epoch in $filter_epochs; do
  echo "Starting hard_structure_momentum filter_epoch=$filter_epoch"
  EDGE_FILTER_MODE=hard_structure_momentum \
  RELIABILITY_FILTER_EPOCH="$filter_epoch" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/comparison_summary.json"
fi

echo "Structure-momentum time pilot completed: $OUTPUT_ROOT"
