#!/usr/bin/env bash
set -euo pipefail

# Six compact comparison runs:
#   modes  = none, hard_consensus, soft_reliability
#   noise  = 0.0, 0.2 degree-preserving uniform replacement
#   seed   = 2026 (override SEEDS only when a second seed is actually needed)
#
# No per-edge diagnostics table is written.  Each run keeps training.log,
# noise provenance, run_manifest.txt, and two compact edge_reliability JSONs.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-yelp2018}"
export NOISE_MODE="degree_preserving_replace"
export REPLACEMENT_SELECTION="uniform"
export NOISE_RATIOS="${NOISE_RATIOS:-0.0 0.2}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/nrgcf_reliability_100e}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
# A value greater than TRAIN_EPOCHS ensures each run reaches epoch 100.
export TRAIN_PATIENCE="${TRAIN_PATIENCE:-20}"
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

for mode in none hard_consensus soft_reliability; do
  echo "Starting mode=$mode ratios=$NOISE_RATIOS seeds=$SEEDS"
  EDGE_FILTER_MODE="$mode" bash "$script_dir/run_edge_diagnostics_grid.sh"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/comparison_summary.json"
fi

echo "Reliability comparison completed: $OUTPUT_ROOT"
