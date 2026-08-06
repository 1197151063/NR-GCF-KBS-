#!/usr/bin/env bash
set -euo pipefail

# Focused timing pilot after outputs_v1.6.  It runs one no-filter reference
# and two early adaptive filtering arms on the same deterministic 20%
# degree-preserving replacement split.  Per-edge CSV/Parquet export is off.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-yelp2018}"
export NOISE_MODE="degree_preserving_replace"
export REPLACEMENT_SELECTION="uniform"
export NOISE_RATIOS="${NOISE_RATIOS:-0.2}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v1.7}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_PATIENCE="${TRAIN_PATIENCE:-20}"
export TRAIN_LR="${TRAIN_LR:-0.0005}"
export TRAIN_INIT_WEIGHT="${TRAIN_INIT_WEIGHT:-0.01}"
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
export REPRESENTATION_MODULATION_RAMP_EPOCHS=0
export REPRESENTATION_MODULATION_LAMBDA=1

adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-2}"
adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-4}"
adaptive_min_coverage="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}"
adaptive_jaccard="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}"
adaptive_stable_checks="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-1}"

echo "Starting no-filter reference: original_always"
EDGE_FILTER_MODE=none \
RELIABILITY_FILTER_SCHEDULE=fixed \
REPRESENTATION_MODULATION_MODE=original_always \
  bash "$script_dir/run_edge_diagnostics_grid.sh"

for modulation_mode in original_always reliability_weighted_always; do
  echo "Starting early adaptive filtering arm: $modulation_mode"
  EDGE_FILTER_MODE=hard_structure_momentum \
  RELIABILITY_FILTER_SCHEDULE=adaptive \
  RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
  RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
  RELIABILITY_ADAPTIVE_MIN_COVERAGE="$adaptive_min_coverage" \
  RELIABILITY_ADAPTIVE_JACCARD="$adaptive_jaccard" \
  RELIABILITY_ADAPTIVE_STABLE_CHECKS="$adaptive_stable_checks" \
  REPRESENTATION_MODULATION_MODE="$modulation_mode" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/comparison_summary.json"
fi

echo "Early adaptive filtering pilot completed: $OUTPUT_ROOT"
