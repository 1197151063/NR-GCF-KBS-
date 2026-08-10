#!/usr/bin/env bash
set -euo pipefail

# Matched no-filter controls for the selected MovieLens CrossNorm blend.
# Compare these two runs with outputs_v2.9_ml_norm/noise_{0,0p2}/norm_0p2
# to isolate the net contribution of structure--momentum hard filtering.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET=ml-1m
export NOISE_MODE=degree_preserving_replace
export REPLACEMENT_SELECTION=uniform
export NOISE_RATIOS="${NOISE_RATIOS:-0 0.2}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.0_ml_norm_matched_baseline}"

export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_PATIENCE="${TRAIN_PATIENCE:-20}"
export TRAIN_LR="${TRAIN_LR:-0.0005}"
export TRAIN_INIT_WEIGHT="${TRAIN_INIT_WEIGHT:-0.01}"

export EDGE_FILTER_MODE=none
export RELIABILITY_FILTER_SCHEDULE=fixed
export REPRESENTATION_MODULATION_MODE=blend_always
export REPRESENTATION_MODULATION_RAMP_EPOCHS=0
export REPRESENTATION_MODULATION_LAMBDA="${REPRESENTATION_MODULATION_LAMBDA:-0.20}"

export STOP_AFTER_FILTER=0
export SUMMARY_ONLY=1
export RUN_PILOT_ANALYSIS=0
export KEEP_EDGE_LABELS=0
export KEEP_GENERATED_TRAIN=0
export STRUCTURAL_MODE=two_hop_minhash
export TOPK="${TOPK:-10}"
export CHUNK_SIZE="${CHUNK_SIZE:-8192}"
export MIN_DEGREE="${MIN_DEGREE:-2}"
export RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}"
export RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}"
export RELIABILITY_STRUCTURE_WEIGHT="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
export RELIABILITY_MAX_REMOVAL_RATIO=1.0
export RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}"
export REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}"
export NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}"
export DRY_RUN="${DRY_RUN:-0}"

if [[ "$REPRESENTATION_MODULATION_LAMBDA" != "0.20" && \
      "$REPRESENTATION_MODULATION_LAMBDA" != "0.2" ]]; then
  echo "Warning: this matched control was designed for mu=0.2; got $REPRESENTATION_MODULATION_LAMBDA" >&2
fi

echo "MovieLens matched no-filter CrossNorm control"
echo "  noise ratios:      $NOISE_RATIOS"
echo "  seeds:             $SEEDS"
echo "  modulation weight: $REPRESENTATION_MODULATION_LAMBDA"
echo "  edge filter:       none"
echo "  total runs:        2 per seed"
echo "  output:            $OUTPUT_ROOT"

bash "$script_dir/run_edge_diagnostics_grid.sh"

if [[ "$DRY_RUN" != "1" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/comparison_summary.json"
fi

echo "MovieLens matched no-filter control completed: $OUTPUT_ROOT"
