#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/yelp_embedding_magnitude_10e}"
seed="${SEED:-2026}"
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" ]] && ! python3 -c 'import matplotlib' 2>/dev/null; then
  echo "matplotlib is required; install it in the active experiment environment." >&2
  exit 2
fi

run_arm() {
  local name="$1"
  local mode="$2"
  DATASET=yelp2018 \
  NOISE_MODE=degree_preserving_replace \
  NOISE_RATIOS=0 \
  SEEDS="$seed" \
  GPU_ID="${GPU_ID:-0}" \
  OUTPUT_ROOT="$output_root/$name" \
  TRAIN_EPOCHS=10 \
  TRAIN_PATIENCE=20 \
  TRAIN_BATCH_SIZE=2048 \
  TRAIN_LR=0.0005 \
  TRAIN_INIT_METHOD=normal \
  TRAIN_INIT_WEIGHT=0.01 \
  TRAIN_DECAY=0.0001 \
  TRAINING_OBJECTIVE=bpr \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE="$mode" \
  REPRESENTATION_MODULATION_LAMBDA=0.4 \
  STOP_AFTER_FILTER=0 \
  SUMMARY_ONLY=1 \
  RUN_PILOT_ANALYSIS=0 \
  KEEP_EDGE_LABELS=0 \
  KEEP_GENERATED_TRAIN=0 \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  DRY_RUN="${DRY_RUN:-0}" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"
}

run_arm lightgcn none
run_arm norm blend_always

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

find_summary() {
  local root="$1"
  local count
  count="$(find "$root" -name training_summary.json -type f | wc -l | tr -d ' ')"
  if [[ "$count" -ne 1 ]]; then
    echo "Expected one training summary under $root; found $count." >&2
    exit 1
  fi
  find "$root" -name training_summary.json -type f -print -quit
}

lightgcn_summary="$(find_summary "$output_root/lightgcn")"
norm_summary="$(find_summary "$output_root/norm")"

python3 "$script_dir/plot_embedding_magnitudes.py" \
  --lightgcn "$lightgcn_summary" \
  --norm "$norm_summary" \
  --output-dir "$output_root/figure"
