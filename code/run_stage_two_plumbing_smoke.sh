#!/usr/bin/env bash
set -euo pipefail

# Focused verification of the corrected stage-two graph data flow. It uses the
# primary uniform degree-preserving replacement protocol and runs exactly two
# training epochs after the epoch-15 filtering transition.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-yelp2018}"
export NOISE_MODE=degree_preserving_replace
export REPLACEMENT_SELECTION=uniform
export NOISE_RATIOS="${NOISE_RATIOS:-0.10}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/stage2_plumbing_smoke}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-17}"
export STOP_AFTER_FILTER=0
export DIAGNOSTICS_FORMAT="${DIAGNOSTICS_FORMAT:-parquet}"
export STRUCTURAL_MODE="${STRUCTURAL_MODE:-two_hop_minhash}"
export CHUNK_SIZE="${CHUNK_SIZE:-32768}"
export RUN_PILOT_ANALYSIS="${RUN_PILOT_ANALYSIS:-1}"

exec bash "$script_dir/run_edge_diagnostics_grid.sh"
