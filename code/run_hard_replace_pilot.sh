#!/usr/bin/env bash
set -euo pipefail

# One focused pilot: 10% degree-preserving, structure-aware replacement noise.
# Run from the repository code/ directory. Override any value through the
# environment; no remote login or file-transfer command is executed here.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-yelp2018}"
export NOISE_MODE=degree_preserving_replace
export REPLACEMENT_SELECTION=hard_two_hop
export NOISE_RATIOS="${NOISE_RATIOS:-0.10}"
export SEEDS="${SEEDS:-2026}"
export GPU_ID="${GPU_ID:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
export DIAGNOSTICS_FORMAT="${DIAGNOSTICS_FORMAT:-parquet}"
export STRUCTURAL_MODE="${STRUCTURAL_MODE:-two_hop_minhash}"
export TOPK="${TOPK:-10}"
export CHUNK_SIZE="${CHUNK_SIZE:-32768}"
export HARD_CANDIDATE_POOL="${HARD_CANDIDATE_POOL:-8}"
export HARD_SUPPORT_LIMIT="${HARD_SUPPORT_LIMIT:-16}"
export STOP_AFTER_FILTER="${STOP_AFTER_FILTER:-1}"
export RUN_PILOT_ANALYSIS="${RUN_PILOT_ANALYSIS:-1}"

exec bash "$script_dir/run_edge_diagnostics_grid.sh"
