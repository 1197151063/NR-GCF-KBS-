#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset transfer pilot on the converted NT-SSM LastFM and ML-1M data.
# It runs sequentially on one GPU and compares the same backbone with and
# without the proposed hard structure--momentum filter.  Defaults intentionally
# use one seed and clean/20% replacement noise; no per-dataset tuning is done.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

datasets="${DATASETS:-lastfm ml-1m}"
noise_ratios="${NOISE_RATIOS:-0 0.2}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v2.3_ntssm_datasets}"

train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
train_lr="${TRAIN_LR:-0.0005}"
train_init_weight="${TRAIN_INIT_WEIGHT:-0.01}"

# These are fixed transfer values selected previously on Amazon-Book.  Override
# them explicitly for a later dataset-specific sensitivity experiment.
fusion_weight="${RELIABILITY_STRUCTURE_WEIGHT:-0.60}"
modulation_weight="${REPRESENTATION_MODULATION_LAMBDA:-0.40}"

common_environment() {
  export DATASET="$1"
  export NOISE_MODE=degree_preserving_replace
  export REPLACEMENT_SELECTION=uniform
  export NOISE_RATIOS="$noise_ratios"
  export SEEDS="$seeds"
  export GPU_ID="$gpu_id"
  export OUTPUT_ROOT="$output_root"
  export TRAIN_EPOCHS="$train_epochs"
  export TRAIN_PATIENCE="$train_patience"
  export TRAIN_LR="$train_lr"
  export TRAIN_INIT_WEIGHT="$train_init_weight"
  export STOP_AFTER_FILTER=0
  export SUMMARY_ONLY=1
  export RUN_PILOT_ANALYSIS=0
  export KEEP_EDGE_LABELS="${KEEP_EDGE_LABELS:-0}"
  export KEEP_GENERATED_TRAIN="${KEEP_GENERATED_TRAIN:-0}"
  export STRUCTURAL_MODE=two_hop_minhash
  export TOPK="${TOPK:-10}"
  export CHUNK_SIZE="${CHUNK_SIZE:-8192}"
  export MIN_DEGREE="${MIN_DEGREE:-2}"
  export RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}"
  export RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}"
  export RELIABILITY_STRUCTURE_WEIGHT="$fusion_weight"
  export RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}"
  export RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}"
  export REPRESENTATION_MODULATION_MODE=blend_always
  export REPRESENTATION_MODULATION_RAMP_EPOCHS=0
  export REPRESENTATION_MODULATION_LAMBDA="$modulation_weight"
  export REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}"
  export DRY_RUN="${DRY_RUN:-0}"
}

echo "NR-GCF NT-SSM-dataset transfer pilot"
echo "  datasets:          $datasets"
echo "  noise ratios:      $noise_ratios"
echo "  seeds:             $seeds"
echo "  GPU:               $gpu_id"
echo "  fusion weight:     $fusion_weight"
echo "  modulation weight: $modulation_weight"
echo "  output:             $output_root"

for dataset in $datasets; do
  if [[ ! -f "$script_dir/../data/$dataset/conversion_metadata.json" ]]; then
    echo "Missing converted dataset metadata: data/$dataset/conversion_metadata.json" >&2
    exit 2
  fi
  common_environment "$dataset"

  echo "[$dataset] starting matched no-filter references"
  EDGE_FILTER_MODE=none \
  RELIABILITY_FILTER_SCHEDULE=fixed \
    bash "$script_dir/run_edge_diagnostics_grid.sh"

  echo "[$dataset] starting adaptive hard structure--momentum runs"
  EDGE_FILTER_MODE=hard_structure_momentum \
  RELIABILITY_FILTER_SCHEDULE=adaptive \
  RELIABILITY_ADAPTIVE_MIN_EPOCH="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-2}" \
  RELIABILITY_ADAPTIVE_MAX_EPOCH="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-4}" \
  RELIABILITY_ADAPTIVE_MIN_COVERAGE="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}" \
  RELIABILITY_ADAPTIVE_JACCARD="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}" \
  RELIABILITY_ADAPTIVE_STABLE_CHECKS="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-1}" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run completed; no training or files were generated."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/comparison_summary.json"

echo "NT-SSM-dataset transfer pilot completed: $output_root"
