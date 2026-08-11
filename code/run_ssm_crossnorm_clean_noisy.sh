#!/usr/bin/env bash
set -euo pipefail

# Matched clean/noisy confirmation for the tuned in-batch SSM + CrossNorm.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.7_ssm_crossnorm_clean_noisy}"
gpu_id="${GPU_ID:-0}"
dry_run="${DRY_RUN:-0}"

if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi

echo "SSM + always-on CrossNorm clean/noisy confirmation"
echo "  noise mode/ratios: degree-preserving replacement / 0 0.2"
echo "  init:              Xavier Uniform, gain 1"
echo "  decay/lr/tau:      1e-5 / 5e-4 / 0.09"
echo "  message dropout:   0.1"
echo "  epochs/patience:   100 / 20"
echo "  seed/GPU:          2020 / $gpu_id"
echo "  output:            $output_root"

DATASET=yelp2018 \
NOISE_MODE=degree_preserving_replace \
REPLACEMENT_SELECTION=uniform \
NOISE_RATIOS="0 0.2" \
SEEDS=2020 \
GPU_ID="$gpu_id" \
OUTPUT_ROOT="$output_root" \
TRAIN_EPOCHS=100 \
TRAIN_PATIENCE=20 \
TRAIN_BATCH_SIZE=2048 \
TRAIN_LR=0.0005 \
TRAIN_INIT_METHOD=xavier_uniform \
TRAIN_INIT_WEIGHT=1.0 \
TRAIN_DECAY=0.00001 \
TRAINING_OBJECTIVE=ssm \
SSM_TAU=0.09 \
OBJECTIVE_MESSAGE_DROPOUT=0.1 \
EDGE_FILTER_MODE=none \
REPRESENTATION_MODULATION_MODE=original_always \
REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
REPRESENTATION_MODULATION_LAMBDA=1.0 \
STOP_AFTER_FILTER=0 \
SUMMARY_ONLY=1 \
RUN_PILOT_ANALYSIS=0 \
KEEP_EDGE_LABELS=0 \
KEEP_GENERATED_TRAIN=0 \
REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
DRY_RUN="$dry_run" \
  bash "$script_dir/run_edge_diagnostics_grid.sh"

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"
python3 "$script_dir/analyze_ssm_clean_noisy.py" \
  --input "$output_root/all_runs.json" \
  --noise-ratios "0 0.2" \
  --seed 2020 \
  --output "$output_root/clean_noisy_comparison.json" \
  --markdown "$output_root/clean_noisy_comparison.md"

echo "Completed: $output_root"
echo "Table:     $output_root/clean_noisy_comparison.md"
