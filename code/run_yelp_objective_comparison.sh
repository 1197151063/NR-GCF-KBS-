#!/usr/bin/env bash
set -euo pipefail

# Yelp2018 clean-data backbone pilot under SSM and Alignment--Uniformity.
# Within each objective, compare ordinary LightGCN propagation against the
# released always-on direct CrossNorm. Edge filtering is disabled throughout.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset=yelp2018
objectives="${OBJECTIVES:-ssm au}"
modulation_modes="${MODULATION_MODES:-none original_always}"
seed="${SEED:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.3_yelp_objectives}"
train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

if [[ "$objectives" != "ssm au" ]]; then
  echo "This controlled pilot requires OBJECTIVES='ssm au'." >&2
  exit 2
fi
if [[ "$modulation_modes" != "none original_always" ]]; then
  echo "This controlled pilot requires MODULATION_MODES='none original_always'." >&2
  exit 2
fi
for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done
if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi

run_combo() {
  local objective="$1"
  local modulation_mode="$2"
  local combo_root="${output_root%/}/${objective}/${modulation_mode}"
  local completed="$combo_root/comparison_summary.json"

  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed objective=$objective modulation=$modulation_mode"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$combo_root" ]]; then
    echo "Existing incomplete directory: $combo_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start objective=$objective modulation=$modulation_mode"
  DATASET="$dataset" \
  NOISE_MODE=degree_preserving_replace \
  NOISE_RATIOS=0 \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$combo_root" \
  TRAIN_EPOCHS="$train_epochs" \
  TRAIN_PATIENCE="$train_patience" \
  TRAIN_LR="${TRAIN_LR:-0.0005}" \
  TRAIN_INIT_WEIGHT="${TRAIN_INIT_WEIGHT:-0.01}" \
  TRAINING_OBJECTIVE="$objective" \
  SSM_NUM_NEG="${SSM_NUM_NEG:-1024}" \
  SSM_TAU="${SSM_TAU:-0.1}" \
  AU_UNIFORMITY_WEIGHT="${AU_UNIFORMITY_WEIGHT:-1.0}" \
  AU_UNIFORMITY_T="${AU_UNIFORMITY_T:-2.0}" \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE="$modulation_mode" \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA=1.0 \
  STOP_AFTER_FILTER=0 \
  SUMMARY_ONLY=1 \
  RUN_PILOT_ANALYSIS=0 \
  KEEP_EDGE_LABELS=0 \
  KEEP_GENERATED_TRAIN=0 \
  STRUCTURAL_MODE=two_hop_minhash \
  TOPK="${TOPK:-10}" \
  CHUNK_SIZE="${CHUNK_SIZE:-8192}" \
  MIN_DEGREE="${MIN_DEGREE:-2}" \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
  DRY_RUN="$dry_run" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"

  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$combo_root" --output "$completed"
  fi
  echo "Done objective=$objective modulation=$modulation_mode"
}

echo "Yelp2018 SSM/AU objective comparison"
echo "  objectives:          $objectives"
echo "  propagation modes:   $modulation_modes"
echo "  edge filtering:      none"
echo "  SSM negatives/tau:   ${SSM_NUM_NEG:-1024}/${SSM_TAU:-0.1}"
echo "  AU weight/t:         ${AU_UNIFORMITY_WEIGHT:-1.0}/${AU_UNIFORMITY_T:-2.0}"
echo "  seed:                $seed"
echo "  output:              $output_root"
echo "  planned runs:        4"

for objective in $objectives; do
  for modulation_mode in $modulation_modes; do
    run_combo "$objective" "$modulation_mode"
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"
python3 "$script_dir/analyze_objective_backbones.py" \
  --input "$output_root/all_runs.json" \
  --dataset "$dataset" \
  --objectives "$objectives" \
  --baseline-mode none \
  --treatment-mode original_always \
  --noise-ratio 0 \
  --seed "$seed" \
  --output "$output_root/objective_backbone_comparison.json" \
  --markdown "$output_root/objective_backbone_comparison.md"

echo "Yelp2018 objective comparison completed: $output_root"
echo "  table:    $output_root/objective_backbone_comparison.md"
echo "  JSON:     $output_root/objective_backbone_comparison.json"
echo "  all runs: $output_root/all_runs.json"
