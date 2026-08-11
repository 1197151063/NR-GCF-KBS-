#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Adap_tau LightGCN training protocol inside NR-GCF. CrossNorm is
# evaluated only with fixed-temperature SSM; Adap-tau remains an independent
# comparison method and is never combined with CrossNorm. Yelp has no
# validation file in either repository, so the existing NR-GCF direct-test
# Recall@20 selection matches the reference's effective protocol.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset=yelp2018
objectives="${OBJECTIVES:-ssm adap_tau}"
# The reference entry point fixes all random seeds to 2020.
seed="${SEED:-2020}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.5_yelp_adap_tau_reference}"
train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-51}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

if [[ "$objectives" != "ssm adap_tau" ]]; then
  echo "This reproduction requires OBJECTIVES='ssm adap_tau'." >&2
  exit 2
fi
for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done

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
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}" \
  TRAIN_LR="${TRAIN_LR:-0.001}" \
  TRAIN_DECAY="${TRAIN_DECAY:-0.1}" \
  TRAIN_INIT_WEIGHT=1.0 \
  TRAINING_OBJECTIVE="$objective" \
  SSM_NUM_NEG=1024 \
  SSM_TAU="${SSM_TAU:-0.1}" \
  OBJECTIVE_MESSAGE_DROPOUT="${OBJECTIVE_MESSAGE_DROPOUT:-0.1}" \
  ADAP_TAU_MODE="${ADAP_TAU_MODE:-weight_mean}" \
  ADAP_TAU_TEMPERATURE_2="${ADAP_TAU_TEMPERATURE_2:-1.5}" \
  ADAP_TAU_LOSS_QUANTILE="${ADAP_TAU_LOSS_QUANTILE:-1.0}" \
  ADAP_TAU_RECALIBRATION_EPOCH="${ADAP_TAU_RECALIBRATION_EPOCH:-100}" \
  ADAP_TAU_DEGREE_QUANTILE="${ADAP_TAU_DEGREE_QUANTILE:-0.2}" \
  ADAP_TAU_INITIAL_POSITIVE_GAP="${ADAP_TAU_INITIAL_POSITIVE_GAP:-0.7}" \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE="$modulation_mode" \
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

  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$combo_root" --output "$completed"
  fi
  echo "Done objective=$objective modulation=$modulation_mode"
}

echo "Yelp2018 Adap_tau-reference objective reproduction"
echo "  objectives:        $objectives"
echo "  planned methods:   SSM+LightGCN, SSM+CrossNorm, Adap-tau+LightGCN"
echo "  negatives:         B-1 in-batch (n_negs ignored)"
echo "  batch/lr/L2:       ${TRAIN_BATCH_SIZE:-2048}/${TRAIN_LR:-0.001}/${TRAIN_DECAY:-0.1}"
echo "  message dropout:   ${OBJECTIVE_MESSAGE_DROPOUT:-0.1}"
echo "  SSM tau:           ${SSM_TAU:-0.1}"
echo "  Adap-tau mode/t2:  ${ADAP_TAU_MODE:-weight_mean}/${ADAP_TAU_TEMPERATURE_2:-1.5}"
echo "  selection:         direct test Recall@20, patience $train_patience"
echo "  output:            $output_root"
echo "  planned runs:      3"

run_combo ssm none
run_combo ssm original_always
run_combo adap_tau none

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"
python3 "$script_dir/analyze_objective_backbones.py" \
  --input "$output_root/all_runs.json" \
  --dataset "$dataset" \
  --objectives ssm \
  --baseline-mode none \
  --treatment-mode original_always \
  --noise-ratio 0 \
  --seed "$seed" \
  --output "$output_root/objective_backbone_comparison.json" \
  --markdown "$output_root/objective_backbone_comparison.md"

echo "Completed: $output_root"
