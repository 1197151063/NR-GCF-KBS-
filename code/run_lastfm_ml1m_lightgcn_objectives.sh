#!/usr/bin/env bash
set -euo pipefail

# Clean LightGCN and CrossNorm-blend objective experiment on LastFM and ML-1M.
# Baselines are completed first and can be reused before the nonzero modulation
# weights are run. Edge filtering remains disabled in every arm.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
datasets="${DATASETS:-lastfm ml-1m}"
objectives="${OBJECTIVES:-ssm au}"
weights="${MODULATION_WEIGHTS:-0.2 0.4 0.6 0.8 1.0}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
phase="${PHASE:-all}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v4.3_lastfm_ml1m_lightgcn_objectives}"
learning_rate="${TRAIN_LR:-0.0005}"
decay="${TRAIN_DECAY:-0.0001}"
message_dropout="${OBJECTIVE_MESSAGE_DROPOUT:-0.0}"
batch_size="${TRAIN_BATCH_SIZE:-2048}"
ssm_tau="${SSM_TAU:-0.1}"
au_uniformity_weight="${AU_UNIFORMITY_WEIGHT:-1.0}"
au_uniformity_t="${AU_UNIFORMITY_T:-2.0}"
train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

if [[ "$datasets" != "lastfm ml-1m" ]]; then
  echo "This controlled experiment requires DATASETS='lastfm ml-1m'." >&2
  exit 2
fi
if [[ "$objectives" != "ssm au" ]]; then
  echo "This controlled experiment requires OBJECTIVES='ssm au'." >&2
  exit 2
fi
if [[ "$phase" != "all" && "$phase" != "baseline" && "$phase" != "modulation" ]]; then
  echo "PHASE must be all, baseline, or modulation." >&2
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
read -r -a dataset_values <<<"$datasets"
read -r -a objective_values <<<"$objectives"
read -r -a weight_values <<<"$weights"
read -r -a seed_values <<<"$seeds"
python3 - "$weights" <<'PY'
import math
import sys
values = [float(raw) for raw in sys.argv[1].split()]
if len(set(values)) != len(values):
    raise SystemExit("Modulation weights must be unique")
for value in values:
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise SystemExit("Modulation weights must be within (0, 1]")
PY
for dataset in "${dataset_values[@]}"; do
  for split in train test; do
    if [[ ! -f "$script_dir/../data/$dataset/$split.txt" ]]; then
      echo "Missing converted split: data/$dataset/$split.txt" >&2
      exit 2
    fi
  done
done

run_case() {
  local dataset="$1"
  local objective="$2"
  local seed="$3"
  local mode="$4"
  local weight="$5"
  local stage="$6"
  local case_root
  if [[ "$mode" == "none" ]]; then
    # Preserve the v4.3 baseline path so already completed runs are reusable.
    case_root="${output_root%/}/${dataset}/${objective}/seed_${seed}"
  else
    local weight_token="${weight//./p}"
    weight_token="${weight_token//-/m}"
    case_root="${output_root%/}/${dataset}/${objective}/modulation/mu_${weight_token}/seed_${seed}"
  fi
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed stage=$stage dataset=$dataset objective=$objective mu=$weight seed=$seed"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start stage=$stage dataset=$dataset objective=$objective mode=$mode mu=$weight seed=$seed"
  DATASET="$dataset" \
  NOISE_MODE=degree_preserving_replace \
  NOISE_RATIOS=0 \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$case_root" \
  TRAIN_EPOCHS="$train_epochs" \
  TRAIN_PATIENCE="$train_patience" \
  TRAIN_BATCH_SIZE="$batch_size" \
  TRAIN_LR="$learning_rate" \
  TRAIN_INIT_METHOD=xavier_uniform \
  TRAIN_INIT_WEIGHT=1.0 \
  TRAIN_DECAY="$decay" \
  TRAINING_OBJECTIVE="$objective" \
  SSM_TAU="$ssm_tau" \
  OBJECTIVE_MESSAGE_DROPOUT="$message_dropout" \
  AU_UNIFORMITY_WEIGHT="$au_uniformity_weight" \
  AU_UNIFORMITY_T="$au_uniformity_t" \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE="$mode" \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA="$weight" \
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
      --root "$case_root" --output "$completed"
  fi
  echo "Done stage=$stage dataset=$dataset objective=$objective mu=$weight seed=$seed"
}

baseline_runs=$((${#dataset_values[@]} * ${#objective_values[@]} * ${#seed_values[@]}))
modulation_runs=$((baseline_runs * ${#weight_values[@]}))
echo "LastFM/ML-1M LightGCN and CrossNorm objective experiment"
echo "  datasets:           $datasets"
echo "  objectives:         $objectives"
echo "  stage order:        LightGCN baselines, then modulation sweep"
echo "  modulation weights:$weights"
echo "  seeds:              $seeds"
echo "  SSM tau:            $ssm_tau"
echo "  AU weight/t:        $au_uniformity_weight/$au_uniformity_t"
echo "  lr/decay/dropout:   $learning_rate/$decay/$message_dropout"
echo "  batch:              $batch_size"
echo "  epochs/patience:    $train_epochs/$train_patience"
echo "  edge filtering:     none"
echo "  planned runs:       baseline=$baseline_runs modulation=$modulation_runs"
echo "  phase:              $phase"
echo "  output:             $output_root"

if [[ "$phase" == "all" || "$phase" == "baseline" ]]; then
  for dataset in "${dataset_values[@]}"; do
    for objective in "${objective_values[@]}"; do
      for seed in "${seed_values[@]}"; do
        run_case "$dataset" "$objective" "$seed" none 0 baseline
      done
    done
  done
fi

if [[ "$phase" == "all" || "$phase" == "modulation" ]]; then
  for dataset in "${dataset_values[@]}"; do
    for objective in "${objective_values[@]}"; do
      for weight in "${weight_values[@]}"; do
        for seed in "${seed_values[@]}"; do
          run_case "$dataset" "$objective" "$seed" blend_always "$weight" modulation
        done
      done
    done
  done
fi

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

if [[ "$phase" == "baseline" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$output_root" --output "$output_root/lightgcn_objective_baselines.json"
  echo "LightGCN baseline phase completed: $output_root"
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"

analysis_command=(
  python3 "$script_dir/analyze_cross_dataset_objective_modulation.py"
  --input "$output_root/all_runs.json"
  --datasets
)
analysis_command+=("${dataset_values[@]}")
analysis_command+=(--objectives)
analysis_command+=("${objective_values[@]}")
analysis_command+=(--weights)
analysis_command+=("${weight_values[@]}")
analysis_command+=(--seeds)
analysis_command+=("${seed_values[@]}")
analysis_command+=(
  --learning-rate "$learning_rate"
  --decay "$decay"
  --message-dropout "$message_dropout"
  --batch-size "$batch_size"
  --ssm-tau "$ssm_tau"
  --au-uniformity-weight "$au_uniformity_weight"
  --au-uniformity-t "$au_uniformity_t"
  --output "$output_root/objective_modulation_sensitivity.json"
  --markdown "$output_root/objective_modulation_sensitivity.md"
)
"${analysis_command[@]}"

echo "Cross-dataset objective modulation experiment completed: $output_root"
echo "  table: $output_root/objective_modulation_sensitivity.md"
echo "  JSON:  $output_root/objective_modulation_sensitivity.json"
echo "  runs:  $output_root/all_runs.json"
