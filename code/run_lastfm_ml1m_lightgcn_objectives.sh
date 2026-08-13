#!/usr/bin/env bash
set -euo pipefail

# Clean LightGCN objective baselines on the converted LastFM and ML-1M data.
# This script deliberately disables both edge filtering and CrossNorm so the
# only controlled difference within a dataset is SSM versus AU.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
datasets="${DATASETS:-lastfm ml-1m}"
objectives="${OBJECTIVES:-ssm au}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
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
read -r -a seed_values <<<"$seeds"
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
  local case_root="${output_root%/}/${dataset}/${objective}/seed_${seed}"
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed dataset=$dataset objective=$objective seed=$seed"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start dataset=$dataset objective=$objective seed=$seed"
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
  REPRESENTATION_MODULATION_MODE=none \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA=0 \
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
  echo "Done dataset=$dataset objective=$objective seed=$seed"
}

planned_runs=$((${#dataset_values[@]} * ${#objective_values[@]} * ${#seed_values[@]}))
echo "LastFM/ML-1M LightGCN SSM/AU objective baselines"
echo "  datasets:           $datasets"
echo "  objectives:         $objectives"
echo "  seeds:              $seeds"
echo "  SSM tau:            $ssm_tau"
echo "  AU weight/t:        $au_uniformity_weight/$au_uniformity_t"
echo "  lr/decay/dropout:   $learning_rate/$decay/$message_dropout"
echo "  batch:              $batch_size"
echo "  epochs/patience:    $train_epochs/$train_patience"
echo "  edge filtering:     none"
echo "  modulation:         none (ordinary LightGCN)"
echo "  planned runs:       $planned_runs"
echo "  output:             $output_root"

for dataset in "${dataset_values[@]}"; do
  for objective in "${objective_values[@]}"; do
    for seed in "${seed_values[@]}"; do
      run_case "$dataset" "$objective" "$seed"
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"

analysis_command=(
  python3 "$script_dir/analyze_cross_dataset_lightgcn_objectives.py"
  --input "$output_root/all_runs.json"
  --datasets
)
analysis_command+=("${dataset_values[@]}")
analysis_command+=(--objectives)
analysis_command+=("${objective_values[@]}")
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
  --output "$output_root/lightgcn_objective_baselines.json"
  --markdown "$output_root/lightgcn_objective_baselines.md"
)
"${analysis_command[@]}"

echo "Cross-dataset LightGCN objective baselines completed: $output_root"
echo "  table: $output_root/lightgcn_objective_baselines.md"
echo "  JSON:  $output_root/lightgcn_objective_baselines.json"
echo "  runs:  $output_root/all_runs.json"
