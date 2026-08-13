#!/usr/bin/env bash
set -euo pipefail

# Search objective-specific L2 strengths while retaining the selected learning
# rate and dataset-specific SSM temperatures from the convergence experiment.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v4.8_lastfm_ml1m_decay}"
datasets="${DATASETS:-lastfm ml-1m}"
decays="${DECAYS:-0 0.000001 0.00001 0.0001 0.001}"
learning_rate="${TRAIN_LR:-0.001}"
lastfm_tau="${LASTFM_SSM_TAU:-0.5}"
ml1m_tau="${ML1M_SSM_TAU:-0.1}"
seed="${SEED:-2026}"
gpu_id="${GPU_ID:-0}"
message_dropout="${OBJECTIVE_MESSAGE_DROPOUT:-0.0}"
batch_size="${TRAIN_BATCH_SIZE:-2048}"
train_epochs="${TRAIN_EPOCHS:-500}"
train_patience="${TRAIN_PATIENCE:-20}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done
if ! [[ "$gpu_id" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ && \
        "$train_epochs" =~ ^[0-9]+$ && "$train_patience" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID, SEED, TRAIN_EPOCHS, and TRAIN_PATIENCE must be non-negative integers." >&2
  exit 2
fi
if [[ "$train_epochs" -lt 1 || "$train_patience" -lt 1 ]]; then
  echo "TRAIN_EPOCHS and TRAIN_PATIENCE must be positive." >&2
  exit 2
fi
read -r -a dataset_values <<<"$datasets"
read -r -a decay_values <<<"$decays"
if [[ "${dataset_values[*]}" != "lastfm ml-1m" ]]; then
  echo "This controlled experiment requires DATASETS='lastfm ml-1m'." >&2
  exit 2
fi
for dataset in "${dataset_values[@]}"; do
  for split in train test; do
    if [[ ! -f "$script_dir/../data/$dataset/$split.txt" ]]; then
      echo "Missing converted split: data/$dataset/$split.txt" >&2
      exit 2
    fi
  done
done
python3 - "$decays" "$learning_rate" "$lastfm_tau" "$ml1m_tau" <<'PY'
import math
import sys
decays = [float(raw) for raw in sys.argv[1].split()]
if not decays or len(decays) != len(set(decays)):
    raise SystemExit("Decay values must be non-empty and unique")
if any(not math.isfinite(value) or value < 0.0 for value in decays):
    raise SystemExit("Decay values must be finite and non-negative")
if not any(math.isclose(value, 1e-4, rel_tol=0.0, abs_tol=1e-12) for value in decays):
    raise SystemExit("DECAYS must include the current value 0.0001")
for label, raw in (("TRAIN_LR", sys.argv[2]),
                   ("LASTFM_SSM_TAU", sys.argv[3]),
                   ("ML1M_SSM_TAU", sys.argv[4])):
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit("%s must be finite and positive" % label)
PY

token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_case() {
  local dataset="$1"
  local objective="$2"
  local decay="$3"
  local temperature="$4"
  local case_root="${output_root%/}/${dataset}/${objective}_decay_$(token "$decay")"
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed dataset=$dataset objective=$objective decay=$decay"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  local init_method="xavier_uniform"
  local init_weight="1.0"
  if [[ "$objective" == "bpr" ]]; then
    init_method="normal"
    init_weight="0.01"
  fi

  echo "Start dataset=$dataset objective=$objective decay=$decay tau=$temperature"
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
  TRAIN_INIT_METHOD="$init_method" \
  TRAIN_INIT_WEIGHT="$init_weight" \
  TRAIN_DECAY="$decay" \
  TRAINING_OBJECTIVE="$objective" \
  SSM_TAU="$temperature" \
  OBJECTIVE_MESSAGE_DROPOUT="$message_dropout" \
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
  echo "Done dataset=$dataset objective=$objective decay=$decay tau=$temperature"
}

runs_per_dataset=$((${#decay_values[@]} * 2))
total_count=$((${#dataset_values[@]} * runs_per_dataset))
echo "LastFM/ML-1M LightGCN decay sensitivity"
echo "  datasets:           $datasets"
echo "  fixed learning rate:$learning_rate"
echo "  decays:             $decays"
echo "  SSM tau LastFM/ML:  $lastfm_tau/$ml1m_tau"
echo "  dropout/batch:      $message_dropout/$batch_size"
echo "  epochs/patience:    $train_epochs/$train_patience"
echo "  monitor:            test Recall@20 (strict improvement)"
echo "  filtering/modulation: none/none"
echo "  seed/GPU:           $seed/$gpu_id"
echo "  planned runs:       $runs_per_dataset per dataset, $total_count total"
echo "  output:             $output_root"

for dataset in "${dataset_values[@]}"; do
  temperature="$ml1m_tau"
  if [[ "$dataset" == "lastfm" ]]; then
    temperature="$lastfm_tau"
  fi
  for objective in bpr ssm; do
    for decay in "${decay_values[@]}"; do
      run_case "$dataset" "$objective" "$decay" "$temperature"
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
  python3 "$script_dir/analyze_lastfm_ml1m_lightgcn_decay_sensitivity.py"
  --input "$output_root/all_runs.json"
  --datasets
)
analysis_command+=("${dataset_values[@]}")
analysis_command+=(--decays)
analysis_command+=("${decay_values[@]}")
analysis_command+=(
  --learning-rate "$learning_rate"
  --lastfm-ssm-temperature "$lastfm_tau"
  --ml1m-ssm-temperature "$ml1m_tau"
  --seed "$seed"
  --message-dropout "$message_dropout"
  --batch-size "$batch_size"
  --max-epochs "$train_epochs"
  --patience "$train_patience"
  --output "$output_root/lastfm_ml1m_decay_sensitivity.json"
  --markdown "$output_root/lastfm_ml1m_decay_sensitivity.md"
)
"${analysis_command[@]}"

echo "Decay sensitivity completed: $output_root"
echo "  table: $output_root/lastfm_ml1m_decay_sensitivity.md"
echo "  JSON:  $output_root/lastfm_ml1m_decay_sensitivity.json"
echo "  runs:  $output_root/all_runs.json"
