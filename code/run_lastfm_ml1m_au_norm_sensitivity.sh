#!/usr/bin/env bash
set -euo pipefail

# Clean LastFM/ML-1M AU CrossNorm blend sensitivity.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v5.1_lastfm_ml1m_au_norm}"
datasets="${DATASETS:-lastfm ml-1m}"
modulation_weights="${MODULATION_WEIGHTS:-0 0.2 0.4 0.6 0.8 1.0}"
lastfm_au_weight="${LASTFM_AU_WEIGHT:-1.0}"
ml1m_au_weight="${ML1M_AU_WEIGHT:-1.0}"
learning_rate="${TRAIN_LR:-0.0005}"
configured_decay="${TRAIN_DECAY:-0.001}"
uniformity_t="${AU_UNIFORMITY_T:-2.0}"
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
read -r -a modulation_values <<<"$modulation_weights"
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
python3 - "$modulation_weights" "$lastfm_au_weight" "$ml1m_au_weight" \
  "$learning_rate" "$configured_decay" "$uniformity_t" <<'PY'
import math
import sys
weights = [float(raw) for raw in sys.argv[1].split()]
if not weights or len(weights) != len(set(weights)):
    raise SystemExit("Modulation weights must be non-empty and unique")
if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights):
    raise SystemExit("Modulation weights must be within [0, 1]")
if not any(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12) for value in weights):
    raise SystemExit("MODULATION_WEIGHTS must include 0")
for label, raw, allow_zero in (
        ("LASTFM_AU_WEIGHT", sys.argv[2], True),
        ("ML1M_AU_WEIGHT", sys.argv[3], True),
        ("TRAIN_LR", sys.argv[4], False),
        ("TRAIN_DECAY", sys.argv[5], True),
        ("AU_UNIFORMITY_T", sys.argv[6], False)):
    value = float(raw)
    if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
        raise SystemExit("%s has an invalid value" % label)
PY

token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_case() {
  local dataset="$1"
  local modulation_weight="$2"
  local au_weight="$3"
  local mode="blend_always"
  if [[ "$modulation_weight" == "0" || "$modulation_weight" == "0.0" ]]; then
    mode="none"
  fi
  local case_root="${output_root%/}/${dataset}/au_mu_$(token "$modulation_weight")"
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed dataset=$dataset mu=$modulation_weight"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start dataset=$dataset mu=$modulation_weight au_weight=$au_weight"
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
  TRAIN_DECAY="$configured_decay" \
  TRAINING_OBJECTIVE=au \
  OBJECTIVE_MESSAGE_DROPOUT="$message_dropout" \
  AU_UNIFORMITY_WEIGHT="$au_weight" \
  AU_UNIFORMITY_T="$uniformity_t" \
  EDGE_FILTER_MODE=none \
  REPRESENTATION_MODULATION_MODE="$mode" \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA="$modulation_weight" \
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
  echo "Done dataset=$dataset mu=$modulation_weight au_weight=$au_weight"
}

total_count=$((${#dataset_values[@]} * ${#modulation_values[@]}))
echo "LastFM/ML-1M AU CrossNorm sensitivity"
echo "  datasets:           $datasets"
echo "  modulation weights:$modulation_weights"
echo "  AU weights LastFM/ML:$lastfm_au_weight/$ml1m_au_weight"
echo "  alignment weight/t:1/$uniformity_t"
echo "  fixed lr:           $learning_rate"
echo "  configured decay:   $configured_decay (unused by AU loss)"
echo "  batch:              $batch_size"
echo "  epochs/patience:    $train_epochs/$train_patience"
echo "  edge filtering:     none"
echo "  seed/GPU:           $seed/$gpu_id"
echo "  planned runs:       $total_count"
echo "  output:             $output_root"

for dataset in "${dataset_values[@]}"; do
  au_weight="$ml1m_au_weight"
  if [[ "$dataset" == "lastfm" ]]; then
    au_weight="$lastfm_au_weight"
  fi
  for modulation_weight in "${modulation_values[@]}"; do
    run_case "$dataset" "$modulation_weight" "$au_weight"
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"

analysis_command=(
  python3 "$script_dir/analyze_lastfm_ml1m_au_norm_sensitivity.py"
  --input "$output_root/all_runs.json"
  --datasets
)
analysis_command+=("${dataset_values[@]}")
analysis_command+=(--modulation-weights)
analysis_command+=("${modulation_values[@]}")
analysis_command+=(
  --lastfm-au-weight "$lastfm_au_weight"
  --ml1m-au-weight "$ml1m_au_weight"
  --learning-rate "$learning_rate"
  --configured-decay "$configured_decay"
  --uniformity-t "$uniformity_t"
  --seed "$seed"
  --message-dropout "$message_dropout"
  --batch-size "$batch_size"
  --max-epochs "$train_epochs"
  --patience "$train_patience"
  --output "$output_root/lastfm_ml1m_au_norm_sensitivity.json"
  --markdown "$output_root/lastfm_ml1m_au_norm_sensitivity.md"
)
"${analysis_command[@]}"

echo "AU CrossNorm sensitivity completed: $output_root"
echo "  table: $output_root/lastfm_ml1m_au_norm_sensitivity.md"
echo "  JSON:  $output_root/lastfm_ml1m_au_norm_sensitivity.json"
echo "  runs:  $output_root/all_runs.json"
