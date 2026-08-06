#!/usr/bin/env bash
set -euo pipefail

# Parallel hyperparameter sensitivity for the active structure--momentum
# fusion weight:
#
#   risk(e) = lambda * (1 - structure_rank(e))
#             + (1 - lambda) * momentum_rank(e)
#
# IMPORTANT: this lambda maps to RELIABILITY_STRUCTURE_WEIGHT.  It is not the
# legacy --lambda_ argument, which does not participate in original_always
# direct CrossNorm and is fixed to 1 below.
#
# The launcher creates GPU worker slots.  Each slot executes independent
# (noise ratio, lambda, seed) jobs sequentially, while slots run concurrently.
# This keeps multiple jobs resident on each GPU without allowing unbounded
# background processes.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset="${DATASET:-yelp2018}"
noise_ratios="${NOISE_RATIOS:-0 0.1 0.2}"
lambda_values="${LAMBDA_VALUES:-0.00 0.25 0.50 0.75 0.90 0.95 1.00}"
seeds="${SEEDS:-2026}"
gpu_ids_text="${GPU_IDS:-${GPU_ID:-0}}"
gpu_ids_text="${gpu_ids_text//,/ }"
jobs_per_gpu="${JOBS_PER_GPU:-2}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/lambda_sensitivity}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
train_lr="${TRAIN_LR:-0.0005}"
train_init_weight="${TRAIN_INIT_WEIGHT:-0.01}"
omp_threads="${NRGCF_OMP_NUM_THREADS:-2}"
adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-2}"
adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-4}"
adaptive_min_coverage="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}"
adaptive_jaccard="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}"
adaptive_stable_checks="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-1}"

if ! [[ "$jobs_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS_PER_GPU must be a positive integer." >&2
  exit 2
fi
for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done

read -r -a gpu_ids <<<"$gpu_ids_text"
read -r -a ratio_array <<<"$noise_ratios"
read -r -a lambda_array <<<"$lambda_values"
read -r -a seed_array <<<"$seeds"
if [[ ${#gpu_ids[@]} -eq 0 || ${#ratio_array[@]} -eq 0 || \
      ${#lambda_array[@]} -eq 0 || ${#seed_array[@]} -eq 0 ]]; then
  echo "GPU_IDS, NOISE_RATIOS, LAMBDA_VALUES, and SEEDS cannot be empty." >&2
  exit 2
fi
for gpu in "${gpu_ids[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID: $gpu" >&2
    exit 2
  fi
done

python3 - "$noise_ratios" "$lambda_values" <<'PY'
import math
import sys

for label, text, bounded in (
    ("noise ratio", sys.argv[1], False),
    ("lambda", sys.argv[2], True),
):
    for raw in text.split():
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if not math.isfinite(value) or value < 0 or (bounded and value > 1):
            raise SystemExit("Invalid %s: %s" % (label, raw))
PY

ratio_tag() {
  python3 - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(format(value, ".12g").replace("-", "m").replace(".", "p"))
PY
}

lambda_tag() {
  python3 - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(format(value, ".12g").replace("-", "m").replace(".", "p"))
PY
}

tasks_ratio=()
tasks_lambda=()
tasks_seed=()
for ratio in "${ratio_array[@]}"; do
  for active_lambda in "${lambda_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid integer seed: $seed" >&2
        exit 2
      fi
      tasks_ratio+=("$ratio")
      tasks_lambda+=("$active_lambda")
      tasks_seed+=("$seed")
    done
  done
done

total_slots=$((${#gpu_ids[@]} * jobs_per_gpu))
total_tasks=${#tasks_ratio[@]}
launcher_log_dir="${output_root%/}/launcher_logs"
mkdir -p "$launcher_log_dir"

echo "NR-GCF lambda sensitivity"
echo "  dataset:          $dataset"
echo "  noise ratios:     $noise_ratios"
echo "  active lambdas:   $lambda_values"
echo "  seeds:            $seeds"
echo "  GPUs:             ${gpu_ids[*]}"
echo "  jobs per GPU:     $jobs_per_gpu"
echo "  worker slots:     $total_slots"
echo "  total jobs:       $total_tasks"
echo "  output:           $output_root"
echo "  lambda semantics: edge-reliability structure weight"
echo "  legacy --lambda_: fixed to 1 and ignored by original_always"

worker_loop() {
  local slot_index="$1"
  local gpu="$2"
  local worker_index="$3"
  local task_index ratio active_lambda seed ltag rtag lambda_root
  local run_name run_dir log_path

  for ((task_index=slot_index; task_index<total_tasks; task_index+=total_slots)); do
    ratio="${tasks_ratio[$task_index]}"
    active_lambda="${tasks_lambda[$task_index]}"
    seed="${tasks_seed[$task_index]}"
    ltag="$(lambda_tag "$active_lambda")"
    rtag="$(ratio_tag "$ratio")"
    lambda_root="${output_root%/}/lambda_${ltag}"
    run_name="hard_structure_momentum_replace_noise_${rtag}_seed_${seed}_filter_adaptive_${adaptive_min_epoch}_${adaptive_max_epoch}_mod_original_always"
    run_dir="$lambda_root/$dataset/$run_name"
    log_path="$launcher_log_dir/gpu_${gpu}_worker_${worker_index}_noise_${rtag}_lambda_${ltag}_seed_${seed}.log"

    if [[ "$dry_run" != "1" && "$skip_completed" == "1" && \
          -f "$run_dir/edge_reliability/training_summary.json" ]]; then
      echo "[GPU $gpu worker $worker_index] skip completed: noise=$ratio lambda=$active_lambda seed=$seed"
      continue
    fi
    if [[ "$dry_run" != "1" && -e "$run_dir" ]]; then
      echo "Incomplete/existing run blocks resume: $run_dir" >&2
      echo "Move that single directory aside, or set a new OUTPUT_ROOT." >&2
      return 1
    fi

    echo "[GPU $gpu worker $worker_index] start: noise=$ratio lambda=$active_lambda seed=$seed"
    DATASET="$dataset" \
    NOISE_MODE=degree_preserving_replace \
    REPLACEMENT_SELECTION=uniform \
    NOISE_RATIOS="$ratio" \
    SEEDS="$seed" \
    GPU_ID="$gpu" \
    OUTPUT_ROOT="$lambda_root" \
    TRAIN_EPOCHS="$train_epochs" \
    TRAIN_PATIENCE="$train_patience" \
    TRAIN_LR="$train_lr" \
    TRAIN_INIT_WEIGHT="$train_init_weight" \
    NRGCF_OMP_NUM_THREADS="$omp_threads" \
    STOP_AFTER_FILTER=0 \
    SUMMARY_ONLY=1 \
    RUN_PILOT_ANALYSIS=0 \
    KEEP_EDGE_LABELS=0 \
    KEEP_GENERATED_TRAIN=0 \
    STRUCTURAL_MODE=two_hop_minhash \
    TOPK="${TOPK:-10}" \
    CHUNK_SIZE="${CHUNK_SIZE:-8192}" \
    MIN_DEGREE="${MIN_DEGREE:-2}" \
    EDGE_FILTER_MODE=hard_structure_momentum \
    RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}" \
    RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}" \
    RELIABILITY_STRUCTURE_WEIGHT="$active_lambda" \
    RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
    RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
    RELIABILITY_FILTER_SCHEDULE=adaptive \
    RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
    RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
    RELIABILITY_ADAPTIVE_MIN_COVERAGE="$adaptive_min_coverage" \
    RELIABILITY_ADAPTIVE_JACCARD="$adaptive_jaccard" \
    RELIABILITY_ADAPTIVE_STABLE_CHECKS="$adaptive_stable_checks" \
    REPRESENTATION_MODULATION_MODE=original_always \
    REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
    REPRESENTATION_MODULATION_LAMBDA=1 \
    REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
    DRY_RUN="$dry_run" \
      bash "$script_dir/run_edge_diagnostics_grid.sh" >"$log_path" 2>&1
    echo "[GPU $gpu worker $worker_index] done: noise=$ratio lambda=$active_lambda seed=$seed"
  done
}

child_pids=()
slot=0
for gpu in "${gpu_ids[@]}"; do
  for ((worker=0; worker<jobs_per_gpu; worker++)); do
    worker_loop "$slot" "$gpu" "$worker" &
    child_pids+=("$!")
    slot=$((slot + 1))
  done
done

terminate_children() {
  local pid
  for pid in "${child_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

failed=0
for pid in "${child_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
trap - INT TERM
if [[ "$failed" != "0" ]]; then
  echo "At least one lambda-sensitivity worker failed. Inspect $launcher_log_dir" >&2
  exit 1
fi

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training or summary was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/lambda_grid_runs.json"

python3 "$script_dir/select_lambda_sensitivity.py" \
  --input "$output_root/lambda_grid_runs.json" \
  --output "$output_root/best_lambda_by_noise.json" \
  --markdown "$output_root/lambda_sensitivity_table.md" \
  --selection-metric "${SELECTION_METRIC:-best_recall_at_20}"

echo "Lambda sensitivity completed."
echo "  compact run table: $output_root/lambda_grid_runs.json"
echo "  selected lambdas:  $output_root/best_lambda_by_noise.json"
echo "  readable table:    $output_root/lambda_sensitivity_table.md"
