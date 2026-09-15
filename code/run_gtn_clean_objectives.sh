#!/usr/bin/env bash
set -euo pipefail

# Independent GTN comparison under the project's BPR, SSM, and AU protocol.
# LastFM and ML-1M run a compact lambda search; Yelp and Amazon-Book use the
# supplied GTN reproduction values. SDR-GCF filtering and CrossNorm are off.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
profile_file="${PROFILE_FILE:-$repo_root/configs/gtn_clean_objectives.json}"
datasets="${DATASETS:-lastfm ml-1m yelp2018 amazon-book}"
objectives="${OBJECTIVES:-bpr ssm au}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_gtn_clean_objectives}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

for flag in dry_run skip_completed; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${flag^^} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ ! -f "$profile_file" ]]; then
  echo "Missing profile: $profile_file" >&2
  exit 2
fi
if [[ "$dry_run" != "1" ]] && ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer." >&2
  exit 2
fi

read -r -a dataset_values <<<"$datasets"
read -r -a objective_values <<<"$objectives"
read -r -a seed_values <<<"$seeds"

profile_values() {
  local objective="$1"
  local dataset="$2"
  python3 - "$profile_file" "$objective" "$dataset" <<'PY'
import json
import sys

path, objective, dataset = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    profile = json.load(stream)
if objective not in profile["objectives"]:
    raise SystemExit("Unsupported objective: " + objective)
if dataset not in profile["datasets"]:
    raise SystemExit("Unsupported dataset: " + dataset)
common = profile["common"]
objective_profile = profile["profiles"][objective]
dataset_profile = objective_profile["datasets"][dataset]
values = [
    common["train_epochs"], common["train_patience"],
    common["train_batch_size"], common["train_lr"],
    common["gtn_iterations"], common["gtn_prop_dropout"],
    objective_profile["train_init_method"],
    objective_profile["train_init_weight"],
    dataset_profile["train_decay"],
    dataset_profile.get("ssm_tau", 0.1),
    dataset_profile.get("au_uniformity_weight", 1.0),
    common["au_uniformity_t"],
]
print("\t".join(map(str, values)))
PY
}

lambda_values() {
  local dataset="$1"
  python3 - "$profile_file" "$dataset" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    profile = json.load(stream)
dataset = sys.argv[2]
if dataset not in profile["datasets"]:
    raise SystemExit("Unsupported dataset: " + dataset)
print(" ".join(map(str, profile["gtn_datasets"][dataset]["lambda_candidates"])))
PY
}

run_case() {
  local dataset="$1"
  local objective="$2"
  local seed="$3"
  local gtn_lambda="$4"
  local lambda_token="${gtn_lambda//./p}"
  lambda_token="${lambda_token//-/m}"
  local case_root="${output_root%/}/${objective}/${dataset}/lambda_${lambda_token}/seed_${seed}"
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed objective=$objective dataset=$dataset lambda=$gtn_lambda seed=$seed"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete directory: $case_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  local epochs patience batch_size learning_rate iterations prop_dropout
  local init_method init_weight decay ssm_tau au_weight au_t
  IFS=$'\t' read -r epochs patience batch_size learning_rate iterations \
    prop_dropout init_method init_weight decay ssm_tau au_weight au_t \
    < <(profile_values "$objective" "$dataset")

  echo "Start GTN objective=$objective dataset=$dataset lambda=$gtn_lambda seed=$seed"
  DATASET="$dataset" \
  NOISE_MODE=prepared \
  NOISE_RATIOS=0 \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$case_root" \
  BACKBONE=gtn \
  TRAIN_K="$iterations" \
  GTN_LAMBDA="$gtn_lambda" \
  GTN_PROP_DROPOUT="$prop_dropout" \
  TRAIN_EPOCHS="$epochs" \
  TRAIN_PATIENCE="$patience" \
  TRAIN_BATCH_SIZE="$batch_size" \
  TRAIN_LR="$learning_rate" \
  TRAIN_INIT_METHOD="$init_method" \
  TRAIN_INIT_WEIGHT="$init_weight" \
  TRAIN_DECAY="$decay" \
  TRAINING_OBJECTIVE="$objective" \
  SSM_TAU="$ssm_tau" \
  OBJECTIVE_MESSAGE_DROPOUT=0.0 \
  AU_UNIFORMITY_WEIGHT="$au_weight" \
  AU_UNIFORMITY_T="$au_t" \
  EDGE_FILTER_MODE=none \
  STRUCTURAL_MODE=none \
  REPRESENTATION_MODULATION_MODE=none \
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
  echo "Done GTN objective=$objective dataset=$dataset lambda=$gtn_lambda seed=$seed"
}

planned=0
for dataset in "${dataset_values[@]}"; do
  read -r -a candidate_values <<<"$(lambda_values "$dataset")"
  planned=$((planned + ${#candidate_values[@]} * ${#objective_values[@]} * ${#seed_values[@]}))
done
echo "GTN clean objective experiment"
echo "  datasets:        $datasets"
echo "  objectives:      $objectives"
echo "  seeds:           $seeds"
echo "  planned runs:    $planned"
echo "  model selection: best Recall@20 epoch (paired NDCG@20)"
echo "  filtering/norm:  none/none"
echo "  profile:         $profile_file"
echo "  output:          $output_root"

for objective in "${objective_values[@]}"; do
  for dataset in "${dataset_values[@]}"; do
    read -r -a candidate_values <<<"$(lambda_values "$dataset")"
    for gtn_lambda in "${candidate_values[@]}"; do
      for seed in "${seed_values[@]}"; do
        run_case "$dataset" "$objective" "$seed" "$gtn_lambda"
      done
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"
python3 "$script_dir/analyze_gtn_clean_objectives.py" \
  --input "$output_root/all_runs.json" \
  --profile "$profile_file" \
  --output "$output_root/gtn_clean_objectives_summary.json" \
  --markdown "$output_root/gtn_clean_objectives_summary.md"

echo "GTN clean experiment completed: $output_root"
echo "  table: $output_root/gtn_clean_objectives_summary.md"
echo "  JSON:  $output_root/gtn_clean_objectives_summary.json"
