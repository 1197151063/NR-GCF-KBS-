#!/usr/bin/env bash
set -euo pipefail

# Full objective robustness curve for SSM or AU. The default profile remains
# corrected in-batch SSM; the AU wrapper supplies its own objective profile.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile_file="${PROFILE_FILE:-$script_dir/../configs/full_ssm_edge_filter_norm.json}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v5.4_ssm_full_noise_curve_lr_split}"
datasets="${DATASETS:-yelp2018 amazon-book lastfm ml-1m}"
noise_ratios="${NOISE_RATIOS:-0 0.1 0.2 0.3 0.4 0.5}"
seeds="${SEEDS:-2026}"
arms="${ARMS:-lightgcn full}"
gpu_id="${GPU_ID:-0}"
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
  echo "Missing profile file: $profile_file" >&2
  exit 2
fi
training_objective="$(python3 - "$profile_file" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream).get("objective", ""))
PY
)"
if [[ "$training_objective" != "ssm" && "$training_objective" != "au" ]]; then
  echo "Profile objective must be ssm or au." >&2
  exit 2
fi
objective_label="$(printf '%s' "$training_objective" | tr '[:lower:]' '[:upper:]')"
if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer." >&2
  exit 2
fi

read -r -a dataset_values <<<"$datasets"
read -r -a ratio_values <<<"$noise_ratios"
read -r -a seed_values <<<"$seeds"
read -r -a arm_values <<<"$arms"

python3 - "$profile_file" "$datasets" "$noise_ratios" "$seeds" "$arms" <<'PY'
import json
import math
import sys

profile_path, datasets_text, ratios_text, seeds_text, arms_text = sys.argv[1:]
with open(profile_path, encoding="utf-8") as stream:
    profile = json.load(stream)
if profile.get("objective") not in {"ssm", "au"}:
    raise SystemExit("The selected profile is not an SSM/AU profile")
datasets = datasets_text.split()
unknown = sorted(set(datasets) - set(profile["datasets"]))
if unknown:
    raise SystemExit("Datasets missing from profile: %s" % ", ".join(unknown))
if not datasets or len(datasets) != len(set(datasets)):
    raise SystemExit("DATASETS must be non-empty and unique")
ratios = [float(raw) for raw in ratios_text.split()]
if not ratios or len(ratios) != len(set(ratios)):
    raise SystemExit("NOISE_RATIOS must be non-empty and unique")
if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios):
    raise SystemExit("Every noise ratio must be finite and within [0, 1]")
seeds = seeds_text.split()
if not seeds or len(seeds) != len(set(seeds)) or any(not x.isdigit() for x in seeds):
    raise SystemExit("SEEDS must contain unique non-negative integers")
arms = arms_text.split()
if not arms or len(arms) != len(set(arms)):
    raise SystemExit("ARMS must be non-empty and unique")
unknown_arms = sorted(set(arms) - set(profile["arms"]))
if unknown_arms:
    raise SystemExit("Unsupported arms: %s" % ", ".join(unknown_arms))
PY

profile_value() {
  local dataset="$1"
  local key="$2"
  python3 - "$profile_file" "$dataset" "$key" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    profile = json.load(stream)
dataset, key = sys.argv[2:]
value = profile.get("common", {}).get(key, profile["datasets"][dataset].get(key))
if value is None:
    raise SystemExit("Missing profile key %s for %s" % (key, dataset))
if isinstance(value, bool):
    print("1" if value else "0")
else:
    print(value)
PY
}

number_tag() {
  python3 - "$1" <<'PY'
import sys
print(format(float(sys.argv[1]), ".12g").replace("-", "m").replace(".", "p"))
PY
}

run_case() {
  local dataset="$1"
  local arm="$2"
  local ratio="$3"
  local seed="$4"
  local ratio_tag
  ratio_tag="$(number_tag "$ratio")"
  local case_root="${output_root%/}/${dataset}/${arm}/noise_${ratio_tag}/seed_${seed}"
  local completed="$case_root/comparison_summary.json"
  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed dataset=$dataset arm=$arm noise=$ratio seed=$seed"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$case_root" ]]; then
    echo "Existing incomplete case: $case_root" >&2
    echo "Move only this case aside, then resume with SKIP_COMPLETED=1." >&2
    exit 1
  fi

  local filter_mode="none"
  local modulation_mode="none"
  local modulation_weight="0"
  local train_lr
  train_lr="$(profile_value "$dataset" non_crossnorm_train_lr)"
  if [[ "$arm" == "norm_only" || "$arm" == "full" ]]; then
    modulation_mode="blend_always"
    modulation_weight="$(profile_value "$dataset" modulation_weight)"
    train_lr="$(profile_value "$dataset" crossnorm_train_lr)"
  fi
  if [[ "$arm" == "filter_only" || "$arm" == "full" ]]; then
    filter_mode="hard_structure_momentum"
  fi
  local filter_schedule="fixed"
  local structural_mode="none"
  if [[ "$filter_mode" == "hard_structure_momentum" ]]; then
    filter_schedule="$(profile_value "$dataset" filter_schedule)"
    structural_mode="$(profile_value "$dataset" structural_mode)"
  fi

  local ssm_tau="0.1"
  local au_uniformity_weight="1.0"
  local au_uniformity_t="2.0"
  local objective_detail
  if [[ "$training_objective" == "ssm" ]]; then
    ssm_tau="$(profile_value "$dataset" ssm_tau)"
    objective_detail="tau=$ssm_tau"
  else
    au_uniformity_weight="$(profile_value "$dataset" au_uniformity_weight)"
    au_uniformity_t="$(profile_value "$dataset" au_uniformity_t)"
    objective_detail="uniformity_weight=$au_uniformity_weight uniformity_t=$au_uniformity_t"
  fi

  echo "Start dataset=$dataset objective=$training_objective arm=$arm noise=$ratio seed=$seed lr=$train_lr $objective_detail"
  DATASET="$dataset" \
  NOISE_MODE=degree_preserving_replace \
  REPLACEMENT_SELECTION=uniform \
  NOISE_RATIOS="$ratio" \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$case_root" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-$(profile_value "$dataset" train_epochs)}" \
  TRAIN_PATIENCE="${TRAIN_PATIENCE:-$(profile_value "$dataset" train_patience)}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$(profile_value "$dataset" train_batch_size)}" \
  TRAIN_LR="$train_lr" \
  TRAIN_INIT_METHOD="$(profile_value "$dataset" train_init_method)" \
  TRAIN_INIT_WEIGHT="$(profile_value "$dataset" train_init_weight)" \
  TRAIN_DECAY="$(profile_value "$dataset" train_decay)" \
  TRAINING_OBJECTIVE="$training_objective" \
  SSM_TAU="$ssm_tau" \
  AU_UNIFORMITY_WEIGHT="$au_uniformity_weight" \
  AU_UNIFORMITY_T="$au_uniformity_t" \
  OBJECTIVE_MESSAGE_DROPOUT="$(profile_value "$dataset" objective_message_dropout)" \
  STOP_AFTER_FILTER=0 \
  SUMMARY_ONLY=1 \
  RUN_PILOT_ANALYSIS=0 \
  KEEP_EDGE_LABELS=0 \
  KEEP_GENERATED_TRAIN=0 \
  STRUCTURAL_MODE="$structural_mode" \
  TOPK="$(profile_value "$dataset" structural_topk)" \
  CHUNK_SIZE="$(profile_value "$dataset" chunk_size)" \
  MIN_DEGREE="${MIN_DEGREE:-2}" \
  EDGE_FILTER_MODE="$filter_mode" \
  RELIABILITY_MOMENTUM_Q="$(profile_value "$dataset" momentum_quantile)" \
  RELIABILITY_STRUCTURE_Q="$(profile_value "$dataset" structure_quantile)" \
  RELIABILITY_STRUCTURE_WEIGHT="$(profile_value "$dataset" structure_weight)" \
  RELIABILITY_MAX_REMOVAL_RATIO="$(profile_value "$dataset" max_removal_ratio)" \
  RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
  RELIABILITY_MOMENTUM_DECAY="$(profile_value "$dataset" momentum_decay)" \
  RELIABILITY_FILTER_SCHEDULE="$filter_schedule" \
  RELIABILITY_FILTER_EPOCH="$(profile_value "$dataset" filter_epoch)" \
  RELIABILITY_ADAPTIVE_MIN_EPOCH="$(profile_value "$dataset" adaptive_min_epoch)" \
  RELIABILITY_ADAPTIVE_MAX_EPOCH="$(profile_value "$dataset" adaptive_max_epoch)" \
  RELIABILITY_ADAPTIVE_MIN_COVERAGE="$(profile_value "$dataset" adaptive_min_coverage)" \
  RELIABILITY_ADAPTIVE_JACCARD="$(profile_value "$dataset" adaptive_jaccard)" \
  RELIABILITY_ADAPTIVE_STABLE_CHECKS="$(profile_value "$dataset" adaptive_stable_checks)" \
  REPRESENTATION_MODULATION_MODE="$modulation_mode" \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA="$modulation_weight" \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
  DRY_RUN="$dry_run" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"

  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$case_root" --output "$completed"
  fi
  echo "Done dataset=$dataset arm=$arm noise=$ratio seed=$seed"
}

total=$((${#dataset_values[@]} * ${#arm_values[@]} * ${#ratio_values[@]} * ${#seed_values[@]}))
echo "$objective_label LightGCN vs full edge-filter + CrossNorm experiment"
echo "  profile:      $profile_file"
echo "  datasets:     $datasets"
echo "  arms:         $arms"
echo "  noise ratios: $noise_ratios"
echo "  seeds:        $seeds"
echo "  GPU:          $gpu_id"
echo "  planned runs: $total"
echo "  output:       $output_root"
for dataset in "${dataset_values[@]}"; do
  objective_profile=""
  if [[ "$training_objective" == "ssm" ]]; then
    objective_profile="tau=$(profile_value "$dataset" ssm_tau)"
  else
    objective_profile="uniformity_weight=$(profile_value "$dataset" au_uniformity_weight), uniformity_t=$(profile_value "$dataset" au_uniformity_t)"
  fi
  echo "Dataset profile $dataset: LightGCN lr=$(profile_value "$dataset" non_crossnorm_train_lr), Full lr=$(profile_value "$dataset" crossnorm_train_lr), $objective_profile, decay=$(profile_value "$dataset" train_decay), mu=$(profile_value "$dataset" modulation_weight), structure_weight=$(profile_value "$dataset" structure_weight), cap=$(profile_value "$dataset" max_removal_ratio), schedule=$(profile_value "$dataset" filter_schedule)"
  for arm in "${arm_values[@]}"; do
    for ratio in "${ratio_values[@]}"; do
      for seed in "${seed_values[@]}"; do
        run_case "$dataset" "$arm" "$ratio" "$seed"
      done
    done
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training or output files were created."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" --output "$output_root/all_runs.json"
python3 "$script_dir/analyze_full_edge_filter_norm.py" \
  --input "$output_root/all_runs.json" \
  --profile "$profile_file" \
  --datasets "${dataset_values[@]}" \
  --arms "${arm_values[@]}" \
  --noise-ratios "${ratio_values[@]}" \
  --seeds "${seed_values[@]}" \
  --output "$output_root/full_edge_filter_norm_summary.json" \
  --markdown "$output_root/full_edge_filter_norm_summary.md"

echo "Full $objective_label experiment completed: $output_root"
echo "  table: $output_root/full_edge_filter_norm_summary.md"
echo "  JSON:  $output_root/full_edge_filter_norm_summary.json"
echo "  runs:  $output_root/all_runs.json"
