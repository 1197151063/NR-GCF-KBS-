#!/usr/bin/env bash
set -euo pipefail

# Complete only the missing clean component-ablation cases:
#   norm_only:  Yelp2018 and Amazon-Book (CrossNorm, no filtering)
#   filter_only: all four datasets (filtering, no CrossNorm)
# for BPR, SSM, and AU. Existing LastFM/ML-1M norm-only cases are not rerun.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_component_ablation_clean_missing}"
gpu_id="${GPU_ID:-0}"
seed="${SEED:-2026}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

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

run_subset() {
  local objective="$1"
  local arm="$2"
  local datasets="$3"
  local runner="$4"
  local profile="$5"
  local subset_root="${output_root%/}/${objective}/${arm}"

  echo "Start missing ablation objective=$objective arm=$arm datasets=$datasets"
  PROFILE_FILE="$profile" \
  OUTPUT_ROOT="$subset_root" \
  DATASETS="$datasets" \
  ARMS="$arm" \
  NOISE_RATIOS=0 \
  SEEDS="$seed" \
  GPU_ID="$gpu_id" \
  DRY_RUN="$dry_run" \
  SKIP_COMPLETED="$skip_completed" \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
    bash "$runner"
  echo "Done missing ablation objective=$objective arm=$arm datasets=$datasets"
}

bpr_runner="$script_dir/run_full_edge_filter_norm_bpr.sh"
objective_runner="$script_dir/run_full_edge_filter_norm_ssm.sh"
bpr_profile="$repo_root/configs/full_bpr_edge_filter_norm.json"
ssm_profile="$repo_root/configs/full_ssm_edge_filter_norm.json"
au_profile="$repo_root/configs/full_au_edge_filter_norm.json"

echo "Missing component-ablation clean experiment"
echo "  planned runs: 18"
echo "  seed:         $seed"
echo "  GPU:          $gpu_id"
echo "  output:       $output_root"
echo "  existing norm-only LastFM/ML-1M cases are intentionally skipped"

run_subset bpr norm_only "yelp2018 amazon-book" "$bpr_runner" "$bpr_profile"
run_subset bpr filter_only "lastfm ml-1m yelp2018 amazon-book" "$bpr_runner" "$bpr_profile"
run_subset ssm norm_only "yelp2018 amazon-book" "$objective_runner" "$ssm_profile"
run_subset ssm filter_only "lastfm ml-1m yelp2018 amazon-book" "$objective_runner" "$ssm_profile"
run_subset au norm_only "yelp2018 amazon-book" "$objective_runner" "$au_profile"
run_subset au filter_only "lastfm ml-1m yelp2018 amazon-book" "$objective_runner" "$au_profile"

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run completed; no training was executed."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/all_missing_runs.json"

python3 "$script_dir/analyze_missing_component_ablation.py" \
  --input "$output_root/all_missing_runs.json" \
  --output "$output_root/missing_component_ablation_summary.json" \
  --markdown "$output_root/missing_component_ablation_summary.md" \
  --seed "$seed"

echo "Missing component ablations completed: $output_root"
echo "  table: $output_root/missing_component_ablation_summary.md"
echo "  JSON:  $output_root/missing_component_ablation_summary.json"
