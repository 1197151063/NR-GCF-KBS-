#!/usr/bin/env bash
set -euo pipefail

# Focused LastFM configuration search in dependency order:
#   A. choose one no-filter CrossNorm blend shared by clean/noisy cases,
#   B. select a conservative 1%--4% hard-filter cap,
#   C. compare ranking signals under the selected cap.
# Defaults use one seed, noise 0/0.2, one GPU, and compact JSON only.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=lastfm
noise_ratios="${NOISE_RATIOS:-0 0.2}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.1_lastfm_complete_pilots}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

norm_weights="${NORM_WEIGHTS:-0.00 0.20 0.40 1.00}"
removal_caps="${REMOVAL_CAPS:-0.01 0.02 0.03 0.04}"
ranking_extra_weights="${RANKING_WEIGHTS:-0.00 0.50 1.00}"
reference_structure_weight="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
clean_tolerance="${CLEAN_RECALL_TOLERANCE:-0.002}"

train_epochs="${TRAIN_EPOCHS:-100}"
train_patience="${TRAIN_PATIENCE:-20}"
adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-5}"
adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-10}"
adaptive_stable_checks="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-2}"

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
read -r -a seed_array <<<"$seeds"
if [[ ${#seed_array[@]} -ne 1 ]]; then
  echo "This focused selector requires exactly one seed; got: $seeds" >&2
  exit 2
fi

python3 - "$noise_ratios" "$norm_weights" "$removal_caps" \
  "$ranking_extra_weights" "$reference_structure_weight" <<'PY'
import math
import sys

for label, text, upper in (
    ("noise ratio", sys.argv[1], None),
    ("norm weight", sys.argv[2], 1.0),
    ("removal cap", sys.argv[3], 1.0),
    ("ranking structure weight", sys.argv[4], 1.0),
    ("reference structure weight", sys.argv[5], 1.0),
):
    for raw in text.split():
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if not math.isfinite(value) or value < 0:
            raise SystemExit("Invalid %s: %s" % (label, raw))
        if upper is not None and value > upper:
            raise SystemExit("Invalid %s: %s" % (label, raw))
ratios = sorted(float(value) for value in sys.argv[1].split())
if ratios != [0.0, 0.2]:
    raise SystemExit(
        "This focused selector requires NOISE_RATIOS='0 0.2'; got %s" % ratios
    )
PY

number_tag() {
  python3 - "$1" <<'PY'
import sys
print(format(float(sys.argv[1]), ".12g").replace("-", "m").replace(".", "p"))
PY
}

run_combo() {
  local label="$1"
  local combo_root="$2"
  local filter_mode="$3"
  local filter_schedule="$4"
  local modulation_weight="$5"
  local removal_cap="$6"
  local structure_weight="$7"
  local completed="$combo_root/comparison_summary.json"

  if [[ "$dry_run" != "1" && "$skip_completed" == "1" && -f "$completed" ]]; then
    echo "Skip completed $label"
    return
  fi
  if [[ "$dry_run" != "1" && -e "$combo_root" ]]; then
    echo "Existing incomplete directory for $label: $combo_root" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT." >&2
    exit 1
  fi

  echo "Start $label"
  DATASET="$dataset" \
  NOISE_MODE=degree_preserving_replace \
  REPLACEMENT_SELECTION=uniform \
  NOISE_RATIOS="$noise_ratios" \
  SEEDS="$seeds" \
  GPU_ID="$gpu_id" \
  OUTPUT_ROOT="$combo_root" \
  TRAIN_EPOCHS="$train_epochs" \
  TRAIN_PATIENCE="$train_patience" \
  TRAIN_LR="${TRAIN_LR:-0.0005}" \
  TRAIN_INIT_WEIGHT="${TRAIN_INIT_WEIGHT:-0.01}" \
  STOP_AFTER_FILTER=0 \
  SUMMARY_ONLY=1 \
  RUN_PILOT_ANALYSIS=0 \
  KEEP_EDGE_LABELS=0 \
  KEEP_GENERATED_TRAIN=0 \
  STRUCTURAL_MODE=two_hop_minhash \
  TOPK="${TOPK:-10}" \
  CHUNK_SIZE="${CHUNK_SIZE:-8192}" \
  MIN_DEGREE="${MIN_DEGREE:-2}" \
  EDGE_FILTER_MODE="$filter_mode" \
  RELIABILITY_MOMENTUM_Q="${RELIABILITY_MOMENTUM_Q:-0.80}" \
  RELIABILITY_STRUCTURE_Q="${RELIABILITY_STRUCTURE_Q:-0.20}" \
  RELIABILITY_STRUCTURE_WEIGHT="$structure_weight" \
  RELIABILITY_MAX_REMOVAL_RATIO="$removal_cap" \
  RELIABILITY_MIN_WEIGHT="${RELIABILITY_MIN_WEIGHT:-0.10}" \
  RELIABILITY_MOMENTUM_DECAY="${RELIABILITY_MOMENTUM_DECAY:-0.90}" \
  RELIABILITY_FILTER_SCHEDULE="$filter_schedule" \
  RELIABILITY_ADAPTIVE_MIN_EPOCH="$adaptive_min_epoch" \
  RELIABILITY_ADAPTIVE_MAX_EPOCH="$adaptive_max_epoch" \
  RELIABILITY_ADAPTIVE_MIN_COVERAGE="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}" \
  RELIABILITY_ADAPTIVE_JACCARD="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}" \
  RELIABILITY_ADAPTIVE_STABLE_CHECKS="$adaptive_stable_checks" \
  REPRESENTATION_MODULATION_MODE=blend_always \
  REPRESENTATION_MODULATION_RAMP_EPOCHS=0 \
  REPRESENTATION_MODULATION_LAMBDA="$modulation_weight" \
  REQUIRE_CLEAN_REPO="${REQUIRE_CLEAN_REPO:-1}" \
  NRGCF_OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}" \
  DRY_RUN="$dry_run" \
    bash "$script_dir/run_edge_diagnostics_grid.sh"

  if [[ "$dry_run" != "1" ]]; then
    python3 "$script_dir/summarize_reliability_runs.py" \
      --root "$combo_root" --output "$completed"
  fi
  echo "Done $label"
}

echo "LastFM complete focused pilots"
echo "  noise ratios:       $noise_ratios"
echo "  seed:               $seeds"
echo "  norm weights:       $norm_weights"
echo "  removal caps:       $removal_caps"
echo "  ranking extra arms: $ranking_extra_weights"
echo "  reference weight:   $reference_structure_weight"
echo "  adaptive window:    ${adaptive_min_epoch}-${adaptive_max_epoch}"
echo "  output:             $output_root"
echo "  planned runs:       22"

# A. No-filter norm selection: eight runs.
for modulation_weight in $norm_weights; do
  tag="$(number_tag "$modulation_weight")"
  run_combo \
    "no-filter norm mu=$modulation_weight" \
    "${output_root%/}/norm_baseline/mu_${tag}" \
    none fixed "$modulation_weight" 1.0 "$reference_structure_weight"
done

if [[ "$dry_run" != "1" ]]; then
  python3 "$script_dir/summarize_reliability_runs.py" \
    --root "$output_root/norm_baseline" \
    --output "$output_root/norm_runs.json"
  python3 "$script_dir/analyze_common_modulation.py" \
    --input "$output_root/norm_runs.json" \
    --dataset "$dataset" \
    --output "$output_root/norm_selection.json" \
    --markdown "$output_root/norm_table.md"
  selected_norm="$(python3 - "$output_root/norm_selection.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
print(format(float(report["selected"]["modulation_lambda"]), ".12g"))
PY
)"
else
  selected_norm="${DRY_SELECTED_NORM_WEIGHT:-0.40}"
  echo "Dry-run provisional norm weight: $selected_norm"
fi
echo "Selected common norm weight: $selected_norm"

# B. Conservative budget search: eight runs.
for removal_cap in $removal_caps; do
  tag="$(number_tag "$removal_cap")"
  run_combo \
    "budget cap=$removal_cap" \
    "${output_root%/}/budget/cap_${tag}" \
    hard_structure_momentum adaptive "$selected_norm" "$removal_cap" \
    "$reference_structure_weight"
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run stops before data-dependent cap selection and ranking arms."
  exit 0
fi

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/preselection_runs.json"
python3 "$script_dir/analyze_movielens_removal_budget.py" \
  --input "$output_root/preselection_runs.json" \
  --dataset "$dataset" \
  --output "$output_root/budget_selection.json" \
  --markdown "$output_root/budget_table.md" \
  --clean-tolerance "$clean_tolerance" \
  --modulation-lambda "$selected_norm"

selected_cap="$(python3 - "$output_root/budget_selection.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
selected = report.get("selected")
if not selected:
    raise SystemExit("Budget selector did not produce a selected cap")
print(format(float(selected["max_removal_ratio"]), ".12g"))
PY
)"
echo "Selected removal cap: $selected_cap"

# C. Extra ranking arms: six runs.  The 0.95 arm is reused from stage B.
for structure_weight in $ranking_extra_weights; do
  tag="$(number_tag "$structure_weight")"
  run_combo \
    "ranking structure weight=$structure_weight cap=$selected_cap" \
    "${output_root%/}/ranking/weight_${tag}" \
    hard_structure_momentum adaptive "$selected_norm" "$selected_cap" \
    "$structure_weight"
done

python3 "$script_dir/summarize_reliability_runs.py" \
  --root "$output_root" \
  --output "$output_root/all_runs.json"

all_ranking_weights="$ranking_extra_weights $reference_structure_weight"
python3 "$script_dir/analyze_ranking_ablation.py" \
  --input "$output_root/all_runs.json" \
  --dataset "$dataset" \
  --removal-cap "$selected_cap" \
  --modulation-lambda "$selected_norm" \
  --weights "$all_ranking_weights" \
  --output "$output_root/ranking_ablation_runs.json" \
  --markdown "$output_root/ranking_table.md"

echo "LastFM focused pilots completed: $output_root"
echo "  norm selection:    $output_root/norm_selection.json"
echo "  norm table:        $output_root/norm_table.md"
echo "  budget selection:  $output_root/budget_selection.json"
echo "  budget table:      $output_root/budget_table.md"
echo "  ranking ablation:  $output_root/ranking_ablation_runs.json"
echo "  ranking table:     $output_root/ranking_table.md"
echo "  all runs:          $output_root/all_runs.json"
