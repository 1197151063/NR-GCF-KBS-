#!/usr/bin/env bash
set -euo pipefail

# Complete the remaining focused MovieLens pilots in dependency order:
#   A. no-filter CrossNorm endpoints (mu=0, 0.2, 1),
#   B. 0.5%--0.8% hard-filter cap refinement at mu=0.2,
#   C. rank-signal ablation at the automatically selected cap.
#
# Defaults run one seed and noise ratios 0/0.2 sequentially on one GPU.  No
# per-edge tables or generated train files are retained.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

noise_ratios="${NOISE_RATIOS:-0 0.2}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-0}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/outputs_v3.0_ml_complete_pilots}"
dry_run="${DRY_RUN:-0}"
skip_completed="${SKIP_COMPLETED:-1}"

norm_baseline_weights="${NORM_BASELINE_WEIGHTS:-0.00 0.20 1.00}"
removal_caps="${REMOVAL_CAPS:-0.005 0.006 0.007 0.008}"
ranking_weights="${RANKING_WEIGHTS:-0.00 0.50 1.00}"
selected_norm="${SELECTED_NORM_WEIGHT:-0.20}"
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

python3 - "$noise_ratios" "$norm_baseline_weights" "$removal_caps" \
  "$ranking_weights" "$selected_norm" "$reference_structure_weight" <<'PY'
import math
import sys

for label, text, upper in (
    ("noise ratio", sys.argv[1], None),
    ("norm weight", sys.argv[2], 1.0),
    ("removal cap", sys.argv[3], 1.0),
    ("ranking structure weight", sys.argv[4], 1.0),
    ("selected norm", sys.argv[5], 1.0),
    ("reference structure weight", sys.argv[6], 1.0),
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
  DATASET=ml-1m \
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

echo "MovieLens complete focused pilots"
echo "  noise ratios:         $noise_ratios"
echo "  seed:                 $seeds"
echo "  no-filter norm arms:  $norm_baseline_weights"
echo "  removal caps:         $removal_caps"
echo "  ranking extra arms:   $ranking_weights"
echo "  selected norm:        $selected_norm"
echo "  reference rank weight:$reference_structure_weight"
echo "  adaptive window:      ${adaptive_min_epoch}-${adaptive_max_epoch}"
echo "  output:               $output_root"
echo "  planned runs:         20"

# A. Norm effect without filtering: six runs.
for modulation_weight in $norm_baseline_weights; do
  tag="$(number_tag "$modulation_weight")"
  run_combo \
    "no-filter norm mu=$modulation_weight" \
    "${output_root%/}/norm_baseline/mu_${tag}" \
    none fixed "$modulation_weight" 1.0 "$reference_structure_weight"
done

# B. Cap refinement under the selected norm and 95/5 ranking: eight runs.
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
echo "Selected removal cap for rank ablation: $selected_cap"

# C. Extra ranking arms.  The selected cap's 0.95 arm is reused from stage B.
for structure_weight in $ranking_weights; do
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

python3 - "$output_root/all_runs.json" "$output_root/budget_selection.json" \
  "$output_root/ranking_ablation_runs.json" "$output_root/ranking_table.md" \
  "$ranking_weights" "$reference_structure_weight" <<'PY'
import json
import math
import sys

all_path, selection_path, output_path, table_path, extra_text, reference = sys.argv[1:]
with open(all_path, encoding="utf-8") as stream:
    report = json.load(stream)
with open(selection_path, encoding="utf-8") as stream:
    selection = json.load(stream)
selected_cap = float(selection["selected"]["max_removal_ratio"])
weights = sorted(set(
    [float(value) for value in extra_text.split()] + [float(reference)]
))
rows = []
seen = set()
for row in report.get("runs", []):
    if row.get("mode") != "hard_structure_momentum":
        continue
    cap = row.get("max_removal_ratio")
    weight = row.get("structure_weight")
    if cap is None or weight is None:
        continue
    if not math.isclose(float(cap), selected_cap, rel_tol=0.0, abs_tol=1e-12):
        continue
    if not any(math.isclose(float(weight), target, rel_tol=0.0, abs_tol=1e-12)
               for target in weights):
        continue
    identity = (float(row["requested_noise_ratio"]), float(weight), int(row["seed"]))
    if identity in seen:
        raise SystemExit("Duplicate ranking identity: %r" % (identity,))
    seen.add(identity)
    rows.append(row)

expected = 2 * len(weights)
if len(rows) != expected:
    raise SystemExit("Expected %d ranking rows, found %d" % (expected, len(rows)))
rows.sort(key=lambda row: (
    float(row["requested_noise_ratio"]), float(row["structure_weight"])
))
compact = {
    "schema_version": "nrgcf_movielens_ranking_ablation_v1",
    "selected_max_removal_ratio": selected_cap,
    "weight_semantics": (
        "risk=w_s*(1-structure_rank)+(1-w_s)*momentum_rank; the same "
        "selected cap fixes removal count across ranking arms"
    ),
    "runs": rows,
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(compact, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")

lines = [
    "# MovieLens ranking-signal ablation",
    "",
    "Selected removal cap: `%.4g`." % selected_cap,
    "",
    "| Noise | Structure weight | Recall@20 | NDCG@20 | Removed | "
    "Noisy removal | Precision |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    noisy_rate = row.get("noisy_removal_rate")
    precision = row.get("removed_precision_noisy")
    lines.append(
        "| {noise:.3g} | {weight:.2f} | {recall:.6f} | {ndcg:.6f} | "
        "{removed:.4f} | {noisy} | {precision} |".format(
            noise=float(row["requested_noise_ratio"]),
            weight=float(row["structure_weight"]),
            recall=float(row["best_recall_at_20"]),
            ndcg=float(row["best_ndcg_at_20"]),
            removed=float(row["removed_ratio"]),
            noisy="--" if noisy_rate is None else "%.4f" % float(noisy_rate),
            precision="--" if precision is None else "%.4f" % float(precision),
        )
    )
with open(table_path, "w", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
PY

echo "MovieLens focused pilots completed: $output_root"
echo "  all runs:          $output_root/all_runs.json"
echo "  budget selection:  $output_root/budget_selection.json"
echo "  budget table:      $output_root/budget_table.md"
echo "  ranking ablation:  $output_root/ranking_ablation_runs.json"
echo "  ranking table:     $output_root/ranking_table.md"
