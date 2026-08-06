#!/usr/bin/env bash
set -euo pipefail

# Run an NR-GCF diagnostics grid over synthetic-noise train splits.
#
# Required layout for non-zero noise ratios:
#   $NOISE_DATA_ROOT/$DATASET/noise_<ratio>/seed_<seed>/train.txt
#
# Example:
#   DATASET=yelp2018 \
#   NOISE_RATIOS="0 0.05 0.10 0.20" \
#   SEEDS="2026 2027 2028" \
#   GPU_ID=0 \
#   NOISE_DATA_ROOT=/path/to/prepared_noise \
#   OUTPUT_ROOT=/path/to/results \
#   bash run_edge_diagnostics_grid.sh
#
# Degree-preserving replacement protocol (recommended structural pilot):
#   DATASET=yelp2018 \
#   NOISE_MODE=degree_preserving_replace \
#   NOISE_RATIOS="0.10" \
#   SEEDS="2026" \
#   GPU_ID=0 \
#   OUTPUT_ROOT=/path/to/results \
#   bash run_edge_diagnostics_grid.sh
#
# Built-in training-only random-nonedge protocol (explicit opt-in):
#   DATASET=yelp2018 \
#   NOISE_MODE=uniform_train_nonedge \
#   NOISE_RATIOS="0 0.05 0.10 0.20" \
#   SEEDS="2026 2027 2028" \
#   GPU_ID=0 \
#   OUTPUT_ROOT=/path/to/results \
#   bash run_edge_diagnostics_grid.sh
#
# In additive modes, ratio 0 uses the repository's clean train.txt unless a
# prepared ratio-0 split exists. --requested-noise-ratio remains metadata-only
# in NR-GCF; this launcher verifies and installs the corresponding train split
# in an isolated temporary Git worktree before invoking the existing entry.

usage() {
  cat <<'EOF'
Run edge diagnostics for a grid of noise ratios and random seeds.

Environment variables:
  DATASET          Dataset name (default: yelp2018)
  NOISE_RATIOS     Space-separated ratios (default: "0.10")
  SEEDS            Space-separated seeds (default: "2026")
  GPU_ID           CUDA device visible to each run (required unless DRY_RUN=1)
  NOISE_DATA_ROOT  Root of prepared noisy splits (required in prepared mode)
  OUTPUT_ROOT      New root directory for run outputs (required)

Optional variables:
  NOISE_MODE             prepared: consume externally prepared train.txt files;
                         uniform_train_nonedge: generate unique random training
                         non-edges without reading validation/test data;
                         degree_preserving_replace: swap item endpoints between
                         pairs of edges, preserving every user/item degree
                         (default: prepared)
  STOP_AFTER_FILTER       1: stop after diagnostics export; 0: continue training
                          (default: 1)
  DIAGNOSTICS_FORMAT      parquet, csv, or csv_gzip (default: parquet)
  STRUCTURAL_MODE         two_hop_minhash or none (default: two_hop_minhash)
  TOPK                    Structural top-k (default: 10)
  CHUNK_SIZE              Export chunk size (default: 8192)
  MIN_DEGREE              Connectivity-risk degree threshold (default: 2)
  NRGCF_OMP_NUM_THREADS   OpenMP threads (default: 4)
  REQUIRE_CLEAN_REPO      Refuse a tracked dirty source tree (default: 1)
  DRY_RUN                 1: validate paths and print the grid only (default: 0)
  REPLACEMENT_SELECTION   uniform or hard_two_hop (default: uniform)
  HARD_CANDIDATE_POOL     candidates ranked per hard swap (default: 8)
  HARD_SUPPORT_LIMIT      bounded same-type neighbors per side (default: 16)
  RUN_PILOT_ANALYSIS      write pilot_analysis.json after export (default: 1)
  TRAIN_EPOCHS            optional total epoch override, e.g. 17 for two
                          post-filter smoke epochs (default: entry default)
  TRAIN_PATIENCE          optional early-stopping patience override
  TRAIN_LR                optional learning-rate override
  TRAIN_INIT_WEIGHT       optional embedding initialization std override
  EDGE_FILTER_MODE        current, none, hard_consensus, hard_structure_only,
                          soft_reliability, gated_soft_reliability, or
                          hard_structure_momentum
                          (default: current)
  REPRESENTATION_MODULATION_MODE
                          none, legacy_always, original_always, blend_always,
                          original_stage_two, reliability_weighted_always,
                          paper_stage_two (alias), or
                          reliability_weighted_stage_two
                          (default: original_stage_two)
  REPRESENTATION_MODULATION_RAMP_EPOCHS
                          post-filter linear transition length (default: 0)
  REPRESENTATION_MODULATION_LAMBDA
                          propagation/CrossNorm blend weight in [0,1] used by
                          blend_always; direct modes ignore it (default: 1)
  SUMMARY_ONLY            1: write compact reliability JSON and no per-edge
                          CSV/Parquet (default: 0)
  KEEP_EDGE_LABELS        retain the temporary synthetic-label CSV (default: 1)
  KEEP_GENERATED_TRAIN    retain generated_train.txt after the run (default: 1)
  RELIABILITY_MOMENTUM_Q  high-momentum quantile (default: 0.80)
  RELIABILITY_STRUCTURE_Q low-structure quantile (default: 0.20)
  RELIABILITY_STRUCTURE_WEIGHT soft structural rank weight (default: 0.95)
  RELIABILITY_MIN_WEIGHT  minimum soft propagation weight (default: 0.10)
  RELIABILITY_FILTER_EPOCH filtering epoch for hard_structure_momentum
                          (default: 15)
  RELIABILITY_FILTER_SCHEDULE fixed or adaptive (default: fixed)
  RELIABILITY_ADAPTIVE_MIN_EPOCH earliest adaptive check (default: 5)
  RELIABILITY_ADAPTIVE_MAX_EPOCH forced adaptive filter epoch (default: 10)
  RELIABILITY_ADAPTIVE_MIN_COVERAGE required observed-edge fraction
                          (default: 0.99)
  RELIABILITY_ADAPTIVE_JACCARD removed-set stability threshold (default: 0.90)
  RELIABILITY_ADAPTIVE_STABLE_CHECKS consecutive stable checks (default: 2)
  RELIABILITY_MOMENTUM_DECAY stable EMA decay (default: 0.90)

Prepared additive split layout:
  NOISE_DATA_ROOT/DATASET/noise_RATIO/seed_SEED/train.txt

In prepared mode, every noisy train.txt must contain all clean training edges
plus the requested number of unique injected edges. Replacement mode generates
and validates its own fixed-size split and does not use NOISE_DATA_ROOT.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "This launcher uses environment variables and accepts no arguments." >&2
  usage >&2
  exit 2
fi

dataset="${DATASET:-yelp2018}"
noise_ratios="${NOISE_RATIOS:-0.10}"
seeds="${SEEDS:-2026}"
gpu_id="${GPU_ID:-}"
noise_data_root="${NOISE_DATA_ROOT:-}"
noise_mode="${NOISE_MODE:-prepared}"
output_root="${OUTPUT_ROOT:-}"
stop_after_filter="${STOP_AFTER_FILTER:-1}"
diagnostics_format="${DIAGNOSTICS_FORMAT:-parquet}"
structural_mode="${STRUCTURAL_MODE:-two_hop_minhash}"
topk="${TOPK:-10}"
chunk_size="${CHUNK_SIZE:-8192}"
min_degree="${MIN_DEGREE:-2}"
omp_threads="${NRGCF_OMP_NUM_THREADS:-4}"
require_clean_repo="${REQUIRE_CLEAN_REPO:-1}"
dry_run="${DRY_RUN:-0}"
replacement_selection="${REPLACEMENT_SELECTION:-uniform}"
hard_candidate_pool="${HARD_CANDIDATE_POOL:-8}"
hard_support_limit="${HARD_SUPPORT_LIMIT:-16}"
run_pilot_analysis="${RUN_PILOT_ANALYSIS:-1}"
train_epochs="${TRAIN_EPOCHS:-}"
train_patience="${TRAIN_PATIENCE:-}"
train_lr="${TRAIN_LR:-}"
train_init_weight="${TRAIN_INIT_WEIGHT:-}"
edge_filter_mode="${EDGE_FILTER_MODE:-current}"
summary_only="${SUMMARY_ONLY:-0}"
keep_edge_labels="${KEEP_EDGE_LABELS:-1}"
keep_generated_train="${KEEP_GENERATED_TRAIN:-1}"
reliability_momentum_q="${RELIABILITY_MOMENTUM_Q:-0.80}"
reliability_structure_q="${RELIABILITY_STRUCTURE_Q:-0.20}"
reliability_structure_weight="${RELIABILITY_STRUCTURE_WEIGHT:-0.95}"
reliability_min_weight="${RELIABILITY_MIN_WEIGHT:-0.10}"
reliability_filter_epoch="${RELIABILITY_FILTER_EPOCH:-15}"
reliability_filter_schedule="${RELIABILITY_FILTER_SCHEDULE:-fixed}"
reliability_adaptive_min_epoch="${RELIABILITY_ADAPTIVE_MIN_EPOCH:-5}"
reliability_adaptive_max_epoch="${RELIABILITY_ADAPTIVE_MAX_EPOCH:-10}"
reliability_adaptive_min_coverage="${RELIABILITY_ADAPTIVE_MIN_COVERAGE:-0.99}"
reliability_adaptive_jaccard="${RELIABILITY_ADAPTIVE_JACCARD:-0.90}"
reliability_adaptive_stable_checks="${RELIABILITY_ADAPTIVE_STABLE_CHECKS:-2}"
reliability_momentum_decay="${RELIABILITY_MOMENTUM_DECAY:-0.90}"
representation_modulation_mode="${REPRESENTATION_MODULATION_MODE:-original_stage_two}"
representation_modulation_ramp_epochs="${REPRESENTATION_MODULATION_RAMP_EPOCHS:-0}"
representation_modulation_lambda="${REPRESENTATION_MODULATION_LAMBDA:-1}"

for binary_flag in stop_after_filter require_clean_repo dry_run run_pilot_analysis summary_only keep_edge_labels keep_generated_train; do
  value="${!binary_flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "${binary_flag^^} must be 0 or 1 (got: $value)" >&2
    exit 2
  fi
done
if [[ "$edge_filter_mode" != "current" && "$edge_filter_mode" != "none" && \
      "$edge_filter_mode" != "hard_consensus" && \
      "$edge_filter_mode" != "hard_structure_only" && \
      "$edge_filter_mode" != "soft_reliability" && \
      "$edge_filter_mode" != "gated_soft_reliability" && \
      "$edge_filter_mode" != "hard_structure_momentum" ]]; then
  echo "Unsupported EDGE_FILTER_MODE: $edge_filter_mode" >&2
  exit 2
fi
if ! [[ "$reliability_filter_epoch" =~ ^[0-9]+$ ]] || \
   [[ "$reliability_filter_epoch" -lt 2 ]]; then
  echo "RELIABILITY_FILTER_EPOCH must be an integer >= 2." >&2
  exit 2
fi
if [[ "$reliability_filter_schedule" != "fixed" && \
      "$reliability_filter_schedule" != "adaptive" ]]; then
  echo "RELIABILITY_FILTER_SCHEDULE must be fixed or adaptive." >&2
  exit 2
fi
if [[ "$reliability_filter_schedule" == "adaptive" && \
      "$edge_filter_mode" != "hard_structure_momentum" ]]; then
  echo "Adaptive filtering requires EDGE_FILTER_MODE=hard_structure_momentum." >&2
  exit 2
fi
if ! [[ "$reliability_adaptive_min_epoch" =~ ^[0-9]+$ ]] || \
   [[ "$reliability_adaptive_min_epoch" -lt 2 ]]; then
  echo "RELIABILITY_ADAPTIVE_MIN_EPOCH must be an integer >= 2." >&2
  exit 2
fi
if ! [[ "$reliability_adaptive_max_epoch" =~ ^[0-9]+$ ]] || \
   [[ "$reliability_adaptive_max_epoch" -lt "$reliability_adaptive_min_epoch" ]]; then
  echo "RELIABILITY_ADAPTIVE_MAX_EPOCH must be >= adaptive min epoch." >&2
  exit 2
fi
if ! [[ "$reliability_adaptive_stable_checks" =~ ^[0-9]+$ ]] || \
   [[ "$reliability_adaptive_stable_checks" -lt 1 ]]; then
  echo "RELIABILITY_ADAPTIVE_STABLE_CHECKS must be positive." >&2
  exit 2
fi
for value in "$reliability_adaptive_min_coverage" "$reliability_adaptive_jaccard"; do
  if ! python3 - "$value" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
  then
    echo "Adaptive coverage and Jaccard values must be within [0,1]." >&2
    exit 2
  fi
done
if [[ "$representation_modulation_mode" != "none" && \
      "$representation_modulation_mode" != "legacy_always" && \
      "$representation_modulation_mode" != "original_always" && \
      "$representation_modulation_mode" != "blend_always" && \
      "$representation_modulation_mode" != "original_stage_two" && \
      "$representation_modulation_mode" != "paper_stage_two" && \
      "$representation_modulation_mode" != "reliability_weighted_always" && \
      "$representation_modulation_mode" != "reliability_weighted_stage_two" ]]; then
  echo "Unsupported REPRESENTATION_MODULATION_MODE: $representation_modulation_mode" >&2
  exit 2
fi
if ! [[ "$representation_modulation_ramp_epochs" =~ ^[0-9]+$ ]]; then
  echo "REPRESENTATION_MODULATION_RAMP_EPOCHS must be a non-negative integer." >&2
  exit 2
fi
if ! python3 - "$representation_modulation_lambda" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
then
  echo "REPRESENTATION_MODULATION_LAMBDA must be within [0,1]." >&2
  exit 2
fi
if [[ ( "$representation_modulation_mode" == "reliability_weighted_always" || \
        "$representation_modulation_mode" == "reliability_weighted_stage_two" ) && \
      "$edge_filter_mode" != "hard_structure_momentum" ]]; then
  echo "Reliability-weighted modulation requires EDGE_FILTER_MODE=hard_structure_momentum." >&2
  exit 2
fi
if [[ -z "$output_root" ]]; then
  echo "OUTPUT_ROOT is required." >&2
  exit 2
fi
if [[ "$dry_run" != "1" && -z "$gpu_id" ]]; then
  echo "GPU_ID is required unless DRY_RUN=1." >&2
  exit 2
fi
if [[ "$diagnostics_format" != "parquet" && "$diagnostics_format" != "csv" && \
      "$diagnostics_format" != "csv_gzip" ]]; then
  echo "DIAGNOSTICS_FORMAT must be parquet, csv, or csv_gzip." >&2
  exit 2
fi
if [[ "$replacement_selection" != "uniform" && \
      "$replacement_selection" != "hard_two_hop" ]]; then
  echo "REPLACEMENT_SELECTION must be uniform or hard_two_hop." >&2
  exit 2
fi
if [[ -n "$train_epochs" ]] && \
   { ! [[ "$train_epochs" =~ ^[0-9]+$ ]] || [[ "$train_epochs" -lt 15 ]]; }; then
  echo "TRAIN_EPOCHS must be an integer >= 15 when provided." >&2
  exit 2
fi
if [[ -n "$train_patience" ]] && \
   { ! [[ "$train_patience" =~ ^[0-9]+$ ]] || [[ "$train_patience" -lt 1 ]]; }; then
  echo "TRAIN_PATIENCE must be a positive integer when provided." >&2
  exit 2
fi
if [[ "$edge_filter_mode" == "hard_structure_momentum" && \
      -n "$train_epochs" ]]; then
  if [[ "$reliability_filter_schedule" == "fixed" && \
        "$reliability_filter_epoch" -gt "$train_epochs" ]]; then
    echo "RELIABILITY_FILTER_EPOCH cannot exceed TRAIN_EPOCHS." >&2
    exit 2
  fi
  if [[ "$reliability_filter_schedule" == "adaptive" && \
        "$reliability_adaptive_max_epoch" -gt "$train_epochs" ]]; then
    echo "RELIABILITY_ADAPTIVE_MAX_EPOCH cannot exceed TRAIN_EPOCHS." >&2
    exit 2
  fi
fi
if [[ "$structural_mode" != "two_hop_minhash" && "$structural_mode" != "none" ]]; then
  echo "STRUCTURAL_MODE must be two_hop_minhash or none." >&2
  exit 2
fi
if [[ "$noise_mode" != "prepared" && \
      "$noise_mode" != "uniform_train_nonedge" && \
      "$noise_mode" != "degree_preserving_replace" ]]; then
  echo "NOISE_MODE must be prepared, uniform_train_nonedge, or degree_preserving_replace." >&2
  exit 2
fi

invocation_dir="$(pwd)"
if [[ "$output_root" != /* ]]; then
  output_root="$invocation_dir/$output_root"
fi
if [[ -n "$noise_data_root" && "$noise_data_root" != /* ]]; then
  noise_data_root="$invocation_dir/$noise_data_root"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
clean_train="$repo_root/data/$dataset/train.txt"
if [[ ! -f "$clean_train" ]]; then
  echo "Clean training split not found: $clean_train" >&2
  exit 2
fi
if ! git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "The repository must be a Git worktree: $repo_root" >&2
  exit 2
fi
if [[ "$require_clean_repo" == "1" ]] && \
   [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked repository files are dirty; commit/stash them or set REQUIRE_CLEAN_REPO=0." >&2
  git -C "$repo_root" status --short --untracked-files=no >&2
  exit 2
fi

commit_hash="$(git -C "$repo_root" rev-parse HEAD)"
temporary_parent=""
active_worktree=""

cleanup() {
  if [[ -n "$active_worktree" && -d "$active_worktree" ]]; then
    git -C "$repo_root" worktree remove --force "$active_worktree" >/dev/null 2>&1 || true
  fi
  if [[ -n "$temporary_parent" && -d "$temporary_parent" ]]; then
    rm -rf -- "$temporary_parent"
  fi
}
trap cleanup EXIT INT TERM

ratio_is_zero() {
  python3 - "$1" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(2)
raise SystemExit(0 if value == 0.0 else 1)
PY
}

ratio_tag() {
  python3 - "$1" <<'PY'
import sys
value = float(sys.argv[1])
text = format(value, ".12g")
print(text.replace("-", "m").replace(".", "p"))
PY
}

prepared_train_path() {
  local ratio="$1"
  local seed="$2"
  if [[ -z "$noise_data_root" ]]; then
    return 1
  fi
  printf '%s/%s/noise_%s/seed_%s/train.txt\n' \
    "${noise_data_root%/}" "$dataset" "$ratio" "$seed"
}

resolve_train_path() {
  local ratio="$1"
  local seed="$2"
  local candidate=""
  if candidate="$(prepared_train_path "$ratio" "$seed")" && [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if ratio_is_zero "$ratio"; then
    printf '%s\n' "$clean_train"
    return 0
  fi
  echo "Missing prepared noisy split for ratio=$ratio seed=$seed" >&2
  if [[ -n "$candidate" ]]; then
    echo "Expected: $candidate" >&2
  else
    echo "Set NOISE_DATA_ROOT and use the documented directory layout." >&2
  fi
  return 1
}

generate_uniform_train_nonedges() {
  local requested_ratio="$1"
  local seed="$2"
  local destination="$3"
  local generation_metadata="$4"
  python3 - "$clean_train" "$requested_ratio" "$seed" \
    "$destination" "$generation_metadata" <<'PY'
import json
import math
import pathlib
import random
import sys

clean_path = pathlib.Path(sys.argv[1])
ratio = float(sys.argv[2])
seed = int(sys.argv[3])
destination = pathlib.Path(sys.argv[4])
metadata_path = pathlib.Path(sys.argv[5])
if not math.isfinite(ratio) or ratio < 0:
    raise SystemExit("Noise ratio must be finite and non-negative")

clean_text = clean_path.read_text(encoding="utf-8")
clean_edges = set()
users = set()
items = set()
for line_number, line in enumerate(clean_text.splitlines(), 1):
    fields = line.split()
    if not fields:
        continue
    try:
        user = int(fields[0])
        row_items = [int(value) for value in fields[1:]]
    except ValueError as exc:
        raise SystemExit(f"Non-integer ID in {clean_path}:{line_number}: {exc}")
    users.add(user)
    for item in row_items:
        edge = (user, item)
        if edge in clean_edges:
            raise SystemExit(f"Duplicate clean edge {edge} at line {line_number}")
        clean_edges.add(edge)
        items.add(item)

ordered_users = sorted(users)
ordered_items = sorted(items)
target = int(round(ratio * len(clean_edges)))
available = len(ordered_users) * len(ordered_items) - len(clean_edges)
if target > available:
    raise SystemExit(
        f"Requested {target} noise edges but only {available} training non-edges exist"
    )

rng = random.Random(seed)
injected = set()
while len(injected) < target:
    edge = (rng.choice(ordered_users), rng.choice(ordered_items))
    if edge not in clean_edges:
        injected.add(edge)

destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", encoding="utf-8") as handle:
    handle.write(clean_text)
    if clean_text and not clean_text.endswith("\n"):
        handle.write("\n")
    # Appending one edge per line preserves every original clean edge_id.
    for user, item in sorted(injected):
        handle.write(f"{user} {item}\n")

metadata = {
    "generator": "uniform_train_nonedge",
    "seed": seed,
    "requested_noise_ratio": ratio,
    "noise_ratio_definition": "round(ratio * clean_edge_count) / clean_edge_count",
    "clean_edge_count": len(clean_edges),
    "injected_edge_count": len(injected),
    "user_sampling": "uniform over user IDs observed in clean train.txt",
    "item_sampling": "uniform over item IDs observed in clean train.txt",
    "exclusion": "unique edges present in clean train.txt only",
    "test_data_read": False,
    "held_out_positive_overlap_status": "unknown because test data is intentionally not read",
    "edge_order": "clean file unchanged, followed by injected edges sorted by (user,item)",
}
metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metadata, sort_keys=True))
PY
}

validate_split_and_write_labels() {
  local variant_train="$1"
  local requested_ratio="$2"
  local labels_path="$3"
  local validation_path="$4"
  local noise_type="$5"
  python3 - "$clean_train" "$variant_train" "$requested_ratio" \
    "$labels_path" "$validation_path" "$noise_type" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import sys

clean_path = pathlib.Path(sys.argv[1])
variant_path = pathlib.Path(sys.argv[2])
requested = float(sys.argv[3])
labels_path = pathlib.Path(sys.argv[4])
validation_path = pathlib.Path(sys.argv[5])
noise_type = sys.argv[6]
if not math.isfinite(requested) or requested < 0:
    raise SystemExit("Noise ratio must be finite and non-negative")

def read_edges(path):
    ordered = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                user = int(fields[0])
                items = [int(value) for value in fields[1:]]
            except ValueError as exc:
                raise SystemExit(f"Non-integer ID in {path}:{line_number}: {exc}")
            for item in items:
                edge = (user, item)
                if edge in seen:
                    raise SystemExit(f"Duplicate edge {edge} in {path}:{line_number}")
                seen.add(edge)
                ordered.append(edge)
    if not ordered:
        raise SystemExit(f"No training edges found in {path}")
    return ordered, seen

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

clean_ordered, clean = read_edges(clean_path)
variant_ordered, variant = read_edges(variant_path)
missing = clean - variant
if missing:
    example = next(iter(missing))
    raise SystemExit(
        f"Prepared split removed {len(missing)} clean edges; example={example}. "
        "Only additive synthetic-noise splits are accepted."
    )
injected = variant - clean
actual = len(injected) / len(clean)
tolerance = max(1.0 / len(clean), 1e-9)
if abs(actual - requested) > tolerance:
    raise SystemExit(
        f"Noise ratio mismatch: requested={requested}, actual={actual}, "
        f"clean_edges={len(clean)}, injected_edges={len(injected)}"
    )

labels_path.parent.mkdir(parents=True, exist_ok=True)
with labels_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "edge_id",
        "user_id_internal",
        "item_id_internal",
        "is_original_observed_edge",
        "synthetic_is_noisy",
        "synthetic_noise_type",
    ])
    for edge_id, (user, item) in enumerate(variant_ordered):
        is_original = (user, item) in clean
        writer.writerow([
            edge_id,
            user,
            item,
            is_original,
            not is_original,
            "" if is_original else noise_type,
        ])

validation = {
    "requested_noise_ratio": requested,
    "actual_noise_ratio": actual,
    "noise_ratio_definition": "injected_unique_edges / original_clean_edges",
    "clean_edge_count": len(clean),
    "variant_edge_count": len(variant),
    "injected_edge_count": len(injected),
    "missing_clean_edge_count": 0,
    "duplicate_edge_count": 0,
    "clean_train_sha256": sha256(clean_path),
    "variant_train_sha256": sha256(variant_path),
    "label_join_key": "edge_id equals the variant train.txt loader order",
    "synthetic_noise_type": noise_type,
    "label_leakage_note": "Labels are generated from clean-vs-variant membership and are not read by NR-GCF training or feature computation.",
}
validation_path.write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(validation, sort_keys=True))
PY
}

echo "NR-GCF diagnostics grid"
echo "  repository: $repo_root"
echo "  commit:     $commit_hash"
echo "  dataset:    $dataset"
echo "  noise mode: $noise_mode"
echo "  ratios:     $noise_ratios"
echo "  seeds:      $seeds"
echo "  output:     $output_root"
echo "  replacement selection: $replacement_selection"
echo "  edge filter: $edge_filter_mode"
echo "  reliability filter epoch: $reliability_filter_epoch"
echo "  reliability filter schedule: $reliability_filter_schedule"
if [[ "$reliability_filter_schedule" == "adaptive" ]]; then
  echo "  adaptive window: ${reliability_adaptive_min_epoch}-${reliability_adaptive_max_epoch}"
  echo "  adaptive coverage/Jaccard: ${reliability_adaptive_min_coverage}/${reliability_adaptive_jaccard}"
  echo "  adaptive stable checks: $reliability_adaptive_stable_checks"
fi
echo "  representation modulation: $representation_modulation_mode"
echo "  representation ramp epochs: $representation_modulation_ramp_epochs"
echo "  representation lambda: $representation_modulation_lambda"
echo "  reliability momentum quantile: $reliability_momentum_q"
echo "  reliability structure quantile: $reliability_structure_q"
echo "  reliability structure weight: $reliability_structure_weight"
echo "  train lr: ${train_lr:-entry_default}"
echo "  train init weight: ${train_init_weight:-entry_default}"
echo "  summary only: $summary_only"
echo "  dry run:    $dry_run"

for ratio in $noise_ratios; do
  # Validate numeric syntax before using it in paths or metadata.
  tag="$(ratio_tag "$ratio")"
  for seed in $seeds; do
    if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
      echo "Invalid integer seed: $seed" >&2
      exit 2
    fi
    if [[ "$noise_mode" == "degree_preserving_replace" ]]; then
      if [[ "$replacement_selection" == "hard_two_hop" ]]; then
        run_name="hard_replace_noise_${tag}_seed_${seed}"
      else
        run_name="replace_noise_${tag}_seed_${seed}"
      fi
    else
      run_name="noise_${tag}_seed_${seed}"
    fi
    if [[ "$edge_filter_mode" != "current" ]]; then
      run_name="${edge_filter_mode}_${run_name}"
    fi
    if [[ "$edge_filter_mode" == "hard_structure_momentum" ]]; then
      if [[ "$reliability_filter_schedule" == "adaptive" ]]; then
        run_name="${run_name}_filter_adaptive_${reliability_adaptive_min_epoch}_${reliability_adaptive_max_epoch}"
      else
        run_name="${run_name}_filter_${reliability_filter_epoch}"
      fi
    fi
    run_name="${run_name}_mod_${representation_modulation_mode}"
    run_dir="${output_root%/}/$dataset/$run_name"
    noise_type="prepared_additive_nonedge"
    if [[ "$noise_mode" == "prepared" ]]; then
      variant_train="$(resolve_train_path "$ratio" "$seed")"
    elif [[ "$noise_mode" == "degree_preserving_replace" ]]; then
      variant_train="$run_dir/generated_train.txt"
      noise_type="degree_preserving_edge_swap"
    elif ratio_is_zero "$ratio"; then
      variant_train="$clean_train"
      noise_type="uniform_train_nonedge"
    else
      variant_train="$run_dir/generated_train.txt"
      noise_type="uniform_train_nonedge"
    fi
    echo "[$run_name] train=$variant_train output=$run_dir"
    if [[ "$dry_run" == "1" ]]; then
      continue
    fi
    if [[ -e "$run_dir" ]]; then
      echo "Refusing to overwrite existing run directory: $run_dir" >&2
      exit 2
    fi
    mkdir -p "$run_dir"

    labels_already_written="0"
    if [[ "$noise_mode" == "degree_preserving_replace" ]]; then
      python3 "$script_dir/generate_degree_preserving_replace.py" \
        --clean-train "$clean_train" \
        --noise-ratio "$ratio" \
        --seed "$seed" \
        --output-train "$variant_train" \
        --labels "$run_dir/synthetic_edge_labels.csv" \
        --generation-metadata "$run_dir/noise_generation.json" \
        --validation "$run_dir/noise_validation.json" \
        --selection "$replacement_selection" \
        --candidate-pool-size "$hard_candidate_pool" \
        --structural-support-limit "$hard_support_limit" \
        | tee "$run_dir/noise_generation.log"
      labels_already_written="1"
    elif [[ "$noise_mode" == "uniform_train_nonedge" ]] && ! ratio_is_zero "$ratio"; then
      generate_uniform_train_nonedges \
        "$ratio" "$seed" "$variant_train" "$run_dir/noise_generation.json" \
        | tee "$run_dir/noise_generation.log"
    fi

    if [[ "$labels_already_written" == "0" ]]; then
      validate_split_and_write_labels \
        "$variant_train" "$ratio" \
        "$run_dir/synthetic_edge_labels.csv" \
        "$run_dir/noise_validation.json" "$noise_type" \
        | tee "$run_dir/noise_validation.log"
    else
      python3 -m json.tool "$run_dir/noise_validation.json" \
        | tee "$run_dir/noise_validation.log"
    fi

    temporary_parent="$(mktemp -d "${TMPDIR:-/tmp}/nrgcf-grid.XXXXXX")"
    active_worktree="$temporary_parent/worktree"
    git -C "$repo_root" worktree add --detach "$active_worktree" "$commit_hash" \
      >"$run_dir/worktree_setup.log" 2>&1
    cp -- "$variant_train" "$active_worktree/data/$dataset/train.txt"

    command=(
      python NR-GCF.py
      --dataset "$dataset"
      --seed "$seed"
      --requested-noise-ratio "$ratio"
      --edge-filter-mode "$edge_filter_mode"
      --edge-diagnostics-structural-mode "$structural_mode"
      --edge-diagnostics-topk "$topk"
      --edge-diagnostics-chunk-size "$chunk_size"
      --edge-diagnostics-min-degree "$min_degree"
      --representation-modulation-mode "$representation_modulation_mode"
      --representation-modulation-ramp-epochs "$representation_modulation_ramp_epochs"
      --lambda_ "$representation_modulation_lambda"
    )
    if [[ "$summary_only" == "1" ]]; then
      command+=(
        --export-edge-reliability-summary
        --edge-reliability-dir "$run_dir/edge_reliability"
        --edge-reliability-labels-file "$run_dir/synthetic_edge_labels.csv"
        --edge-reliability-noise-validation-file "$run_dir/noise_validation.json"
        --edge-reliability-momentum-quantile "$reliability_momentum_q"
        --edge-reliability-structure-quantile "$reliability_structure_q"
        --edge-reliability-structure-weight "$reliability_structure_weight"
        --edge-reliability-min-weight "$reliability_min_weight"
        --edge-reliability-filtering-epoch "$reliability_filter_epoch"
        --edge-reliability-filtering-schedule "$reliability_filter_schedule"
        --edge-reliability-adaptive-min-epoch "$reliability_adaptive_min_epoch"
        --edge-reliability-adaptive-max-epoch "$reliability_adaptive_max_epoch"
        --edge-reliability-adaptive-min-coverage "$reliability_adaptive_min_coverage"
        --edge-reliability-adaptive-jaccard "$reliability_adaptive_jaccard"
        --edge-reliability-adaptive-stable-checks "$reliability_adaptive_stable_checks"
        --edge-reliability-momentum-decay "$reliability_momentum_decay"
      )
    else
      command+=(
        --export-edge-diagnostics
        --edge-diagnostics-dir "$run_dir/edge_diagnostics"
        --edge-diagnostics-format "$diagnostics_format"
        --edge-diagnostics-verify-invariance
        --edge-diagnostics-labels-file "$run_dir/synthetic_edge_labels.csv"
        --edge-diagnostics-noise-validation-file "$run_dir/noise_validation.json"
      )
    fi
    if [[ "$stop_after_filter" == "1" ]]; then
      command+=(--edge-diagnostics-stop-after-filter)
    fi
    if [[ -n "$train_epochs" ]]; then
      command+=(--epochs "$train_epochs")
    fi
    if [[ -n "$train_patience" ]]; then
      command+=(--patience "$train_patience")
    fi
    if [[ -n "$train_lr" ]]; then
      command+=(--lr "$train_lr")
    fi
    if [[ -n "$train_init_weight" ]]; then
      command+=(--init_weight "$train_init_weight")
    fi

    {
      echo "base_commit=$commit_hash"
      echo "dataset=$dataset"
      echo "noise_mode=$noise_mode"
      echo "requested_noise_ratio=$ratio"
      echo "seed=$seed"
      echo "source_train=$variant_train"
      echo "gpu_id=$gpu_id"
      echo "stop_after_filter=$stop_after_filter"
      echo "replacement_selection=$replacement_selection"
      echo "edge_filter_mode=$edge_filter_mode"
      echo "reliability_filter_epoch=$reliability_filter_epoch"
      echo "reliability_filter_schedule=$reliability_filter_schedule"
      echo "reliability_adaptive_min_epoch=$reliability_adaptive_min_epoch"
      echo "reliability_adaptive_max_epoch=$reliability_adaptive_max_epoch"
      echo "reliability_adaptive_min_coverage=$reliability_adaptive_min_coverage"
      echo "reliability_adaptive_jaccard=$reliability_adaptive_jaccard"
      echo "reliability_adaptive_stable_checks=$reliability_adaptive_stable_checks"
      echo "reliability_momentum_decay=$reliability_momentum_decay"
      echo "reliability_momentum_quantile=$reliability_momentum_q"
      echo "reliability_structure_quantile=$reliability_structure_q"
      echo "reliability_structure_weight=$reliability_structure_weight"
      echo "representation_modulation_mode=$representation_modulation_mode"
      echo "representation_modulation_ramp_epochs=$representation_modulation_ramp_epochs"
      echo "representation_modulation_lambda=$representation_modulation_lambda"
      echo "summary_only=$summary_only"
      echo "keep_edge_labels=$keep_edge_labels"
      echo "keep_generated_train=$keep_generated_train"
      echo "hard_candidate_pool=$hard_candidate_pool"
      echo "hard_support_limit=$hard_support_limit"
      echo "train_epochs=${train_epochs:-entry_default}"
      echo "train_patience=${train_patience:-entry_default}"
      echo "train_lr=${train_lr:-entry_default}"
      echo "train_init_weight=${train_init_weight:-entry_default}"
      printf 'command='
      printf '%q ' "${command[@]}"
      printf '\n'
    } >"$run_dir/run_manifest.txt"

    (
      cd "$active_worktree/code"
      export PYTHONHASHSEED="$seed"
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export OMP_NUM_THREADS="$omp_threads"
      "${command[@]}"
    ) 2>&1 | tee "$run_dir/training.log"

    if [[ "$run_pilot_analysis" == "1" && "$summary_only" == "0" ]]; then
      python3 "$script_dir/analyze_edge_diagnostics.py" \
        --diagnostics-dir "$run_dir/edge_diagnostics" \
        --labels "$run_dir/synthetic_edge_labels.csv" \
        --noise-validation "$run_dir/noise_validation.json" \
        --output "$run_dir/pilot_analysis.json" \
        2>&1 | tee "$run_dir/pilot_analysis.log"
    fi

    if [[ "$summary_only" == "1" && "$keep_edge_labels" == "0" ]]; then
      rm -f -- "$run_dir/synthetic_edge_labels.csv"
    fi
    if [[ "$keep_generated_train" == "0" && \
          "$variant_train" == "$run_dir/generated_train.txt" ]]; then
      rm -f -- "$run_dir/generated_train.txt"
    fi

    git -C "$repo_root" worktree remove --force "$active_worktree"
    active_worktree=""
    rm -rf -- "$temporary_parent"
    temporary_parent=""
    echo "[$run_name] completed"
  done
done

echo "All grid runs completed."
