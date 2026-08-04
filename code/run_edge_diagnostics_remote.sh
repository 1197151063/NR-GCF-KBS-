#!/usr/bin/env bash
set -euo pipefail

# Remote-only launcher. It does not inject noise or alter NR-GCF training code.
# Usage: bash run_edge_diagnostics_remote.sh smoke|off|on|formal

mode="${1:-smoke}"
: "${DATASET:?Set DATASET}"
: "${SEED:?Set SEED}"
: "${GPU_ID:?Set GPU_ID}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
mkdir -p "$OUTPUT_DIR"

args=(
  --dataset "$DATASET"
  --seed "$SEED"
)

case "$mode" in
  off)
    args+=(--edge-diagnostics-stop-after-filter)
    ;;
  smoke)
    args+=(
      --export-edge-diagnostics
      --requested-noise-ratio 0
      --edge-diagnostics-dir "$OUTPUT_DIR/edge_diagnostics"
      --edge-diagnostics-format parquet
      --edge-diagnostics-structural-mode two_hop_minhash
      --edge-diagnostics-topk 10
      --edge-diagnostics-chunk-size 8192
      --edge-diagnostics-verify-invariance
      --edge-diagnostics-stop-after-filter
    )
    ;;
  on)
    args+=(
      --export-edge-diagnostics
      --edge-diagnostics-dir "$OUTPUT_DIR/edge_diagnostics"
      --edge-diagnostics-format parquet
      --edge-diagnostics-structural-mode two_hop_minhash
      --edge-diagnostics-topk 10
      --edge-diagnostics-chunk-size 8192
      --edge-diagnostics-verify-invariance
      --edge-diagnostics-stop-after-filter
    )
    ;;
  formal)
    : "${NOISE_RATIO:?Set NOISE_RATIO for formal metadata}"
    args+=(
      --requested-noise-ratio "$NOISE_RATIO"
      --export-edge-diagnostics
      --edge-diagnostics-dir "$OUTPUT_DIR/edge_diagnostics"
      --edge-diagnostics-format parquet
      --edge-diagnostics-structural-mode two_hop_minhash
      --edge-diagnostics-topk 10
      --edge-diagnostics-chunk-size 8192
      --edge-diagnostics-verify-invariance
    )
    ;;
  *)
    echo "Unknown mode: $mode (expected smoke, off, on, or formal)" >&2
    exit 2
    ;;
esac

export PYTHONHASHSEED="$SEED"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${NRGCF_OMP_NUM_THREADS:-4}"
python NR-GCF.py "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
