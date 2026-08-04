# Remote NR-GCF edge diagnostics runbook

Run these commands from the repository's `code/` directory so the existing
`../data/` and `../log/` relative paths keep their current meaning.  The
commands do not inject synthetic noise.  `--requested-noise-ratio` records
metadata only; prepare the intended training split externally using the
existing server workflow.

The same commands are packaged in `code/run_edge_diagnostics_remote.sh` with
`smoke`, `off`, `on`, and `formal` modes. The explicit commands below document
exactly what the launcher does.

The NR-GCF entry now applies `--seed` to Python, NumPy, torch CPU, and all
visible CUDA generators before constructing the dataset or model. Exact
bitwise reproducibility can still depend on sparse CUDA kernels and the
server's PyTorch/PyG configuration. Set `PYTHONHASHSEED` in the shell as well
for a fully documented launch environment.

The packaged launcher sets `OMP_NUM_THREADS` to 4 by default to avoid malformed
inherited values. Override it with `NRGCF_OMP_NUM_THREADS` when appropriate.

## Server environment checks

```bash
python --version
python - <<'PY'
import importlib.util
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_runtime:", torch.version.cuda)
print("pyarrow_installed:", importlib.util.find_spec("pyarrow") is not None)
print("torch_sparse_installed:", importlib.util.find_spec("torch_sparse") is not None)
print("torch_geometric_installed:", importlib.util.find_spec("torch_geometric") is not None)
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(index, properties.name, properties.total_memory)

indices = torch.tensor([[0, 1], [0, 1]])
values = torch.ones(2)
sparse = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
dense = torch.ones((2, 2))
print("cpu_sparse_mm:", torch.sparse.mm(sparse, dense))
scatter_target = torch.full((2, 1), 2147483647, dtype=torch.int32)
scatter_index = torch.tensor([[0], [1]], dtype=torch.int64)
scatter_source = torch.tensor([[7], [3]], dtype=torch.int32)
scatter_target.scatter_reduce_(0, scatter_index, scatter_source, reduce="amin", include_self=True)
print("cpu_int32_scatter_amin:", scatter_target)
if torch.cuda.is_available():
    print("cuda_sparse_mm:", torch.sparse.mm(sparse.cuda(), dense.cuda()).cpu())
    cuda_target = torch.full((2, 1), 2147483647, dtype=torch.int32, device="cuda")
    cuda_target.scatter_reduce_(0, scatter_index.cuda(), scatter_source.cuda(), reduce="amin", include_self=True)
    print("cuda_int32_scatter_amin:", cuda_target.cpu())
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

Do not install or upgrade `pyarrow` merely for diagnostics. If it is absent,
the exporter logs the reason and writes one streaming
`edge_diagnostics.csv.gz` file. With `pyarrow`, all chunks are appended to one
Zstandard-compressed `edge_diagnostics.parquet` file. The full edge table is
never retained in memory.

Synthetic labels can be supplied through
`--edge-diagnostics-labels-file`. They are checked against edge ID and both
endpoints while streaming, written into the diagnostic table, and used only
by post-export summaries. They never enter feature computation, filtering, or
training.

## Structural score used by diagnostics v2

`two_hop_minhash` builds deterministic MinHash signatures only from the
pre-filter training graph. For each target edge, per-hash first/second minima
remove the target endpoint from the candidate neighborhood. The target is
compared with at most 16 deterministically selected same-type neighbors.

For candidate/neighbor degrees `a,b`, the signature match rate estimates
Jaccard similarity `J`. Diagnostics converts it to normalized two-hop overlap:

```text
estimated_intersection = J * (a + b) / (1 + J)
structural_similarity  = estimated_intersection / sqrt(a * b)
```

The estimated intersection is capped by `min(a,b)` before normalization so
the score remains consistent with possible set overlap under estimation noise.

The exported mean/max/top-k fields summarize these bounded comparisons. This
avoids the high-degree saturation observed with the previous 64-dimensional
CountSketch. It remains an approximation: finite signatures and bounded
neighbor sampling add variance, which is why both valid and sampled neighbor
counts are exported.

## GPU smoke test

The filtering/export point is hard-coded at epoch 15 in the current entry, so
an edge-table smoke test cannot finish in one or two epochs.  The explicit
`--edge-diagnostics-stop-after-filter` option stops immediately after the
first valid export point and before epoch-15 evaluation or any later update.

```bash
export GPU_ID=0
export DATASET=yelp2018
export SEED=0
export OUTPUT_DIR=/path/to/a/new/smoke-output
mkdir -p "$OUTPUT_DIR"

PYTHONHASHSEED="$SEED" CUDA_VISIBLE_DEVICES="$GPU_ID" python NR-GCF.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --export-edge-diagnostics \
  --requested-noise-ratio 0 \
  --edge-diagnostics-dir "$OUTPUT_DIR/edge_diagnostics" \
  --edge-diagnostics-format parquet \
  --edge-diagnostics-structural-mode two_hop_minhash \
  --edge-diagnostics-topk 10 \
  --edge-diagnostics-chunk-size 8192 \
  --edge-diagnostics-verify-invariance \
  --edge-diagnostics-stop-after-filter \
  2>&1 | tee "$OUTPUT_DIR/training.log"
```

## Diagnostics switch invariance comparison

Use the same prepared dataset and seed. Both commands stop at
the same filtering point.  Keep the output directories separate.

Diagnostics off:

```bash
export GPU_ID=0 DATASET=yelp2018 SEED=0
export OUTPUT_DIR=/path/to/a/new/invariance-off
mkdir -p "$OUTPUT_DIR"
PYTHONHASHSEED="$SEED" CUDA_VISIBLE_DEVICES="$GPU_ID" python NR-GCF.py \
  --dataset "$DATASET" --seed "$SEED" \
  --edge-diagnostics-stop-after-filter \
  2>&1 | tee "$OUTPUT_DIR/training.log"
```

Diagnostics on:

```bash
export OUTPUT_DIR=/path/to/a/new/invariance-on
mkdir -p "$OUTPUT_DIR"
PYTHONHASHSEED="$SEED" CUDA_VISIBLE_DEVICES="$GPU_ID" python NR-GCF.py \
  --dataset "$DATASET" --seed "$SEED" \
  --export-edge-diagnostics \
  --edge-diagnostics-dir "$OUTPUT_DIR/edge_diagnostics" \
  --edge-diagnostics-format parquet \
  --edge-diagnostics-structural-mode two_hop_minhash \
  --edge-diagnostics-topk 10 \
  --edge-diagnostics-chunk-size 8192 \
  --edge-diagnostics-verify-invariance \
  --edge-diagnostics-stop-after-filter \
  2>&1 | tee "$OUTPUT_DIR/training.log"
```

Compare epoch losses and the filtering counts in the two logs.  The enabled
run also writes `invariance.json`, which checks tensors, parameters, and RNG
states immediately before and after exporter execution.

## Formal diagnostics export template

Set the five requested placeholders and point the existing data path at the
already-prepared split before running.  Omit
`--edge-diagnostics-stop-after-filter` if the normal post-filter training run
must continue.

```bash
export DATASET=DATASET
export NOISE_RATIO=NOISE_RATIO
export SEED=SEED
export GPU_ID=GPU_ID
export OUTPUT_DIR=OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

PYTHONHASHSEED="$SEED" CUDA_VISIBLE_DEVICES="$GPU_ID" python NR-GCF.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --requested-noise-ratio "$NOISE_RATIO" \
  --export-edge-diagnostics \
  --edge-diagnostics-dir "$OUTPUT_DIR/edge_diagnostics" \
  --edge-diagnostics-format parquet \
  --edge-diagnostics-structural-mode two_hop_minhash \
  --edge-diagnostics-topk 10 \
  --edge-diagnostics-chunk-size 8192 \
  --edge-diagnostics-verify-invariance \
  2>&1 | tee "$OUTPUT_DIR/training.log"
```

## Focused hard-replacement pilot

The repository includes `code/run_hard_replace_pilot.sh`. It runs one 10%
degree-preserving replacement pilot with seed 2026 by default. Unlike uniform
replacement, each swap is selected from a small random candidate pool using a
bounded leave-one-out bilateral two-hop score. This creates false edges that
are more structurally plausible while still preserving every user and item
degree exactly. It does not read validation/test data and does not construct a
dense node-by-node matrix.

From `code/` on the GPU server:

```bash
GPU_ID=0 \
OUTPUT_ROOT=/root/autodl-tmp/outputs \
bash run_hard_replace_pilot.sh
```

Portable form:

```bash
DATASET=DATASET \
NOISE_RATIOS=0.10 \
SEEDS=2026 \
GPU_ID=GPU_ID \
OUTPUT_ROOT=OUTPUT_DIR \
HARD_CANDIDATE_POOL=8 \
HARD_SUPPORT_LIMIT=16 \
bash run_hard_replace_pilot.sh
```

The launcher automatically creates `pilot_analysis.json` after diagnostics.
It contains clean/noisy metrics, degree controls, filtering precision/recall,
momentum–structure correlations, the four diagnostic quadrants, and an
explicitly exploratory rank-fusion scan. No classifier is trained.

## Files to return after a remote run

Copy back, without changing their relative names:

- `edge_diagnostics/metadata.json`
- `edge_diagnostics/schema.json`
- `edge_diagnostics/summary.json`
- `edge_diagnostics/invariance.json` when verification was enabled
- `edge_diagnostics/edge_diagnostics.parquet`, or the automatic fallback
  `edge_diagnostics/edge_diagnostics.csv.gz`
- `edge_diagnostics/logs/diagnostics.log`
- the captured `training.log`
- `pilot_analysis.json` and `pilot_analysis.log`
- the exact command/configuration used for the run
- the Git commit hash and dirty-worktree state recorded in `metadata.json`
- the server-side run identifier and checkpoint hash, if the external training
  workflow creates a checkpoint

The current NR-GCF entry does not save checkpoints. The grid launcher now
connects its validated synthetic-label and noise-validation sidecars to the
diagnostics exporter; direct runs that omit those arguments keep the label and
actual-noise fields null.
