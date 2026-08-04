"""Create a compact, reproducible pilot report from edge diagnostics.

The script is analysis-only.  It reads the exported table after training,
uses synthetic labels only as evaluation targets, and writes one small JSON
file suitable for copying back from a remote server.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = [
    "edge_id",
    "synthetic_is_noisy",
    "nr_gcf_removed",
    "normalized_edge_score",
    "historical_or_momentum_loss",
    "current_edge_loss",
    "user_degree_before",
    "item_degree_before",
    "min_endpoint_degree",
    "user_side_structure_mean",
    "item_side_structure_mean",
    "bilateral_structure_mean",
]

FLOAT_COLUMNS = {
    "normalized_edge_score",
    "historical_or_momentum_loss",
    "current_edge_loss",
    "user_side_structure_mean",
    "item_side_structure_mean",
    "bilateral_structure_mean",
}
INT_COLUMNS = {
    "edge_id",
    "user_degree_before",
    "item_degree_before",
    "min_endpoint_degree",
}
BOOL_COLUMNS = {"synthetic_is_noisy", "nr_gcf_removed"}


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_bool(value):
    if value is None or value == "":
        return -1
    normalized = str(value).strip().lower()
    if normalized in ("true", "1"):
        return 1
    if normalized in ("false", "0"):
        return 0
    raise ValueError("Invalid boolean value: %r" % value)


def _table_files(diagnostics_dir):
    root = Path(diagnostics_dir)
    preferred = [
        root / "edge_diagnostics.parquet",
        root / "edge_diagnostics.csv.gz",
        root / "edge_diagnostics.csv",
    ]
    for path in preferred:
        if path.exists():
            return [path]
    parts = sorted(root.glob("edge_diagnostics_part_*.parquet"))
    if not parts:
        parts = sorted(root.glob("edge_diagnostics_part_*.csv"))
    if not parts:
        raise FileNotFoundError("No edge diagnostics table found in %s" % root)
    return parts


def _append_csv(path, buffers):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Diagnostics table is missing columns: %s" % sorted(missing))
        chunk = {name: [] for name in REQUIRED_COLUMNS}
        for row in reader:
            for name in REQUIRED_COLUMNS:
                value = row[name]
                if name in FLOAT_COLUMNS:
                    chunk[name].append(float(value) if value != "" else float("nan"))
                elif name in INT_COLUMNS:
                    chunk[name].append(int(value))
                else:
                    chunk[name].append(_parse_bool(value))
            if len(chunk["edge_id"]) >= 100000:
                _flush_chunk(chunk, buffers)
        _flush_chunk(chunk, buffers)


def _flush_chunk(chunk, buffers):
    if not chunk["edge_id"]:
        return
    for name in REQUIRED_COLUMNS:
        if name in FLOAT_COLUMNS:
            dtype = np.float64
        elif name in INT_COLUMNS:
            dtype = np.int64
        else:
            dtype = np.int8
        buffers[name].append(np.asarray(chunk[name], dtype=dtype))
        chunk[name] = []


def _append_parquet(path, buffers):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet requires pyarrow: %s" % exc)
    parquet = pq.ParquetFile(path)
    missing = set(REQUIRED_COLUMNS) - set(parquet.schema.names)
    if missing:
        raise ValueError("Diagnostics table is missing columns: %s" % sorted(missing))
    for batch in parquet.iter_batches(batch_size=100000, columns=REQUIRED_COLUMNS):
        mapping = batch.to_pydict()
        for name in REQUIRED_COLUMNS:
            values = mapping[name]
            if name in FLOAT_COLUMNS:
                array = np.asarray(
                    [float("nan") if value is None else value for value in values],
                    dtype=np.float64,
                )
            elif name in INT_COLUMNS:
                array = np.asarray(values, dtype=np.int64)
            else:
                array = np.asarray(
                    [-1 if value is None else int(bool(value)) for value in values],
                    dtype=np.int8,
                )
            buffers[name].append(array)


def _load_table(diagnostics_dir):
    buffers = {name: [] for name in REQUIRED_COLUMNS}
    files = _table_files(diagnostics_dir)
    for path in files:
        if str(path).endswith(".parquet"):
            _append_parquet(path, buffers)
        else:
            _append_csv(path, buffers)
    arrays = {
        name: np.concatenate(parts) if parts else np.asarray([])
        for name, parts in buffers.items()
    }
    edge_ids = arrays["edge_id"]
    if not np.array_equal(edge_ids, np.arange(edge_ids.size, dtype=np.int64)):
        raise ValueError("edge_id is not one stable contiguous sequence")
    return arrays, [str(path) for path in files]


def _load_external_labels(path, expected_count):
    labels = np.empty(expected_count, dtype=np.int8)
    row_count = 0
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for expected, row in enumerate(reader):
            if expected >= expected_count:
                raise ValueError("External label file has extra rows")
            if int(row["edge_id"]) != expected:
                raise ValueError("External label edge_id mismatch at row %d" % expected)
            labels[expected] = _parse_bool(row["synthetic_is_noisy"])
            row_count += 1
        if row_count != expected_count:
            raise ValueError("External label row count does not match diagnostics")
    return labels


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    repeated_ranks = np.repeat(0.5 * (starts + ends - 1), ends - starts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = repeated_ranks
    return ranks


def _binary_metrics(labels, raw_score, higher_is_noisy=True):
    score = raw_score if higher_is_noisy else -raw_score
    valid = np.isfinite(score) & (labels >= 0)
    y = labels[valid].astype(np.int8)
    score = score[valid].astype(np.float64)
    positive = int(y.sum())
    negative = int(y.size - positive)
    if positive == 0 or negative == 0:
        return {"count": int(y.size), "positive_count": positive, "auroc": None, "average_precision": None}

    ranks = _average_ranks(score) + 1.0
    auc = (float(ranks[y == 1].sum()) - positive * (positive + 1) / 2.0) / (
        positive * negative
    )

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_score.size]
    cumulative_positive = np.cumsum(sorted_y, dtype=np.int64)
    positives_at_end = cumulative_positive[ends - 1]
    positives_before = np.r_[0, positives_at_end[:-1]]
    group_positive = positives_at_end - positives_before
    group_precision = positives_at_end / ends.astype(np.float64)
    precision_weight = float(np.sum(group_precision * group_positive))
    return {
        "count": int(y.size),
        "positive_count": positive,
        "auroc": auc,
        "average_precision": precision_weight / positive,
    }


def _stats(values, mask):
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None, "q20": None, "q80": None}
    return {
        "count": int(selected.size),
        "mean": float(selected.mean()),
        "std": float(selected.std()),
        "min": float(selected.min()),
        "median": float(np.median(selected)),
        "max": float(selected.max()),
        "q20": float(np.quantile(selected, 0.20)),
        "q80": float(np.quantile(selected, 0.80)),
    }


def _group_report(arrays, mask):
    fields = [
        "normalized_edge_score",
        "historical_or_momentum_loss",
        "current_edge_loss",
        "user_degree_before",
        "item_degree_before",
        "min_endpoint_degree",
        "user_side_structure_mean",
        "item_side_structure_mean",
        "bilateral_structure_mean",
    ]
    return {
        "edge_count": int(mask.sum()),
        "fields": {name: _stats(arrays[name].astype(np.float64), mask) for name in fields},
    }


def _corner(mask, labels):
    count = int(mask.sum())
    noisy = int(labels[mask].sum())
    total_noisy = int(labels.sum())
    return {
        "edge_count": count,
        "noisy_count": noisy,
        "noisy_rate": noisy / float(count) if count else None,
        "fraction_of_all_noisy": noisy / float(total_noisy) if total_noisy else None,
    }


def _pearson(left, right, mask):
    valid = mask & np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _correlations(momentum, structure, mask):
    valid = mask & np.isfinite(momentum) & np.isfinite(structure)
    if valid.sum() < 2:
        return {"count": int(valid.sum()), "pearson": None, "spearman": None}
    return {
        "count": int(valid.sum()),
        "pearson": _pearson(momentum, structure, valid),
        "spearman": _pearson(_average_ranks(momentum[valid]), _average_ranks(structure[valid]), np.ones(int(valid.sum()), dtype=bool)),
    }


def analyze(diagnostics_dir, output_path, labels_path=None, noise_validation=None):
    arrays, table_files = _load_table(diagnostics_dir)
    labels = arrays["synthetic_is_noisy"]
    if np.any(labels < 0) and labels_path:
        labels = _load_external_labels(labels_path, labels.size)
    if np.any(labels < 0):
        raise ValueError("Synthetic labels are missing; pass --labels for a legacy export")
    labels = labels.astype(np.int8)
    clean = labels == 0
    noisy = labels == 1
    all_mask = np.ones(labels.size, dtype=bool)
    removed = arrays["nr_gcf_removed"] == 1
    momentum = arrays["normalized_edge_score"]
    structure = arrays["bilateral_structure_mean"]
    valid_joint = np.isfinite(momentum) & np.isfinite(structure)
    loss_low, loss_high = np.quantile(momentum[np.isfinite(momentum)], [0.20, 0.80])
    structure_low, structure_high = np.quantile(structure[np.isfinite(structure)], [0.20, 0.80])

    metrics = {
        "normalized_momentum_loss": _binary_metrics(labels, momentum, True),
        "historical_or_momentum_loss": _binary_metrics(labels, arrays["historical_or_momentum_loss"], True),
        "current_edge_loss": _binary_metrics(labels, arrays["current_edge_loss"], True),
        "user_side_structure": _binary_metrics(labels, arrays["user_side_structure_mean"], False),
        "item_side_structure": _binary_metrics(labels, arrays["item_side_structure_mean"], False),
        "bilateral_structure": _binary_metrics(labels, structure, False),
        "degree_only_negative_min_endpoint_degree": _binary_metrics(labels, arrays["min_endpoint_degree"].astype(np.float64), False),
    }

    fusion = {}
    fusion_valid = valid_joint
    momentum_rank = _average_ranks(momentum[fusion_valid])
    structure_anomaly_rank = _average_ranks(-structure[fusion_valid])
    fusion_labels = labels[fusion_valid]
    for weight in np.linspace(0.0, 1.0, 21):
        fused = weight * momentum_rank + (1.0 - weight) * structure_anomaly_rank
        fusion["%.2f" % weight] = _binary_metrics(fusion_labels, fused, True)
    best_auc = max(fusion, key=lambda key: fusion[key]["auroc"])
    best_ap = max(fusion, key=lambda key: fusion[key]["average_precision"])

    degree = arrays["min_endpoint_degree"]
    buckets = {
        "degree_1": degree == 1,
        "degree_2_5": (degree >= 2) & (degree <= 5),
        "degree_6_20": (degree >= 6) & (degree <= 20),
        "degree_21_100": (degree >= 21) & (degree <= 100),
        "degree_gt_100": degree > 100,
    }
    degree_report = {}
    for name, mask in buckets.items():
        degree_report[name] = {
            "edge_count": int(mask.sum()),
            "noisy_count": int(labels[mask].sum()),
            "noisy_rate": float(labels[mask].mean()) if mask.any() else None,
            "momentum": _binary_metrics(labels[mask], momentum[mask], True),
            "bilateral_structure": _binary_metrics(labels[mask], structure[mask], False),
        }

    report = {
        "analysis_schema_version": "nrgcf_edge_pilot_analysis_v1",
        "diagnostics_dir": os.path.abspath(diagnostics_dir),
        "table_files": table_files,
        "edge_count": int(labels.size),
        "synthetic_clean_count": int(clean.sum()),
        "synthetic_noisy_count": int(noisy.sum()),
        "synthetic_noisy_fraction_of_exported_edges": float(noisy.mean()),
        "classification_metrics": metrics,
        "groups": {
            "all": _group_report(arrays, all_mask),
            "synthetic_clean": _group_report(arrays, clean),
            "synthetic_noisy": _group_report(arrays, noisy),
            "removed": _group_report(arrays, removed),
            "retained": _group_report(arrays, ~removed),
        },
        "filtering": {
            "removed_count": int(removed.sum()),
            "retained_count": int((~removed).sum()),
            "removed_noisy_count": int((removed & noisy).sum()),
            "removed_clean_count": int((removed & clean).sum()),
            "removal_precision": float(labels[removed].mean()) if removed.any() else None,
            "noisy_removal_rate": float((removed & noisy).sum()) / float(noisy.sum()) if noisy.any() else None,
            "clean_removal_rate": float((removed & clean).sum()) / float(clean.sum()) if clean.any() else None,
        },
        "quantile_thresholds": {
            "low_quantile": 0.20,
            "high_quantile": 0.80,
            "momentum_low": loss_low,
            "momentum_high": loss_high,
            "structure_low": structure_low,
            "structure_high": structure_high,
        },
        "momentum_structure_corners": {
            "high_momentum_low_structure": _corner(valid_joint & (momentum >= loss_high) & (structure <= structure_low), labels),
            "high_momentum_high_structure": _corner(valid_joint & (momentum >= loss_high) & (structure >= structure_high), labels),
            "low_momentum_low_structure": _corner(valid_joint & (momentum <= loss_low) & (structure <= structure_low), labels),
            "low_momentum_high_structure": _corner(valid_joint & (momentum <= loss_low) & (structure >= structure_high), labels),
            "high_momentum_all_structure": _corner(valid_joint & (momentum >= loss_high), labels),
            "low_structure_all_momentum": _corner(valid_joint & (structure <= structure_low), labels),
        },
        "conditional_classification_metrics": {
            "momentum_within_low_structure": _binary_metrics(
                labels[valid_joint & (structure <= structure_low)],
                momentum[valid_joint & (structure <= structure_low)],
                True,
            ),
            "structure_within_high_momentum": _binary_metrics(
                labels[valid_joint & (momentum >= loss_high)],
                structure[valid_joint & (momentum >= loss_high)],
                False,
            ),
            "structure_within_low_momentum": _binary_metrics(
                labels[valid_joint & (momentum <= loss_low)],
                structure[valid_joint & (momentum <= loss_low)],
                False,
            ),
        },
        "correlations": {
            "all": _correlations(momentum, structure, all_mask),
            "synthetic_clean": _correlations(momentum, structure, clean),
            "synthetic_noisy": _correlations(momentum, structure, noisy),
            "user_side_vs_item_side": _correlations(arrays["user_side_structure_mean"], arrays["item_side_structure_mean"], all_mask),
        },
        "degree_buckets": degree_report,
        "exploratory_rank_fusion": {
            "warning": "Weights are scanned on this same diagnostic run and must not be treated as a trained or final filtering formula.",
            "momentum_weight_grid": fusion,
            "best_auroc_weight": best_auc,
            "best_average_precision_weight": best_ap,
        },
        "noise_validation": None,
    }
    if noise_validation:
        with open(noise_validation, encoding="utf-8") as stream:
            report["noise_validation"] = json.load(stream)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(_json_safe(report), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description="Analyze one NR-GCF edge diagnostics export")
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", default=None, help="label CSV for legacy unlabeled exports")
    parser.add_argument("--noise-validation", default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    report = analyze(
        diagnostics_dir=args.diagnostics_dir,
        output_path=args.output,
        labels_path=args.labels,
        noise_validation=args.noise_validation,
    )
    print(json.dumps({
        "output": os.path.abspath(args.output),
        "edge_count": report["edge_count"],
        "classification_metrics": report["classification_metrics"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
