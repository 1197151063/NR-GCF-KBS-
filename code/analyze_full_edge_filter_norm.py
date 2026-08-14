#!/usr/bin/env python3
"""Summarize the four-arm BPR edge-filter/CrossNorm experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--noise-ratios", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _arm_from_run(run: str, valid_arms: set[str]) -> str:
    matches = [part for part in Path(run).parts if part in valid_arms]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify one arm from run path: {run!r}")
    return matches[0]


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"count": 0, "mean": math.nan, "sample_std": math.nan}
    return {
        "count": len(values),
        "mean": _mean(values),
        "sample_std": _sample_std(values),
    }


def _relative_gain(value: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (value - baseline) / baseline


def analyze(
    report: dict[str, Any],
    profile: dict[str, Any],
    datasets: list[str],
    arms: list[str],
    noise_ratios: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    valid_arms = set(arms)
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, float, int, str]] = set()

    for row in report.get("runs", []):
        dataset = str(row.get("dataset"))
        if dataset not in datasets:
            continue
        ratio = float(row["requested_noise_ratio"])
        if not any(math.isclose(ratio, target, abs_tol=1e-12) for target in noise_ratios):
            continue
        arm = _arm_from_run(str(row.get("run", "")), valid_arms)
        seed = int(row["seed"])
        identity = (dataset, ratio, seed, arm)
        if identity in identities:
            raise ValueError(f"Duplicate experiment identity: {identity!r}")
        identities.add(identity)
        grouped[(dataset, ratio, arm)].append(row)

    expected = {
        (dataset, ratio, seed, arm)
        for dataset in datasets
        for ratio in noise_ratios
        for seed in seeds
        for arm in arms
    }
    missing = sorted(expected - identities)
    unexpected = sorted(identities - expected)
    if missing or unexpected:
        raise ValueError(
            f"Experiment grid mismatch: missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}"
        )

    rows_out: list[dict[str, Any]] = []
    lookup: dict[tuple[str, float, str], dict[str, Any]] = {}
    metric_keys = (
        "best_recall_at_20",
        "best_ndcg_at_20",
        "best_epoch",
        "removed_ratio",
        "noisy_removal_rate",
        "clean_removal_rate",
        "removed_precision_noisy",
    )
    for dataset in datasets:
        for ratio in noise_ratios:
            for arm in arms:
                case_rows = grouped[(dataset, ratio, arm)]
                compact = {
                    "dataset": dataset,
                    "noise_ratio": ratio,
                    "arm": arm,
                    "seeds": sorted(int(row["seed"]) for row in case_rows),
                    "metrics": {
                        key: _metric_summary(case_rows, key) for key in metric_keys
                    },
                }
                rows_out.append(compact)
                lookup[(dataset, ratio, arm)] = compact

    for row in rows_out:
        dataset = row["dataset"]
        ratio = row["noise_ratio"]
        recall = row["metrics"]["best_recall_at_20"]["mean"]
        ndcg = row["metrics"]["best_ndcg_at_20"]["mean"]
        gains: dict[str, Any] = {}
        for baseline_arm in ("lightgcn", "norm_only", "filter_only"):
            baseline = lookup.get((dataset, ratio, baseline_arm))
            if baseline is None:
                continue
            baseline_recall = baseline["metrics"]["best_recall_at_20"]["mean"]
            baseline_ndcg = baseline["metrics"]["best_ndcg_at_20"]["mean"]
            gains[baseline_arm] = {
                "recall_percent": _relative_gain(recall, baseline_recall),
                "ndcg_percent": _relative_gain(ndcg, baseline_ndcg),
            }
        row["gains_over"] = gains

    return {
        "schema_version": "nrgcf_full_edge_filter_norm_summary_v1",
        "source": report.get("root"),
        "protocol": {
            "objective": "bpr",
            "noise": "degree-preserving uniform edge replacement",
            "selection_split": "test",
            "early_stopping_monitor": "Recall@20",
            "profile_schema": profile.get("schema_version"),
        },
        "datasets": datasets,
        "arms": arms,
        "noise_ratios": noise_ratios,
        "seeds": seeds,
        "run_count": len(identities),
        "rows": rows_out,
    }


def _fmt_metric(summary: dict[str, float]) -> str:
    if summary["count"] == 0 or math.isnan(summary["mean"]):
        return "--"
    if summary["count"] == 1:
        return f"{summary['mean']:.6f}"
    return f"{summary['mean']:.6f} +/- {summary['sample_std']:.6f}"


def _markdown(result: dict[str, Any]) -> str:
    labels = {
        "lightgcn": "LightGCN",
        "norm_only": "+Norm",
        "filter_only": "+Filter",
        "full": "+Filter+Norm",
    }
    lines = [
        "# Full BPR edge-filter + CrossNorm experiment",
        "",
        "All structural features and filtering decisions use training edges only. "
        "Synthetic labels are evaluation-only. Hyperparameter selection and early "
        "stopping follow the project's test Recall@20 protocol.",
        "",
    ]
    for dataset in result["datasets"]:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Noise | Arm | Recall@20 | NDCG@20 | Removed | Noisy removal | Precision | Full gain vs arm |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["rows"]:
            if row["dataset"] != dataset:
                continue
            metrics = row["metrics"]
            gain = "--"
            if row["arm"] != "full":
                full = next(
                    candidate
                    for candidate in result["rows"]
                    if candidate["dataset"] == dataset
                    and math.isclose(candidate["noise_ratio"], row["noise_ratio"])
                    and candidate["arm"] == "full"
                )
                value = full["gains_over"].get(row["arm"], {}).get("recall_percent")
                if value is not None:
                    gain = f"{value:+.2f}%"
            lines.append(
                "| {noise:.2f} | {arm} | {recall} | {ndcg} | {removed} | "
                "{noisy} | {precision} | {gain} |".format(
                    noise=row["noise_ratio"],
                    arm=labels.get(row["arm"], row["arm"]),
                    recall=_fmt_metric(metrics["best_recall_at_20"]),
                    ndcg=_fmt_metric(metrics["best_ndcg_at_20"]),
                    removed=_fmt_metric(metrics["removed_ratio"]),
                    noisy=_fmt_metric(metrics["noisy_removal_rate"]),
                    precision=_fmt_metric(metrics["removed_precision_noisy"]),
                    gain=gain,
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    with open(args.input, encoding="utf-8") as stream:
        report = json.load(stream)
    with open(args.profile, encoding="utf-8") as stream:
        profile = json.load(stream)
    result = analyze(
        report,
        profile,
        args.datasets,
        args.arms,
        args.noise_ratios,
        args.seeds,
    )
    output = Path(args.output)
    markdown = Path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(result), encoding="utf-8")
    print(f"Wrote full experiment report to {output}")


if __name__ == "__main__":
    main()
