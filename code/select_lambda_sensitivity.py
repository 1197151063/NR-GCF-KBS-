"""Select the best fusion or modulation weight for each noise ratio.

Here lambda is the structural weight in the structure--momentum risk:

    risk = lambda * (1 - structure_rank) + (1 - lambda) * momentum_rank

For modulation sensitivity, the selected field is the active ``--lambda_`` in
the explicit ``blend_always`` mode.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


RECALL_METRICS = {
    "best_recall_at_20": "best_ndcg_at_20",
    "best_post_filter_recall_at_20": "best_post_filter_ndcg_at_20",
}

PARAMETERS = {
    "fusion": {
        "field": "structure_weight",
        "expected_mode": "original_always",
        "semantics": (
            "structure weight in risk=lambda*(1-structure_rank)"
            "+(1-lambda)*momentum_rank"
        ),
    },
    "modulation": {
        "field": "representation_modulation_lambda",
        "expected_mode": "blend_always",
        "semantics": (
            "CrossNorm weight in x=lambda*cross_norm(propagated_x)"
            "+(1-lambda)*propagated_x"
        ),
    },
}


def _finite(value, field, run_name):
    if value is None:
        raise ValueError("Missing %s in run %s" % (field, run_name))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite %s in run %s" % (field, run_name))
    return number


def _mean(values):
    return float(statistics.fmean(values))


def _sample_std(values):
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _aggregate(values):
    return {
        "count": len(values),
        "mean": _mean(values),
        "sample_std": _sample_std(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _ratio_key(value):
    return float(value)


def _lambda_key(value):
    return float(value)


def analyze(report, recall_metric, parameter="fusion", strict_complete=True):
    if recall_metric not in RECALL_METRICS:
        raise ValueError("Unsupported selection metric: %s" % recall_metric)
    ndcg_metric = RECALL_METRICS[recall_metric]
    if parameter not in PARAMETERS:
        raise ValueError("Unsupported parameter: %s" % parameter)
    parameter_config = PARAMETERS[parameter]
    grouped = defaultdict(list)
    identities = set()

    for row in report.get("runs", []):
        if row.get("mode") != "hard_structure_momentum":
            continue
        if row.get("representation_modulation_mode") != parameter_config["expected_mode"]:
            continue
        run_name = row.get("run") or "<unknown>"
        dataset = str(row.get("dataset"))
        ratio = _ratio_key(row.get("requested_noise_ratio"))
        active_lambda = _lambda_key(row.get(parameter_config["field"]))
        seed = int(row.get("seed"))
        identity = (dataset, ratio, active_lambda, seed)
        if identity in identities:
            raise ValueError("Duplicate sensitivity identity: %r" % (identity,))
        identities.add(identity)
        grouped[(dataset, ratio, active_lambda)].append({
            "seed": seed,
            "recall": _finite(row.get(recall_metric), recall_metric, run_name),
            "ndcg": _finite(row.get(ndcg_metric), ndcg_metric, run_name),
            "best_epoch": _finite(row.get("best_epoch"), "best_epoch", run_name),
            "filtering_epoch": _finite(
                row.get("filtering_epoch"), "filtering_epoch", run_name
            ),
            "removed_ratio": _finite(
                row.get("removed_ratio"), "removed_ratio", run_name
            ),
            "noisy_removal_rate": (
                None if row.get("noisy_removal_rate") is None
                else _finite(
                    row.get("noisy_removal_rate"),
                    "noisy_removal_rate",
                    run_name,
                )
            ),
            "run": run_name,
        })

    if not grouped:
        raise ValueError("No hard_structure_momentum runs found")

    by_noise = defaultdict(list)
    for (dataset, ratio, active_lambda), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item["seed"])
        recalls = [item["recall"] for item in rows]
        ndcgs = [item["ndcg"] for item in rows]
        epochs = [item["best_epoch"] for item in rows]
        filtering_epochs = [item["filtering_epoch"] for item in rows]
        removed_ratios = [item["removed_ratio"] for item in rows]
        noisy_rates = [
            item["noisy_removal_rate"]
            for item in rows
            if item["noisy_removal_rate"] is not None
        ]
        summary = {
            "dataset": dataset,
            "requested_noise_ratio": ratio,
            "lambda": active_lambda,
            "lambda_semantics": parameter_config["semantics"],
            "seeds": [item["seed"] for item in rows],
            "recall_at_20": _aggregate(recalls),
            "ndcg_at_20": _aggregate(ndcgs),
            "best_epoch": _aggregate(epochs),
            "filtering_epoch": _aggregate(filtering_epochs),
            "removed_ratio": _aggregate(removed_ratios),
            "noisy_removal_rate": (
                _aggregate(noisy_rates) if noisy_rates else None
            ),
            "runs": [item["run"] for item in rows],
        }
        by_noise[(dataset, ratio)].append(summary)

    best_rows = []
    grid_rows = []
    for (dataset, ratio), candidates in sorted(by_noise.items()):
        candidates.sort(key=lambda item: item["lambda"])
        seed_sets = {tuple(item["seeds"]) for item in candidates}
        comparable = len(seed_sets) == 1
        if strict_complete and not comparable:
            details = {
                str(item["lambda"]): item["seeds"] for item in candidates
            }
            raise ValueError(
                "Lambda arms use different seed sets for dataset=%s, ratio=%s: %s"
                % (dataset, ratio, details)
            )
        best = max(
            candidates,
            key=lambda item: (
                item["recall_at_20"]["mean"],
                item["ndcg_at_20"]["mean"],
                -item["lambda"],
            ),
        )
        for candidate in candidates:
            candidate["selected_for_noise_ratio"] = (
                candidate["lambda"] == best["lambda"]
            )
            candidate["seed_sets_comparable"] = comparable
            grid_rows.append(candidate)
        best_rows.append({
            "dataset": dataset,
            "requested_noise_ratio": ratio,
            "selected_lambda": best["lambda"],
            "seeds": best["seeds"],
            "mean_recall_at_20": best["recall_at_20"]["mean"],
            "sample_std_recall_at_20": best["recall_at_20"]["sample_std"],
            "mean_ndcg_at_20": best["ndcg_at_20"]["mean"],
            "sample_std_ndcg_at_20": best["ndcg_at_20"]["sample_std"],
            "mean_removed_ratio": best["removed_ratio"]["mean"],
            "mean_filtering_epoch": best["filtering_epoch"]["mean"],
            "selection_rule": (
                "maximum mean test Recall@20; tie by mean test NDCG@20, "
                "then smaller lambda"
            ),
            "seed_sets_comparable": comparable,
        })

    return {
        "schema_version": "nrgcf_lambda_sensitivity_v1",
        "parameter": parameter,
        "lambda_semantics": parameter_config["semantics"],
        "selection_split": "test",
        "selection_metric": recall_metric,
        "selection_ndcg_tiebreak": ndcg_metric,
        "grid": grid_rows,
        "best_by_noise_ratio": best_rows,
    }


def markdown_table(analysis):
    lines = [
        "# Lambda sensitivity",
        "",
        "Parameter: `%s`." % analysis["parameter"],
        "",
        analysis["lambda_semantics"],
        "",
        "| Dataset | Noise ratio | Lambda | Seeds | Recall@20 | NDCG@20 | "
        "Removed | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in analysis["grid"]:
        recall = row["recall_at_20"]
        ndcg = row["ndcg_at_20"]
        selected = "yes" if row["selected_for_noise_ratio"] else ""
        lines.append(
            "| {dataset} | {ratio:.4g} | {lam:.4g} | {seeds} | "
            "{recall:.6f} $\\pm$ {recall_std:.6f} | "
            "{ndcg:.6f} $\\pm$ {ndcg_std:.6f} | {removed:.4f} | {selected} |".format(
                dataset=row["dataset"],
                ratio=row["requested_noise_ratio"],
                lam=row["lambda"],
                seeds=len(row["seeds"]),
                recall=recall["mean"],
                recall_std=recall["sample_std"],
                ndcg=ndcg["mean"],
                ndcg_std=ndcg["sample_std"],
                removed=row["removed_ratio"]["mean"],
                selected=selected,
            )
        )
    lines.extend(["", "## Selected lambda by noise ratio", ""])
    for row in analysis["best_by_noise_ratio"]:
        lines.append(
            "- `{dataset}`, noise `{ratio:.4g}`: lambda=`{lam:.4g}`, "
            "Recall@20=`{recall:.6f}`, NDCG@20=`{ndcg:.6f}`.".format(
                dataset=row["dataset"],
                ratio=row["requested_noise_ratio"],
                lam=row["selected_lambda"],
                recall=row["mean_recall_at_20"],
                ndcg=row["mean_ndcg_at_20"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="comparison_summary.json")
    parser.add_argument("--output", required=True, help="compact JSON output")
    parser.add_argument("--markdown", default=None, help="optional Markdown table")
    parser.add_argument(
        "--parameter",
        choices=sorted(PARAMETERS),
        default="fusion",
    )
    parser.add_argument(
        "--selection-metric",
        choices=sorted(RECALL_METRICS),
        default="best_recall_at_20",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow lambda arms with different seed sets",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    analysis = analyze(
        report,
        recall_metric=args.selection_metric,
        parameter=args.parameter,
        strict_complete=not args.allow_incomplete,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_table(analysis), encoding="utf-8")
    print(
        "Selected lambda for %d dataset/noise groups -> %s"
        % (len(analysis["best_by_noise_ratio"]), output_path)
    )


if __name__ == "__main__":
    main()
