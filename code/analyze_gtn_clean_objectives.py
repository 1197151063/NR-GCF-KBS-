"""Validate and summarize the fixed-lambda GTN objective noise curve."""

import argparse
import json
import statistics
from pathlib import Path


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def ratio_key(value):
    return round(float(value), 10)


def summarize(report, profile):
    datasets = profile["datasets"]
    objectives = profile["objectives"]
    noise_ratios = [ratio_key(value) for value in profile["noise_ratios"]]
    expected_lambda = float(profile["common"]["gtn_lambda"])
    grouped = {}
    for run in report.get("runs", []):
        dataset = run.get("dataset")
        objective = run.get("training_objective")
        if dataset not in datasets or objective not in objectives:
            continue
        if run.get("backbone") != "gtn":
            raise ValueError(f"Non-GTN comparison run found: {run.get('run')}")
        if run.get("mode") != "none":
            raise ValueError(f"SDR filtering enabled in GTN run: {run.get('run')}")
        if run.get("representation_modulation_mode") != "none":
            raise ValueError(f"CrossNorm enabled in GTN run: {run.get('run')}")
        if int(run.get("train_num_layers")) != int(
            profile["common"]["gtn_iterations"]
        ):
            raise ValueError(f"Wrong GTN iteration count: {run.get('run')}")
        if run.get("gtn_lambda") is None or float(
            run["gtn_lambda"]
        ) != expected_lambda:
            raise ValueError(f"Wrong GTN lambda: {run.get('run')}")
        noise_ratio = ratio_key(run.get("requested_noise_ratio") or 0.0)
        if noise_ratio not in noise_ratios:
            continue
        if any(
            run.get(field) is None
            for field in ("best_epoch", "best_recall_at_20", "best_ndcg_at_20")
        ):
            raise ValueError(f"Missing best-epoch metrics: {run.get('run')}")
        grouped.setdefault((objective, dataset, noise_ratio), []).append(run)

    missing = [
        f"{objective}/{dataset}/noise={noise_ratio:g}"
        for objective in objectives
        for dataset in datasets
        for noise_ratio in noise_ratios
        if (objective, dataset, noise_ratio) not in grouped
    ]
    if missing:
        raise ValueError("Missing GTN comparison cases: " + ", ".join(missing))

    rows = []
    for objective in objectives:
        for dataset in datasets:
            for noise_ratio in noise_ratios:
                runs = grouped[(objective, dataset, noise_ratio)]
                recalls = [float(run["best_recall_at_20"]) for run in runs]
                ndcgs = [float(run["best_ndcg_at_20"]) for run in runs]
                rows.append({
                    "objective": objective,
                    "dataset": dataset,
                    "requested_noise_ratio": noise_ratio,
                    "gtn_lambda": expected_lambda,
                    "seed_count": len(runs),
                    "best_recall_at_20_mean": statistics.fmean(recalls),
                    "best_recall_at_20_std": statistics.pstdev(recalls),
                    "best_ndcg_at_20_mean": statistics.fmean(ndcgs),
                    "best_ndcg_at_20_std": statistics.pstdev(ndcgs),
                    "runs": [
                        {
                            "seed": run.get("seed"),
                            "best_epoch": run["best_epoch"],
                            "best_recall_at_20": run["best_recall_at_20"],
                            "best_ndcg_at_20": run["best_ndcg_at_20"],
                        }
                        for run in runs
                    ],
                })
    return {
        "schema_version": "gtn_objective_noise_curve_summary_v1",
        "comparison_method": "GTN",
        "sdr_gcf_backbone_unchanged": "LightGCN",
        "metric_policy": "best Recall@20 epoch with paired NDCG@20",
        "gtn_lambda": expected_lambda,
        "noise_ratios": noise_ratios,
        "case_count": len(rows),
        "rows": rows,
    }


def markdown(summary, datasets, objectives):
    lookup = {
        (
            row["objective"],
            row["dataset"],
            ratio_key(row["requested_noise_ratio"]),
        ): row
        for row in summary["rows"]
    }
    lines = [
        "# GTN objective noise-curve results",
        "",
        "GTN is independent; SDR-GCF remains LightGCN-based.",
        "GTN lambda is fixed to 1. Metrics come from the best Recall@20 epoch.",
    ]
    for objective in objectives:
        lines.extend([
            "",
            f"## {objective.upper()}",
            "",
            "| Noise | " + " | ".join(
                f"{dataset} R@20 | {dataset} N@20" for dataset in datasets
            ) + " |",
            "|---:|" + "---:|---:|" * len(datasets),
        ])
        for noise_ratio in summary["noise_ratios"]:
            cells = []
            for dataset in datasets:
                row = lookup[(objective, dataset, ratio_key(noise_ratio))]
                cells.extend([
                    f"{row['best_recall_at_20_mean']:.4f}",
                    f"{row['best_ndcg_at_20_mean']:.4f}",
                ])
            lines.append(
                f"| {noise_ratio:g} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    report = load_json(args.input)
    profile = load_json(args.profile)
    result = summarize(report, profile)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(
        markdown(result, profile["datasets"], profile["objectives"]),
        encoding="utf-8",
    )
    print(
        f"Summarized {result['case_count']} fixed-lambda GTN cases "
        f"across {len(result['noise_ratios'])} noise ratios"
    )


if __name__ == "__main__":
    main()
