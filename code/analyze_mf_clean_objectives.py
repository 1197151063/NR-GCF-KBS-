"""Validate and tabulate clean MF best-epoch results."""

import argparse
import json
import statistics
from pathlib import Path


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def summarize(report, profile):
    expected_datasets = profile["datasets"]
    expected_objectives = profile["objectives"]
    grouped = {}
    for run in report.get("runs", []):
        dataset = run.get("dataset")
        objective = run.get("training_objective")
        if dataset not in expected_datasets or objective not in expected_objectives:
            continue
        if run.get("backbone") != "mf":
            raise ValueError(f"Non-MF run found: {run.get('run')}")
        if run.get("mode") != "none":
            raise ValueError(f"Filtering enabled in MF run: {run.get('run')}")
        if run.get("representation_modulation_mode") != "none":
            raise ValueError(f"CrossNorm enabled in MF run: {run.get('run')}")
        if float(run.get("requested_noise_ratio") or 0.0) != 0.0:
            raise ValueError(f"Non-clean MF run found: {run.get('run')}")
        recall = run.get("best_recall_at_20")
        ndcg = run.get("best_ndcg_at_20")
        epoch = run.get("best_epoch")
        if recall is None or ndcg is None or epoch is None:
            raise ValueError(f"Missing best-epoch metrics: {run.get('run')}")
        grouped.setdefault((objective, dataset), []).append(run)

    missing = [
        f"{objective}/{dataset}"
        for objective in expected_objectives
        for dataset in expected_datasets
        if (objective, dataset) not in grouped
    ]
    if missing:
        raise ValueError("Missing MF cases: " + ", ".join(missing))

    rows = []
    for objective in expected_objectives:
        for dataset in expected_datasets:
            runs = grouped[(objective, dataset)]
            recalls = [float(run["best_recall_at_20"]) for run in runs]
            ndcgs = [float(run["best_ndcg_at_20"]) for run in runs]
            rows.append({
                "objective": objective,
                "dataset": dataset,
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
        "schema_version": "mf_clean_objectives_summary_v1",
        "metric_policy": "best Recall@20 epoch with paired NDCG@20",
        "case_count": len(rows),
        "rows": rows,
    }


def markdown(summary, datasets):
    lookup = {
        (row["objective"], row["dataset"]): row
        for row in summary["rows"]
    }
    lines = [
        "# MF clean objective results",
        "",
        "All values are from the best Recall@20 epoch; NDCG@20 is paired from that epoch.",
        "",
        "| Loss | " + " | ".join(
            f"{dataset} R@20 | {dataset} N@20" for dataset in datasets
        ) + " |",
        "|---|" + "---:|---:|" * len(datasets),
    ]
    for objective in ("bpr", "ssm", "au"):
        cells = []
        for dataset in datasets:
            row = lookup[(objective, dataset)]
            cells.extend([
                f"{row['best_recall_at_20_mean']:.4f}",
                f"{row['best_ndcg_at_20_mean']:.4f}",
            ])
        lines.append(f"| {objective.upper()} | " + " | ".join(cells) + " |")
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
        markdown(result, profile["datasets"]), encoding="utf-8"
    )
    print(f"Wrote {result['case_count']} MF cases to {args.output}")


if __name__ == "__main__":
    main()
