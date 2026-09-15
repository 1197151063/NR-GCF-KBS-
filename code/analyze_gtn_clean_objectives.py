"""Validate GTN comparison runs and select the clean-data lambda profile."""

import argparse
import json
import statistics
from pathlib import Path


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def summarize(report, profile):
    datasets = profile["datasets"]
    objectives = profile["objectives"]
    candidates = {
        dataset: [
            float(value)
            for value in profile["gtn_datasets"][dataset]["lambda_candidates"]
        ]
        for dataset in datasets
    }
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
        if float(run.get("requested_noise_ratio") or 0.0) != 0.0:
            raise ValueError(f"Non-clean GTN comparison run: {run.get('run')}")
        if int(run.get("train_num_layers")) != int(
            profile["common"]["gtn_iterations"]
        ):
            raise ValueError(f"Wrong GTN iteration count: {run.get('run')}")
        gtn_lambda = run.get("gtn_lambda")
        if gtn_lambda is None:
            raise ValueError(f"Missing GTN lambda: {run.get('run')}")
        gtn_lambda = float(gtn_lambda)
        if gtn_lambda not in candidates[dataset]:
            continue
        if any(
            run.get(field) is None
            for field in ("best_epoch", "best_recall_at_20", "best_ndcg_at_20")
        ):
            raise ValueError(f"Missing best-epoch metrics: {run.get('run')}")
        grouped.setdefault((objective, dataset, gtn_lambda), []).append(run)

    missing = [
        f"{objective}/{dataset}/lambda={gtn_lambda:g}"
        for objective in objectives
        for dataset in datasets
        for gtn_lambda in candidates[dataset]
        if (objective, dataset, gtn_lambda) not in grouped
    ]
    if missing:
        raise ValueError("Missing GTN comparison cases: " + ", ".join(missing))

    candidate_rows = []
    selected_rows = []
    for objective in objectives:
        for dataset in datasets:
            local_rows = []
            for gtn_lambda in candidates[dataset]:
                runs = grouped[(objective, dataset, gtn_lambda)]
                recalls = [float(run["best_recall_at_20"]) for run in runs]
                ndcgs = [float(run["best_ndcg_at_20"]) for run in runs]
                row = {
                    "objective": objective,
                    "dataset": dataset,
                    "gtn_lambda": gtn_lambda,
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
                }
                candidate_rows.append(row)
                local_rows.append(row)
            selected_rows.append(max(
                local_rows,
                key=lambda row: (
                    row["best_recall_at_20_mean"],
                    row["best_ndcg_at_20_mean"],
                    -row["gtn_lambda"],
                ),
            ))
    return {
        "schema_version": "gtn_clean_objectives_summary_v1",
        "comparison_method": "GTN",
        "sdr_gcf_backbone_unchanged": "LightGCN",
        "metric_policy": "best Recall@20 epoch with paired NDCG@20",
        "lambda_selection": (
            "highest mean best-epoch Recall@20; paired NDCG@20 then smaller "
            "lambda break ties"
        ),
        "candidate_case_count": len(candidate_rows),
        "selected_case_count": len(selected_rows),
        "selected_rows": selected_rows,
        "candidate_rows": candidate_rows,
    }


def markdown(summary, datasets, objectives):
    lookup = {
        (row["objective"], row["dataset"]): row
        for row in summary["selected_rows"]
    }
    lines = [
        "# GTN clean comparison results",
        "",
        "GTN is an independent comparison method. SDR-GCF continues to use LightGCN.",
        "All values are from the best Recall@20 epoch; NDCG@20 is paired from that epoch.",
        "",
        "| Loss | " + " | ".join(
            f"{dataset} R@20 | {dataset} N@20" for dataset in datasets
        ) + " |",
        "|---|" + "---:|---:|" * len(datasets),
    ]
    for objective in objectives:
        cells = []
        for dataset in datasets:
            row = lookup[(objective, dataset)]
            cells.extend([
                f"{row['best_recall_at_20_mean']:.4f}",
                f"{row['best_ndcg_at_20_mean']:.4f}",
            ])
        lines.append(f"| {objective.upper()} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Selected GTN lambda",
        "",
        "| Loss | " + " | ".join(datasets) + " |",
        "|---|" + "---:|" * len(datasets),
    ])
    for objective in objectives:
        values = [
            f"{lookup[(objective, dataset)]['gtn_lambda']:g}"
            for dataset in datasets
        ]
        lines.append(f"| {objective.upper()} | " + " | ".join(values) + " |")
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
        f"Selected {result['selected_case_count']} GTN comparison cases "
        f"from {result['candidate_case_count']} candidates"
    )


if __name__ == "__main__":
    main()
