"""Validate and summarize the 18 missing clean component-ablation cases."""

import argparse
import json
from pathlib import Path


OBJECTIVES = ("bpr", "ssm", "au")
DATASETS = ("lastfm", "ml-1m", "yelp2018", "amazon-book")
NORM_ONLY_DATASETS = ("yelp2018", "amazon-book")


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def arm_name(run):
    mode = run.get("mode")
    modulation = run.get("representation_modulation_mode")
    if mode == "none" and modulation == "blend_always":
        return "norm_only"
    if mode == "hard_structure_momentum" and modulation == "none":
        return "filter_only"
    raise ValueError(
        "Unexpected ablation semantics for run: " + str(run.get("run"))
    )


def expected_keys():
    keys = set()
    for objective in OBJECTIVES:
        keys.update(
            (objective, dataset, "norm_only")
            for dataset in NORM_ONLY_DATASETS
        )
        keys.update(
            (objective, dataset, "filter_only")
            for dataset in DATASETS
        )
    return keys


def summarize(report, expected_seed):
    indexed = {}
    for run in report.get("runs", []):
        objective = run.get("training_objective")
        dataset = run.get("dataset")
        if objective not in OBJECTIVES or dataset not in DATASETS:
            continue
        if run.get("backbone", "nrgcf") != "nrgcf":
            raise ValueError("Non-LightGCN backbone found: " + str(run.get("run")))
        if float(run.get("requested_noise_ratio") or 0.0) != 0.0:
            raise ValueError("Non-clean ablation found: " + str(run.get("run")))
        if int(run.get("seed")) != expected_seed:
            raise ValueError("Unexpected seed: " + str(run.get("run")))
        arm = arm_name(run)
        key = (objective, dataset, arm)
        if key not in expected_keys():
            raise ValueError("Unexpected ablation case: " + str(key))
        if key in indexed:
            raise ValueError("Duplicate ablation case: " + str(key))
        if any(
            run.get(field) is None
            for field in ("best_epoch", "best_recall_at_20", "best_ndcg_at_20")
        ):
            raise ValueError("Missing best-epoch metrics: " + str(run.get("run")))
        indexed[key] = run

    missing = sorted(expected_keys() - set(indexed))
    if missing:
        raise ValueError("Missing ablation cases: " + ", ".join(map(str, missing)))

    rows = []
    for objective in OBJECTIVES:
        for arm, datasets in (
            ("norm_only", NORM_ONLY_DATASETS),
            ("filter_only", DATASETS),
        ):
            for dataset in datasets:
                run = indexed[(objective, dataset, arm)]
                rows.append({
                    "objective": objective,
                    "dataset": dataset,
                    "arm": arm,
                    "table_variant": (
                        "w/o Structure Denoising"
                        if arm == "norm_only"
                        else "w/o Cross-Type Calibration"
                    ),
                    "best_epoch": run["best_epoch"],
                    "best_recall_at_20": run["best_recall_at_20"],
                    "best_ndcg_at_20": run["best_ndcg_at_20"],
                    "run": run.get("run"),
                })
    return {
        "schema_version": "missing_component_ablation_summary_v1",
        "metric_policy": "best Recall@20 epoch with paired NDCG@20",
        "case_count": len(rows),
        "rows": rows,
    }


def markdown(summary):
    lookup = {
        (row["objective"], row["arm"], row["dataset"]): row
        for row in summary["rows"]
    }
    lines = [
        "# Missing clean component-ablation results",
        "",
        "All metrics use the best Recall@20 epoch and paired NDCG@20.",
        "",
        "| Loss | Variant | LastFM | MovieLens-1M | Yelp2018 | Amazon-Book |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for objective in OBJECTIVES:
        for arm in ("norm_only", "filter_only"):
            cells = []
            for dataset in DATASETS:
                row = lookup.get((objective, arm, dataset))
                if row is None:
                    cells.append("existing result")
                else:
                    cells.append(
                        f"{row['best_recall_at_20']:.4f} / "
                        f"{row['best_ndcg_at_20']:.4f}"
                    )
            label = (
                "w/o Structure Denoising"
                if arm == "norm_only"
                else "w/o Cross-Type Calibration"
            )
            lines.append(
                f"| {objective.upper()} | {label} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = summarize(load_json(args.input), args.seed)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(
        markdown(result), encoding="utf-8"
    )
    print(f"Validated {result['case_count']} missing ablation cases")


if __name__ == "__main__":
    main()
