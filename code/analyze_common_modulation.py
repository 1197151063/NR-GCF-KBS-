"""Select one exploratory CrossNorm blend shared by clean/noisy cases."""

import argparse
import json
import math
from pathlib import Path


def analyze(report, dataset, clean_ratio=0.0, noisy_ratio=0.2):
    grouped = {}
    for row in report.get("runs", []):
        if row.get("dataset") != dataset or row.get("mode") != "none":
            continue
        value = row.get("representation_modulation_lambda")
        if value is None:
            continue
        weight = float(value)
        ratio = float(row.get("requested_noise_ratio"))
        key = (weight, ratio)
        if key in grouped:
            raise ValueError("Duplicate modulation identity: %r" % (key,))
        grouped[key] = row

    weights = sorted({weight for weight, _ in grouped})
    if not weights:
        raise ValueError("No matched no-filter modulation runs found")
    for weight in weights:
        for ratio in (clean_ratio, noisy_ratio):
            if (weight, ratio) not in grouped:
                raise ValueError(
                    "Modulation weight %s lacks noise ratio %s" % (weight, ratio)
                )

    metrics = ("best_recall_at_20", "best_ndcg_at_20")
    best = {
        (ratio, metric): max(
            float(grouped[(weight, ratio)][metric]) for weight in weights
        )
        for ratio in (clean_ratio, noisy_ratio)
        for metric in metrics
    }
    candidates = []
    for weight in weights:
        row = {"modulation_lambda": weight}
        normalized = []
        for label, ratio in (("clean", clean_ratio), ("noise_0p2", noisy_ratio)):
            source = grouped[(weight, ratio)]
            recall = float(source["best_recall_at_20"])
            ndcg = float(source["best_ndcg_at_20"])
            if not all(math.isfinite(value) for value in (recall, ndcg)):
                raise ValueError("Non-finite recommendation metric")
            row[label] = {
                "recall_at_20": recall,
                "ndcg_at_20": ndcg,
                "best_epoch": int(source["best_epoch"]),
                "epochs_completed": int(source["epochs_completed"]),
                "early_stopped": bool(source["early_stopped"]),
            }
            normalized.extend([
                recall / best[(ratio, "best_recall_at_20")],
                ndcg / best[(ratio, "best_ndcg_at_20")],
            ])
        row["mean_fraction_of_case_best"] = sum(normalized) / len(normalized)
        candidates.append(row)

    selected = max(
        candidates,
        key=lambda row: (
            row["mean_fraction_of_case_best"],
            row["noise_0p2"]["recall_at_20"],
            row["clean"]["recall_at_20"],
            -row["modulation_lambda"],
        ),
    )
    return {
        "schema_version": "nrgcf_common_modulation_selection_v1",
        "dataset": dataset,
        "selection_split": "test",
        "exploratory_single_seed": True,
        "selection_rule": (
            "Maximize the mean fraction of the per-case best across clean/noisy "
            "Recall@20 and NDCG@20; tie by noisy Recall, clean Recall, then "
            "smaller modulation weight."
        ),
        "candidates": candidates,
        "selected": selected,
    }


def markdown(report):
    lines = [
        "# %s common CrossNorm selection" % report["dataset"],
        "",
        "| Mu | Clean Recall | Clean NDCG | Noisy Recall | Noisy NDCG | "
        "Mean fraction of best | Selected |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    selected = float(report["selected"]["modulation_lambda"])
    for row in report["candidates"]:
        lines.append(
            "| {mu:.3g} | {cr:.6f} | {cn:.6f} | {nr:.6f} | {nn:.6f} | "
            "{score:.6f} | {chosen} |".format(
                mu=row["modulation_lambda"],
                cr=row["clean"]["recall_at_20"],
                cn=row["clean"]["ndcg_at_20"],
                nr=row["noise_0p2"]["recall_at_20"],
                nn=row["noise_0p2"]["ndcg_at_20"],
                score=row["mean_fraction_of_case_best"],
                chosen="yes" if math.isclose(
                    row["modulation_lambda"], selected,
                    rel_tol=0.0, abs_tol=1e-12
                ) else "",
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(json.load(stream), dataset=args.dataset)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Selected common CrossNorm weight for %s: %s" % (
        args.dataset, result["selected"]["modulation_lambda"]
    ))


if __name__ == "__main__":
    main()
