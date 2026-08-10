"""Extract one matched NR-GCF noise-robustness curve."""

import argparse
import json
import math
from pathlib import Path


def analyze(
        report, dataset, removal_cap, modulation_lambda, structure_weight,
        noise_ratios, seed):
    targets = sorted(set(float(value) for value in noise_ratios))
    rows = []
    seen = set()
    for row in report.get("runs", []):
        if row.get("dataset") != dataset:
            continue
        if row.get("mode") != "hard_structure_momentum":
            continue
        if row.get("filtering_schedule") != "fixed":
            continue
        if int(row.get("seed", -1)) != int(seed):
            continue
        checks = (
            (row.get("max_removal_ratio"), removal_cap),
            (row.get("representation_modulation_lambda"), modulation_lambda),
            (row.get("structure_weight"), structure_weight),
        )
        if any(value is None or not math.isclose(
                float(value), float(target), rel_tol=0.0, abs_tol=1e-12
        ) for value, target in checks):
            continue
        noise = float(row["requested_noise_ratio"])
        if not any(math.isclose(
                noise, target, rel_tol=0.0, abs_tol=1e-12
        ) for target in targets):
            continue
        if noise in seen:
            raise ValueError("Duplicate noise-curve ratio: %s" % noise)
        seen.add(noise)
        rows.append(row)

    if len(rows) != len(targets):
        missing = [
            target for target in targets
            if not any(math.isclose(
                target, value, rel_tol=0.0, abs_tol=1e-12
            ) for value in seen)
        ]
        raise ValueError(
            "Expected %d noise-curve rows, found %d; missing=%s" % (
                len(targets), len(rows), missing
            )
        )
    rows.sort(key=lambda row: float(row["requested_noise_ratio"]))
    filtering_epochs = {int(row["filtering_epoch"]) for row in rows}
    if len(filtering_epochs) != 1:
        raise ValueError(
            "Noise curve must use one fixed filtering epoch; found %s" %
            sorted(filtering_epochs)
        )
    return {
        "schema_version": "nrgcf_noise_curve_v1",
        "dataset": dataset,
        "seed": int(seed),
        "filtering_schedule": "fixed",
        "filtering_epoch": next(iter(filtering_epochs)),
        "max_removal_ratio": float(removal_cap),
        "modulation_lambda": float(modulation_lambda),
        "structure_weight": float(structure_weight),
        "runs": rows,
    }


def _value(row, key, digits=4):
    value = row.get(key)
    if value is None:
        return "--"
    return ("%%.%df" % digits) % float(value)


def markdown(report):
    lines = [
        "# %s fixed-time noise curve" % report["dataset"],
        "",
        "Seed `{seed}`, filter epoch `{epoch}`, cap `{cap:.4g}`, "
        "modulation weight `{modulation:.4g}`, structure weight `{weight:.4g}`."
        .format(
            seed=report["seed"],
            epoch=report["filtering_epoch"],
            cap=report["max_removal_ratio"],
            modulation=report["modulation_lambda"],
            weight=report["structure_weight"],
        ),
        "",
        "| Requested noise | Actual noise | Best epoch | Recall@20 | NDCG@20 | "
        "Removed | Noisy removal | Precision | Momentum AUROC | Structure AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            "| {requested} | {actual} | {best} | {recall} | {ndcg} | "
            "{removed} | {noisy} | {precision} | {momentum} | {structure} |".format(
                requested=_value(row, "requested_noise_ratio", 2),
                actual=_value(row, "actual_noise_ratio", 4),
                best=int(row["best_epoch"]),
                recall=_value(row, "best_recall_at_20", 6),
                ndcg=_value(row, "best_ndcg_at_20", 6),
                removed=_value(row, "removed_ratio", 4),
                noisy=_value(row, "noisy_removal_rate", 4),
                precision=_value(row, "removed_precision_noisy", 4),
                momentum=_value(row, "momentum_auroc", 4),
                structure=_value(row, "structure_auroc", 4),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--removal-cap", type=float, required=True)
    parser.add_argument("--modulation-lambda", type=float, required=True)
    parser.add_argument("--structure-weight", type=float, required=True)
    parser.add_argument("--noise-ratios", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream),
            dataset=args.dataset,
            removal_cap=args.removal_cap,
            modulation_lambda=args.modulation_lambda,
            structure_weight=args.structure_weight,
            noise_ratios=args.noise_ratios.split(),
            seed=args.seed,
        )
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote %s noise curve to %s" % (args.dataset, args.output))


if __name__ == "__main__":
    main()
