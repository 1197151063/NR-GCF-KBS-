"""Extract a matched structure--momentum ranking ablation table."""

import argparse
import json
import math
from pathlib import Path


def analyze(
        report, dataset, removal_cap, modulation_lambda, weights,
        noise_ratios=(0.0, 0.2)):
    weights = sorted(set(float(value) for value in weights))
    noise_ratios = sorted(set(float(value) for value in noise_ratios))
    rows = []
    seen = set()
    for row in report.get("runs", []):
        if row.get("dataset") != dataset:
            continue
        if row.get("mode") != "hard_structure_momentum":
            continue
        cap = row.get("max_removal_ratio")
        modulation = row.get("representation_modulation_lambda")
        weight = row.get("structure_weight")
        if cap is None or modulation is None or weight is None:
            continue
        if not math.isclose(float(cap), float(removal_cap), rel_tol=0.0, abs_tol=1e-12):
            continue
        if not math.isclose(
                float(modulation), float(modulation_lambda),
                rel_tol=0.0, abs_tol=1e-12):
            continue
        if not any(math.isclose(
                float(weight), target, rel_tol=0.0, abs_tol=1e-12
        ) for target in weights):
            continue
        noise = float(row["requested_noise_ratio"])
        if not any(math.isclose(
                noise, target, rel_tol=0.0, abs_tol=1e-12
        ) for target in noise_ratios):
            continue
        identity = (
            noise,
            float(weight),
            int(row["seed"]),
        )
        if identity in seen:
            raise ValueError("Duplicate ranking identity: %r" % (identity,))
        seen.add(identity)
        rows.append(row)

    expected = len(noise_ratios) * len(weights)
    if len(rows) != expected:
        raise ValueError(
            "Expected %d ranking rows, found %d" % (expected, len(rows))
        )
    rows.sort(key=lambda row: (
        float(row["requested_noise_ratio"]), float(row["structure_weight"])
    ))
    schedules = {row.get("filtering_schedule") for row in rows}
    filtering_epochs = {int(row["filtering_epoch"]) for row in rows}
    fixed_time_controlled = schedules == {"fixed"} and len(filtering_epochs) == 1
    if fixed_time_controlled:
        weight_semantics = (
            "risk=w_s*(1-structure_rank)+(1-w_s)*momentum_rank; removal "
            "count and filtering epoch are matched across every ranking arm"
        )
    else:
        weight_semantics = (
            "risk=w_s*(1-structure_rank)+(1-w_s)*momentum_rank; the same "
            "selected cap fixes removal count, while adaptive trigger time may "
            "still differ because it monitors the ranked top-B set"
        )
    return {
        "schema_version": "nrgcf_ranking_ablation_v2",
        "dataset": dataset,
        "selected_max_removal_ratio": float(removal_cap),
        "modulation_lambda": float(modulation_lambda),
        "noise_ratios": noise_ratios,
        "fixed_time_controlled": fixed_time_controlled,
        "filtering_epoch": (
            next(iter(filtering_epochs)) if fixed_time_controlled else None
        ),
        "weight_semantics": weight_semantics,
        "runs": rows,
    }


def markdown(report):
    lines = [
        "# %s ranking-signal ablation" % report["dataset"],
        "",
        "Selected removal cap: `%.4g`; modulation weight: `%.4g`." % (
            report["selected_max_removal_ratio"], report["modulation_lambda"]
        ),
        "",
        "| Noise | Structure weight | Filter epoch | Recall@20 | NDCG@20 | "
        "Removed | Noisy removal | Precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if report.get("fixed_time_controlled"):
        lines[3:3] = [
            "All ranking arms filter at fixed epoch `%d`." % report["filtering_epoch"],
            "",
        ]
    for row in report["runs"]:
        noisy_rate = row.get("noisy_removal_rate")
        precision = row.get("removed_precision_noisy")
        lines.append(
            "| {noise:.3g} | {weight:.2f} | {epoch} | {recall:.6f} | "
            "{ndcg:.6f} | {removed:.4f} | {noisy} | {precision} |".format(
                noise=float(row["requested_noise_ratio"]),
                weight=float(row["structure_weight"]),
                epoch=int(row["filtering_epoch"]),
                recall=float(row["best_recall_at_20"]),
                ndcg=float(row["best_ndcg_at_20"]),
                removed=float(row["removed_ratio"]),
                noisy="--" if noisy_rate is None else "%.4f" % float(noisy_rate),
                precision="--" if precision is None else "%.4f" % float(precision),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--removal-cap", type=float, required=True)
    parser.add_argument("--modulation-lambda", type=float, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--noise-ratios", default="0 0.2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream),
            dataset=args.dataset,
            removal_cap=args.removal_cap,
            modulation_lambda=args.modulation_lambda,
            weights=args.weights.split(),
            noise_ratios=args.noise_ratios.split(),
        )
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote %s ranking ablation to %s" % (args.dataset, args.output))


if __name__ == "__main__":
    main()
