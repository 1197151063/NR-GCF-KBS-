"""Compare ordinary LightGCN propagation and always-on CrossNorm per loss."""

import argparse
import json
import math
from pathlib import Path


def analyze(
        report, dataset, objectives, baseline_mode, treatment_mode,
        noise_ratio, seed):
    objectives = list(dict.fromkeys(objectives))
    modes = (baseline_mode, treatment_mode)
    matched = {}
    for row in report.get("runs", []):
        if row.get("dataset") != dataset:
            continue
        if row.get("mode") != "none":
            continue
        if int(row.get("seed", -1)) != int(seed):
            continue
        if not math.isclose(
                float(row.get("requested_noise_ratio", -1)),
                float(noise_ratio), rel_tol=0.0, abs_tol=1e-12):
            continue
        objective = row.get("training_objective")
        mode = row.get("representation_modulation_mode")
        if objective not in objectives or mode not in modes:
            continue
        key = (objective, mode)
        if key in matched:
            raise ValueError("Duplicate objective/backbone row: %r" % (key,))
        matched[key] = row

    comparisons = []
    for objective in objectives:
        missing = [mode for mode in modes if (objective, mode) not in matched]
        if missing:
            raise ValueError(
                "Objective %s is missing modes %s" % (objective, missing)
            )
        baseline = matched[(objective, baseline_mode)]
        treatment = matched[(objective, treatment_mode)]
        recall_base = float(baseline["best_recall_at_20"])
        ndcg_base = float(baseline["best_ndcg_at_20"])
        recall_treatment = float(treatment["best_recall_at_20"])
        ndcg_treatment = float(treatment["best_ndcg_at_20"])
        comparisons.append({
            "objective": objective,
            "baseline": baseline,
            "treatment": treatment,
            "recall_absolute_delta": recall_treatment - recall_base,
            "ndcg_absolute_delta": ndcg_treatment - ndcg_base,
            "recall_relative_percent": (
                (recall_treatment / recall_base - 1.0) * 100.0
            ),
            "ndcg_relative_percent": (
                (ndcg_treatment / ndcg_base - 1.0) * 100.0
            ),
        })
    return {
        "schema_version": "nrgcf_objective_backbone_comparison_v1",
        "dataset": dataset,
        "requested_noise_ratio": float(noise_ratio),
        "seed": int(seed),
        "baseline_mode": baseline_mode,
        "treatment_mode": treatment_mode,
        "comparison_scope": (
            "No edge filtering; only the propagation operator differs within "
            "each training objective."
        ),
        "comparisons": comparisons,
    }


def markdown(report):
    lines = [
        "# %s objective/backbone comparison" % report["dataset"],
        "",
        "Noise `{noise:.3g}`, seed `{seed}`; baseline `{baseline}`, treatment "
        "`{treatment}`; edge filtering disabled.".format(
            noise=report["requested_noise_ratio"],
            seed=report["seed"],
            baseline=report["baseline_mode"],
            treatment=report["treatment_mode"],
        ),
        "",
        "| Objective | Baseline R@20 | CrossNorm R@20 | Relative gain | "
        "Baseline N@20 | CrossNorm N@20 | Relative gain | Best epochs |",
        "|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for comparison in report["comparisons"]:
        baseline = comparison["baseline"]
        treatment = comparison["treatment"]
        lines.append(
            "| {objective} | {br:.6f} | {tr:.6f} | {rg:+.2f}% | "
            "{bn:.6f} | {tn:.6f} | {ng:+.2f}% | {be}/{te} |".format(
                objective=comparison["objective"].upper(),
                br=float(baseline["best_recall_at_20"]),
                tr=float(treatment["best_recall_at_20"]),
                rg=float(comparison["recall_relative_percent"]),
                bn=float(baseline["best_ndcg_at_20"]),
                tn=float(treatment["best_ndcg_at_20"]),
                ng=float(comparison["ndcg_relative_percent"]),
                be=int(baseline["best_epoch"]),
                te=int(treatment["best_epoch"]),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="yelp2018")
    parser.add_argument("--objectives", default="ssm au")
    parser.add_argument("--baseline-mode", default="none")
    parser.add_argument("--treatment-mode", default="original_always")
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream),
            dataset=args.dataset,
            objectives=args.objectives.split(),
            baseline_mode=args.baseline_mode,
            treatment_mode=args.treatment_mode,
            noise_ratio=args.noise_ratio,
            seed=args.seed,
        )
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote objective/backbone comparison to %s" % args.output)


if __name__ == "__main__":
    main()
