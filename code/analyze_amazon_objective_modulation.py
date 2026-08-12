"""Analyze Amazon-book SSM/AU LightGCN and CrossNorm-blend experiments."""

import argparse
import json
import math
import statistics
from pathlib import Path


def _finite(row, field):
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Missing or non-finite %s in %s" % (
            field, row.get("run", "<unknown>")))
    return float(value)


def _aggregate(values):
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "sample_std": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _validate_fixed(row, learning_rate, decay, message_dropout, batch_size,
                    ssm_tau, au_uniformity_weight, au_uniformity_t):
    for field, expected in (
            ("train_learning_rate", learning_rate),
            ("train_decay", decay)):
        if not math.isclose(
                _finite(row, field), float(expected),
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unexpected %s in %s" % (field, row.get("run")))
    if int(_finite(row, "train_batch_size")) != int(batch_size):
        raise ValueError("Unexpected train_batch_size in %s" % row.get("run"))
    if row.get("train_init_method") != "xavier_uniform":
        raise ValueError("Unexpected initialization in %s" % row.get("run"))
    metadata = row.get("training_objective_metadata") or {}
    if not math.isclose(
            float(metadata.get("message_dropout", 0.0)),
            float(message_dropout), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Unexpected message dropout in %s" % row.get("run"))
    objective = row.get("training_objective")
    if objective == "ssm":
        if not math.isclose(
                float(metadata.get("tau")), float(ssm_tau),
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unexpected SSM tau in %s" % row.get("run"))
    elif objective == "au":
        for field, expected in (
                ("uniformity_weight", au_uniformity_weight),
                ("uniformity_t", au_uniformity_t)):
            if not math.isclose(
                    float(metadata.get(field)), float(expected),
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "Unexpected AU %s in %s" % (field, row.get("run")))


def analyze(report, dataset, objectives, weights, seeds, learning_rate, decay,
            message_dropout, batch_size, ssm_tau, au_uniformity_weight,
            au_uniformity_t):
    objectives = list(objectives)
    weights = sorted(float(value) for value in weights)
    seeds = sorted(int(value) for value in seeds)
    matched = {}
    for row in report.get("runs", []):
        if row.get("dataset") != dataset or row.get("mode") != "none":
            continue
        if not math.isclose(
                _finite(row, "requested_noise_ratio"), 0.0,
                rel_tol=0.0, abs_tol=1e-12):
            continue
        objective = row.get("training_objective")
        mode = row.get("representation_modulation_mode")
        if objective not in objectives or mode not in ("none", "blend_always"):
            continue
        _validate_fixed(
            row, learning_rate=learning_rate, decay=decay,
            message_dropout=message_dropout, batch_size=batch_size,
            ssm_tau=ssm_tau, au_uniformity_weight=au_uniformity_weight,
            au_uniformity_t=au_uniformity_t)
        if mode == "none":
            arm = "lightgcn"
            weight = None
        else:
            arm = "modulation"
            weight = _finite(row, "representation_modulation_lambda")
            if weight not in weights:
                continue
        key = (objective, arm, weight, int(row.get("seed")))
        if key in matched:
            raise ValueError("Duplicate experiment identity: %r" % (key,))
        matched[key] = row

    objective_reports = []
    for objective in objectives:
        baseline_runs = []
        for seed in seeds:
            key = (objective, "lightgcn", None, seed)
            if key not in matched:
                raise ValueError("Missing LightGCN baseline: %r" % (key,))
            baseline_runs.append(matched[key])
        baseline = {
            "arm": "lightgcn",
            "modulation_weight": 0.0,
            "weight_note": "ordinary propagation; modulation mode is none",
            "seeds": seeds,
            "recall_at_20": _aggregate([
                _finite(row, "best_recall_at_20") for row in baseline_runs
            ]),
            "ndcg_at_20": _aggregate([
                _finite(row, "best_ndcg_at_20") for row in baseline_runs
            ]),
            "best_epoch": _aggregate([
                _finite(row, "best_epoch") for row in baseline_runs
            ]),
            "runs": [row.get("run") for row in baseline_runs],
        }
        modulation_rows = []
        for weight in weights:
            rows = []
            for seed in seeds:
                key = (objective, "modulation", weight, seed)
                if key not in matched:
                    raise ValueError("Missing modulation run: %r" % (key,))
                rows.append(matched[key])
            summary = {
                "arm": "modulation",
                "modulation_weight": weight,
                "weight_note": (
                    "H_next=(1-mu)*propagated_H+mu*CrossNorm(propagated_H)"
                ),
                "seeds": seeds,
                "recall_at_20": _aggregate([
                    _finite(row, "best_recall_at_20") for row in rows
                ]),
                "ndcg_at_20": _aggregate([
                    _finite(row, "best_ndcg_at_20") for row in rows
                ]),
                "best_epoch": _aggregate([
                    _finite(row, "best_epoch") for row in rows
                ]),
                "runs": [row.get("run") for row in rows],
            }
            summary["recall_gain_over_lightgcn_percent"] = (
                summary["recall_at_20"]["mean"]
                / baseline["recall_at_20"]["mean"] - 1.0
            ) * 100.0
            summary["ndcg_gain_over_lightgcn_percent"] = (
                summary["ndcg_at_20"]["mean"]
                / baseline["ndcg_at_20"]["mean"] - 1.0
            ) * 100.0
            modulation_rows.append(summary)
        best = max(modulation_rows, key=lambda row: (
            row["recall_at_20"]["mean"],
            row["ndcg_at_20"]["mean"],
            -row["modulation_weight"],
        ))
        objective_reports.append({
            "objective": objective,
            "lightgcn_baseline": baseline,
            "modulation_grid": modulation_rows,
            "best_observed_modulation": {
                "modulation_weight": best["modulation_weight"],
                "mean_recall_at_20": best["recall_at_20"]["mean"],
                "mean_ndcg_at_20": best["ndcg_at_20"]["mean"],
                "recall_gain_over_lightgcn_percent": (
                    best["recall_gain_over_lightgcn_percent"]
                ),
                "ndcg_gain_over_lightgcn_percent": (
                    best["ndcg_gain_over_lightgcn_percent"]
                ),
                "seeds": seeds,
            },
        })

    return {
        "schema_version": "nrgcf_amazon_objective_modulation_v1",
        "dataset": dataset,
        "noise_ratio": 0.0,
        "exploratory_sensitivity": True,
        "selection_split": "test",
        "selection_rule": (
            "within each objective, maximum mean Recall@20; tie by mean "
            "NDCG@20, then smaller modulation weight"
        ),
        "fixed_configuration": {
            "learning_rate": float(learning_rate),
            "decay": float(decay),
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "embedding_initialization": "xavier_uniform_gain_1",
            "ssm_temperature": float(ssm_tau),
            "au_uniformity_weight": float(au_uniformity_weight),
            "au_uniformity_t": float(au_uniformity_t),
            "edge_filter_mode": "none",
        },
        "objectives": objective_reports,
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# Amazon-book SSM/AU modulation sensitivity",
        "",
        "Clean data and no edge filtering. LightGCN baselines are trained "
        "first; all modulation arms use `blend_always`.",
        "",
        "Fixed: `lr={lr:g}`, `decay={decay:g}`, `dropout={dropout:g}`, "
        "`batch={batch}`, SSM `tau={tau:g}`, AU uniformity "
        "`weight={weight:g}, t={ut:g}`.".format(
            lr=fixed["learning_rate"], decay=fixed["decay"],
            dropout=fixed["message_dropout"], batch=fixed["batch_size"],
            tau=fixed["ssm_temperature"],
            weight=fixed["au_uniformity_weight"],
            ut=fixed["au_uniformity_t"]),
    ]
    for objective in report["objectives"]:
        baseline = objective["lightgcn_baseline"]
        selected = objective["best_observed_modulation"]["modulation_weight"]
        lines.extend([
            "",
            "## %s" % objective["objective"].upper(),
            "",
            "| Arm | Mu | Seeds | Recall@20 | NDCG@20 | Best epoch | "
            "Recall gain | Selected |",
            "|:---|---:|---:|---:|---:|---:|---:|:---:|",
            "| LightGCN | -- | {seeds} | {recall:.6f} | {ndcg:.6f} | "
            "{epoch:.2f} | -- |  |".format(
                seeds=len(baseline["seeds"]),
                recall=baseline["recall_at_20"]["mean"],
                ndcg=baseline["ndcg_at_20"]["mean"],
                epoch=baseline["best_epoch"]["mean"]),
        ])
        for row in objective["modulation_grid"]:
            lines.append(
                "| Blend | {mu:.3g} | {seeds} | {recall:.6f} | "
                "{ndcg:.6f} | {epoch:.2f} | {gain:+.2f}% | {chosen} |".format(
                    mu=row["modulation_weight"], seeds=len(row["seeds"]),
                    recall=row["recall_at_20"]["mean"],
                    ndcg=row["ndcg_at_20"]["mean"],
                    epoch=row["best_epoch"]["mean"],
                    gain=row["recall_gain_over_lightgcn_percent"],
                    chosen=("yes" if math.isclose(
                        row["modulation_weight"], selected,
                        rel_tol=0.0, abs_tol=1e-12) else "")))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--objectives", nargs="+", default=["ssm", "au"])
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--message-dropout", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--ssm-tau", type=float, required=True)
    parser.add_argument("--au-uniformity-weight", type=float, required=True)
    parser.add_argument("--au-uniformity-t", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream), dataset=args.dataset,
            objectives=args.objectives, weights=args.weights, seeds=args.seeds,
            learning_rate=args.learning_rate, decay=args.decay,
            message_dropout=args.message_dropout, batch_size=args.batch_size,
            ssm_tau=args.ssm_tau,
            au_uniformity_weight=args.au_uniformity_weight,
            au_uniformity_t=args.au_uniformity_t)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    for objective in result["objectives"]:
        print("Best %s modulation weight: %s" % (
            objective["objective"].upper(),
            objective["best_observed_modulation"]["modulation_weight"]))


if __name__ == "__main__":
    main()
