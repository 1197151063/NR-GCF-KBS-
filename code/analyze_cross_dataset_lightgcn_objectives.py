"""Validate and summarize clean LightGCN SSM/AU runs across datasets."""

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


def _close(actual, expected):
    return math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    )


def _validate_configuration(
        row, learning_rate, decay, message_dropout, batch_size,
        ssm_tau, au_uniformity_weight, au_uniformity_t):
    if row.get("mode") != "none":
        raise ValueError("Edge filtering is not disabled in %s" % row.get("run"))
    if row.get("representation_modulation_mode") != "none":
        raise ValueError("Representation modulation is active in %s" % (
            row.get("run")))
    if not _close(_finite(row, "requested_noise_ratio"), 0.0):
        raise ValueError("A non-clean run was found in %s" % row.get("run"))
    for field, expected in (
            ("train_learning_rate", learning_rate),
            ("train_decay", decay)):
        if not _close(_finite(row, field), expected):
            raise ValueError("Unexpected %s in %s" % (field, row.get("run")))
    if int(_finite(row, "train_batch_size")) != int(batch_size):
        raise ValueError("Unexpected train_batch_size in %s" % row.get("run"))
    if row.get("train_init_method") != "xavier_uniform":
        raise ValueError("Unexpected initialization in %s" % row.get("run"))

    metadata = row.get("training_objective_metadata") or {}
    if not _close(metadata.get("message_dropout", 0.0), message_dropout):
        raise ValueError("Unexpected message dropout in %s" % row.get("run"))
    objective = row.get("training_objective")
    if metadata.get("name") != objective:
        raise ValueError("Objective metadata mismatch in %s" % row.get("run"))
    if objective == "ssm":
        if not _close(metadata.get("tau"), ssm_tau):
            raise ValueError("Unexpected SSM tau in %s" % row.get("run"))
    elif objective == "au":
        if not _close(
                metadata.get("uniformity_weight"), au_uniformity_weight):
            raise ValueError("Unexpected AU uniformity weight in %s" % (
                row.get("run")))
        if not _close(metadata.get("uniformity_t"), au_uniformity_t):
            raise ValueError("Unexpected AU uniformity t in %s" % row.get("run"))
    else:
        raise ValueError("Unexpected objective %r" % objective)


def analyze(
        report, datasets, objectives, seeds, learning_rate, decay,
        message_dropout, batch_size, ssm_tau, au_uniformity_weight,
        au_uniformity_t):
    datasets = list(datasets)
    objectives = list(objectives)
    seeds = sorted(int(seed) for seed in seeds)
    expected = {
        (dataset, objective, seed)
        for dataset in datasets
        for objective in objectives
        for seed in seeds
    }
    matched = {}
    for row in report.get("runs", []):
        identity = (
            row.get("dataset"), row.get("training_objective"),
            int(row.get("seed")) if row.get("seed") is not None else None,
        )
        if identity not in expected:
            continue
        _validate_configuration(
            row, learning_rate=learning_rate, decay=decay,
            message_dropout=message_dropout, batch_size=batch_size,
            ssm_tau=ssm_tau,
            au_uniformity_weight=au_uniformity_weight,
            au_uniformity_t=au_uniformity_t)
        if identity in matched:
            raise ValueError("Duplicate experiment identity: %r" % (identity,))
        matched[identity] = row
    missing = sorted(expected.difference(matched))
    if missing:
        raise ValueError("Missing experiment identities: %r" % (missing,))

    dataset_reports = []
    for dataset in datasets:
        objective_reports = []
        for objective in objectives:
            rows = [matched[(dataset, objective, seed)] for seed in seeds]
            objective_reports.append({
                "objective": objective,
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
                "epochs_completed": _aggregate([
                    _finite(row, "epochs_completed") for row in rows
                ]),
                "early_stopped_count": sum(
                    bool(row.get("early_stopped")) for row in rows
                ),
                "runs": [row.get("run") for row in rows],
            })
        dataset_reports.append({
            "dataset": dataset,
            "objectives": objective_reports,
        })

    return {
        "schema_version": "nrgcf_cross_dataset_lightgcn_objectives_v1",
        "protocol": "clean_lightgcn_objective_baselines",
        "noise_ratio": 0.0,
        "edge_filter_mode": "none",
        "representation_modulation_mode": "none",
        "selection_split": "test",
        "fixed_configuration": {
            "learning_rate": float(learning_rate),
            "decay": float(decay),
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "embedding_initialization": "xavier_uniform_gain_1",
            "ssm_temperature": float(ssm_tau),
            "au_uniformity_weight": float(au_uniformity_weight),
            "au_uniformity_t": float(au_uniformity_t),
        },
        "datasets": dataset_reports,
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# LastFM and ML-1M LightGCN objective baselines",
        "",
        "Clean graph, no edge filtering, and no representation modulation.",
        "",
        "Fixed: `lr={lr:g}`, `decay={decay:g}`, `dropout={dropout:g}`, "
        "`batch={batch}`, SSM `tau={tau:g}`, AU uniformity "
        "`weight={weight:g}, t={ut:g}`.".format(
            lr=fixed["learning_rate"], decay=fixed["decay"],
            dropout=fixed["message_dropout"], batch=fixed["batch_size"],
            tau=fixed["ssm_temperature"],
            weight=fixed["au_uniformity_weight"],
            ut=fixed["au_uniformity_t"]),
        "",
        "| Dataset | Objective | Seeds | Recall@20 | NDCG@20 | "
        "Best epoch | Completed epochs | Early stopped |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in report["datasets"]:
        for objective in dataset["objectives"]:
            lines.append(
                "| {dataset} | {objective} | {seeds} | {recall:.6f} | "
                "{ndcg:.6f} | {epoch:.2f} | {completed:.2f} | {stopped} |".format(
                    dataset=dataset["dataset"],
                    objective=objective["objective"].upper(),
                    seeds=len(objective["seeds"]),
                    recall=objective["recall_at_20"]["mean"],
                    ndcg=objective["ndcg_at_20"]["mean"],
                    epoch=objective["best_epoch"]["mean"],
                    completed=objective["epochs_completed"]["mean"],
                    stopped=objective["early_stopped_count"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--objectives", nargs="+", required=True)
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
            json.load(stream), datasets=args.datasets,
            objectives=args.objectives, seeds=args.seeds,
            learning_rate=args.learning_rate, decay=args.decay,
            message_dropout=args.message_dropout,
            batch_size=args.batch_size, ssm_tau=args.ssm_tau,
            au_uniformity_weight=args.au_uniformity_weight,
            au_uniformity_t=args.au_uniformity_t)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote cross-dataset LightGCN objective summary to %s" % args.output)


if __name__ == "__main__":
    main()
