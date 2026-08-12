"""Analyze a clean-data SSM propagation/CrossNorm blend sensitivity grid."""

import argparse
import json
import math
import statistics
from collections import defaultdict
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


def analyze(report, dataset, expected_weights, expected_seeds,
            learning_rate, temperature, decay, message_dropout, batch_size):
    grouped = defaultdict(list)
    identities = set()
    for row in report.get("runs", []):
        if row.get("dataset") != dataset:
            continue
        if row.get("mode") != "none":
            continue
        if row.get("training_objective") != "ssm":
            continue
        if row.get("representation_modulation_mode") != "blend_always":
            continue
        ratio = _finite(row, "requested_noise_ratio")
        if not math.isclose(ratio, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        weight = _finite(row, "representation_modulation_lambda")
        seed = int(row.get("seed"))
        identity = (weight, seed)
        if identity in identities:
            raise ValueError("Duplicate modulation identity: %r" % (identity,))
        identities.add(identity)
        objective = row.get("training_objective_metadata") or {}
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
        if not math.isclose(
                float(objective.get("tau")), float(temperature),
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unexpected SSM temperature in %s" % row.get("run"))
        if not math.isclose(
                float(objective.get("message_dropout")),
                float(message_dropout), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unexpected message dropout in %s" % row.get("run"))
        grouped[weight].append({
            "seed": seed,
            "recall": _finite(row, "best_recall_at_20"),
            "ndcg": _finite(row, "best_ndcg_at_20"),
            "best_epoch": int(_finite(row, "best_epoch")),
            "epochs_completed": int(_finite(row, "epochs_completed")),
            "early_stopped": bool(row.get("early_stopped")),
            "run": row.get("run"),
        })

    expected_weights = sorted(float(value) for value in expected_weights)
    expected_seeds = sorted(int(value) for value in expected_seeds)
    if sorted(grouped) != expected_weights:
        raise ValueError(
            "Expected modulation weights %s, found %s"
            % (expected_weights, sorted(grouped)))

    rows = []
    for weight in expected_weights:
        runs = sorted(grouped[weight], key=lambda item: item["seed"])
        seeds = [item["seed"] for item in runs]
        if seeds != expected_seeds:
            raise ValueError(
                "Weight %s expected seeds %s, found %s"
                % (weight, expected_seeds, seeds))
        rows.append({
            "modulation_weight": weight,
            "seeds": seeds,
            "recall_at_20": _aggregate([item["recall"] for item in runs]),
            "ndcg_at_20": _aggregate([item["ndcg"] for item in runs]),
            "best_epoch": _aggregate([item["best_epoch"] for item in runs]),
            "epochs_completed": _aggregate([
                item["epochs_completed"] for item in runs
            ]),
            "all_early_stopped": all(item["early_stopped"] for item in runs),
            "runs": [item["run"] for item in runs],
        })

    best = max(rows, key=lambda row: (
        row["recall_at_20"]["mean"],
        row["ndcg_at_20"]["mean"],
        -row["modulation_weight"],
    ))
    return {
        "schema_version": "nrgcf_ssm_clean_modulation_sensitivity_v1",
        "dataset": dataset,
        "noise_ratio": 0.0,
        "training_objective": "ssm",
        "modulation_mode": "blend_always",
        "modulation_definition": (
            "H_next=(1-mu)*propagated_H+mu*CrossNorm(propagated_H)"
        ),
        "fixed_configuration": {
            "learning_rate": float(learning_rate),
            "temperature": float(temperature),
            "decay": float(decay),
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "embedding_initialization": "xavier_uniform_gain_1",
            "edge_filter_mode": "none",
        },
        "selection_split": "test",
        "exploratory_sensitivity": True,
        "selection_rule": (
            "maximum mean Recall@20; tie by mean NDCG@20, then smaller mu"
        ),
        "grid": rows,
        "best_observed": {
            "modulation_weight": best["modulation_weight"],
            "mean_recall_at_20": best["recall_at_20"]["mean"],
            "sample_std_recall_at_20": best["recall_at_20"]["sample_std"],
            "mean_ndcg_at_20": best["ndcg_at_20"]["mean"],
            "sample_std_ndcg_at_20": best["ndcg_at_20"]["sample_std"],
            "seeds": best["seeds"],
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# Yelp2018 clean SSM modulation-weight sensitivity",
        "",
        "All arms use `blend_always`, no edge filtering, and",
        "`H_next=(1-mu)*propagated_H+mu*CrossNorm(propagated_H)`.",
        "",
        "Fixed configuration: `lr={lr:g}`, `tau={tau:g}`, `decay={decay:g}`, "
        "`dropout={dropout:g}`, `batch_size={batch}`.".format(
            lr=fixed["learning_rate"], tau=fixed["temperature"],
            decay=fixed["decay"], dropout=fixed["message_dropout"],
            batch=fixed["batch_size"]),
        "",
        "| Mu | Seeds | Recall@20 | NDCG@20 | Best epoch | Selected |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    selected = float(report["best_observed"]["modulation_weight"])
    for row in report["grid"]:
        lines.append(
            "| {mu:.3g} | {seeds} | {recall:.6f} $\\pm$ {rstd:.6f} | "
            "{ndcg:.6f} $\\pm$ {nstd:.6f} | {epoch:.2f} | {chosen} |".format(
                mu=row["modulation_weight"], seeds=len(row["seeds"]),
                recall=row["recall_at_20"]["mean"],
                rstd=row["recall_at_20"]["sample_std"],
                ndcg=row["ndcg_at_20"]["mean"],
                nstd=row["ndcg_at_20"]["sample_std"],
                epoch=row["best_epoch"]["mean"],
                chosen=("yes" if math.isclose(
                    row["modulation_weight"], selected,
                    rel_tol=0.0, abs_tol=1e-12) else "")))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="yelp2018")
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--message-dropout", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        report = analyze(
            json.load(stream), dataset=args.dataset,
            expected_weights=args.weights, expected_seeds=args.seeds,
            learning_rate=args.learning_rate, temperature=args.temperature,
            decay=args.decay, message_dropout=args.message_dropout,
            batch_size=args.batch_size)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(report), encoding="utf-8")
    print("Best observed clean SSM modulation weight: %s" % (
        report["best_observed"]["modulation_weight"]))


if __name__ == "__main__":
    main()
