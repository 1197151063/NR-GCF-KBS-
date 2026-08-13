"""Analyze clean LastFM/ML-1M BPR/SSM CrossNorm sensitivity."""

import argparse
import json
import math
from pathlib import Path


def _finite(row, field):
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Missing or non-finite %s in %s" % (
            field, row.get("run", "<unknown>")))
    return float(value)


def _close(actual, expected):
    return math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)


def analyze(
        report, datasets, weights, learning_rate, decay, ssm_temperatures,
        seed, message_dropout, batch_size, max_epochs, patience):
    datasets = list(datasets)
    weights = sorted(float(value) for value in weights)
    learning_rate = float(learning_rate)
    decay = float(decay)
    ssm_temperatures = {
        str(dataset): float(value)
        for dataset, value in ssm_temperatures.items()
    }
    seed = int(seed)
    max_epochs = int(max_epochs)
    patience = int(patience)
    if not weights or not _close(weights[0], 0.0):
        raise ValueError("Weights must include the ordinary LightGCN arm at 0")
    if set(ssm_temperatures) != set(datasets):
        raise ValueError("SSM temperature mapping must match datasets")
    expected = {
        (dataset, objective, weight)
        for dataset in datasets
        for objective in ("bpr", "ssm")
        for weight in weights
    }
    matched = {}
    for row in report.get("runs", []):
        dataset = row.get("dataset")
        objective = row.get("training_objective")
        if dataset not in datasets or objective not in ("bpr", "ssm"):
            continue
        if int(row.get("seed")) != seed:
            continue
        if row.get("mode") != "none":
            raise ValueError("Edge filtering is active in %s" % row.get("run"))
        if not _close(_finite(row, "requested_noise_ratio"), 0.0):
            raise ValueError("A non-clean run was found in %s" % row.get("run"))
        if not _close(_finite(row, "train_learning_rate"), learning_rate):
            raise ValueError("Unexpected learning rate in %s" % row.get("run"))
        if not _close(_finite(row, "train_decay"), decay):
            raise ValueError("Unexpected decay in %s" % row.get("run"))
        if int(_finite(row, "train_batch_size")) != int(batch_size):
            raise ValueError("Unexpected batch size in %s" % row.get("run"))

        mode = row.get("representation_modulation_mode")
        if mode == "none":
            weight = 0.0
        elif mode == "blend_always":
            observed = _finite(row, "representation_modulation_lambda")
            matching = [value for value in weights if _close(value, observed)]
            if not matching:
                continue
            weight = matching[0]
            if _close(weight, 0.0):
                raise ValueError("Zero-weight baseline must use mode=none")
        else:
            continue
        identity = (dataset, objective, weight)
        if identity in matched:
            raise ValueError("Duplicate experiment identity: %r" % (identity,))

        expected_init = "normal" if objective == "bpr" else "xavier_uniform"
        if row.get("train_init_method") != expected_init:
            raise ValueError("Unexpected initialization in %s" % row.get("run"))
        metadata = row.get("training_objective_metadata") or {}
        if metadata.get("name") != objective:
            raise ValueError("Objective metadata mismatch in %s" % row.get("run"))
        if objective == "ssm":
            if not _close(metadata.get("tau"), ssm_temperatures[dataset]):
                raise ValueError("Unexpected SSM temperature in %s" % row.get("run"))
            if not _close(metadata.get("message_dropout", 0.0), message_dropout):
                raise ValueError("Unexpected message dropout in %s" % row.get("run"))
        epochs_completed = int(_finite(row, "epochs_completed"))
        if epochs_completed > max_epochs:
            raise ValueError("Run exceeds max_epochs in %s" % row.get("run"))
        matched[identity] = row

    missing = sorted(expected.difference(matched))
    if missing:
        raise ValueError("Missing experiment identities: %r" % missing)

    dataset_reports = []
    for dataset in datasets:
        objective_reports = []
        for objective in ("bpr", "ssm"):
            grid = []
            for weight in weights:
                row = matched[(dataset, objective, weight)]
                best_epoch = int(_finite(row, "best_epoch"))
                epochs_completed = int(_finite(row, "epochs_completed"))
                early_stopped = bool(row.get("early_stopped"))
                grid.append({
                    "arm": "lightgcn" if _close(weight, 0.0) else "blend_always",
                    "modulation_weight": weight,
                    "best_recall_at_20": _finite(row, "best_recall_at_20"),
                    "best_ndcg_at_20": _finite(row, "best_ndcg_at_20"),
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "early_stopped": early_stopped,
                    "hit_epoch_cap": (
                        not early_stopped and epochs_completed >= max_epochs),
                    "best_at_final_epoch": best_epoch == epochs_completed,
                    "final_training_loss": _finite(row, "final_training_loss"),
                    "run": row.get("run"),
                })
            baseline = grid[0]
            for row in grid:
                row["recall_gain_over_lightgcn_percent"] = (
                    row["best_recall_at_20"]
                    / baseline["best_recall_at_20"] - 1.0) * 100.0
                row["ndcg_gain_over_lightgcn_percent"] = (
                    row["best_ndcg_at_20"]
                    / baseline["best_ndcg_at_20"] - 1.0) * 100.0
            ranking = sorted(grid, key=lambda row: (
                -row["best_recall_at_20"], -row["best_ndcg_at_20"],
                row["modulation_weight"]))
            for rank, row in enumerate(ranking, 1):
                row["rank"] = rank
            best = ranking[0]
            objective_reports.append({
                "objective": objective,
                "grid": grid,
                "ranking": ranking,
                "best_observed": dict(best),
                "norm_is_helpful": best["modulation_weight"] > 0.0,
                "epoch_cap_count": sum(row["hit_epoch_cap"] for row in grid),
            })
        dataset_reports.append({
            "dataset": dataset,
            "objectives": objective_reports,
        })

    return {
        "schema_version": "nrgcf_lastfm_ml1m_lightgcn_norm_sensitivity_v1",
        "protocol": "clean_lightgcn_and_crossnorm_blend_sensitivity",
        "datasets": dataset_reports,
        "backbone": "lightgcn",
        "noise_ratio": 0.0,
        "edge_filter_mode": "none",
        "selection_split": "test",
        "selection_rule": (
            "Within each dataset/objective: Recall@20 descending, then "
            "NDCG@20 descending, then smaller modulation weight"),
        "seed": seed,
        "search_space": {
            "modulation_weights": weights,
            "learning_rate": learning_rate,
            "decay": decay,
            "ssm_temperatures": ssm_temperatures,
            "runs_per_dataset": len(weights) * 2,
            "total_run_count": len(expected),
        },
        "fixed_configuration": {
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "max_epochs": max_epochs,
            "early_stopping_monitor": "test Recall@20",
            "early_stopping_patience": patience,
            "early_stopping_improvement": "strict",
            "modulation_mode": "blend_always_for_nonzero_weights",
            "blend_definition": (
                "H_next=(1-mu)*propagated_H+mu*CrossNorm(propagated_H)"),
            "bpr_initialization": "normal_std_0.01",
            "ssm_initialization": "xavier_uniform_gain_1",
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    search = report["search_space"]
    lines = [
        "# LastFM/ML-1M LightGCN CrossNorm sensitivity",
        "",
        "Clean graph and no edge filtering. `Mu=0` is ordinary LightGCN; "
        "nonzero values use `blend_always`.",
        "",
        "Fixed `lr={lr:g}`, `decay={decay:g}`. Dataset-specific SSM "
        "temperatures: `{temps}`. Maximum `{epochs}` epochs with test "
        "Recall@20 patience `{patience}`.".format(
            lr=search["learning_rate"], decay=search["decay"],
            temps=search["ssm_temperatures"], epochs=fixed["max_epochs"],
            patience=fixed["early_stopping_patience"]),
    ]
    for dataset in report["datasets"]:
        for objective in dataset["objectives"]:
            selected = objective["best_observed"]["modulation_weight"]
            lines.extend([
                "",
                "## %s / %s" % (
                    dataset["dataset"], objective["objective"].upper()),
                "",
                "| Arm | Mu | Recall@20 | NDCG@20 | Best epoch | Completed | "
                "Recall gain | Selected |",
                "|:---|---:|---:|---:|---:|---:|---:|:---:|",
            ])
            for row in objective["grid"]:
                lines.append(
                    "| {arm} | {mu:.3g} | {recall:.6f} | {ndcg:.6f} | "
                    "{epoch} | {completed} | {gain:+.2f}% | {selected} |".format(
                        arm=("LightGCN" if _close(
                            row["modulation_weight"], 0.0) else "Blend"),
                        mu=row["modulation_weight"],
                        recall=row["best_recall_at_20"],
                        ndcg=row["best_ndcg_at_20"], epoch=row["best_epoch"],
                        completed=row["epochs_completed"],
                        gain=row["recall_gain_over_lightgcn_percent"],
                        selected=("yes" if _close(
                            row["modulation_weight"], selected) else "")))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--lastfm-ssm-temperature", type=float, required=True)
    parser.add_argument("--ml1m-ssm-temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--message-dropout", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream), datasets=args.datasets, weights=args.weights,
            learning_rate=args.learning_rate, decay=args.decay,
            ssm_temperatures={
                "lastfm": args.lastfm_ssm_temperature,
                "ml-1m": args.ml1m_ssm_temperature,
            },
            seed=args.seed, message_dropout=args.message_dropout,
            batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote LastFM/ML-1M CrossNorm sensitivity report to %s" % args.output)


if __name__ == "__main__":
    main()
