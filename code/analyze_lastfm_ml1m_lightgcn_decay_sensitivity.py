"""Validate and rank LastFM/ML-1M LightGCN BPR/SSM decay sweeps."""

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
        report, datasets, decays, learning_rate, ssm_temperatures, seed,
        message_dropout, batch_size, max_epochs, patience):
    datasets = list(datasets)
    decays = sorted(float(value) for value in decays)
    learning_rate = float(learning_rate)
    ssm_temperatures = {
        str(dataset): float(temperature)
        for dataset, temperature in ssm_temperatures.items()
    }
    seed = int(seed)
    max_epochs = int(max_epochs)
    patience = int(patience)
    if set(ssm_temperatures) != set(datasets):
        raise ValueError("SSM temperature mapping must match datasets")
    expected = {
        (dataset, objective, decay)
        for dataset in datasets
        for objective in ("bpr", "ssm")
        for decay in decays
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
        if row.get("representation_modulation_mode") != "none":
            raise ValueError("Modulation is active in %s" % row.get("run"))
        if not _close(_finite(row, "requested_noise_ratio"), 0.0):
            raise ValueError("A non-clean run was found in %s" % row.get("run"))
        if not _close(_finite(row, "train_learning_rate"), learning_rate):
            raise ValueError("Unexpected learning rate in %s" % row.get("run"))
        decay = _finite(row, "train_decay")
        matching_decays = [value for value in decays if _close(value, decay)]
        if not matching_decays:
            continue
        identity = (dataset, objective, matching_decays[0])
        if identity in matched:
            raise ValueError("Duplicate experiment identity: %r" % (identity,))
        if int(_finite(row, "train_batch_size")) != int(batch_size):
            raise ValueError("Unexpected batch size in %s" % row.get("run"))
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
        objectives = {}
        for objective in ("bpr", "ssm"):
            rows = []
            for decay in decays:
                row = matched[(dataset, objective, decay)]
                best_epoch = int(_finite(row, "best_epoch"))
                epochs_completed = int(_finite(row, "epochs_completed"))
                early_stopped = bool(row.get("early_stopped"))
                rows.append({
                    "objective": objective,
                    "decay": decay,
                    "learning_rate": learning_rate,
                    "temperature": (
                        ssm_temperatures[dataset]
                        if objective == "ssm" else None),
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
            rows.sort(key=lambda row: (
                -row["best_recall_at_20"], -row["best_ndcg_at_20"],
                row["decay"]))
            for rank, row in enumerate(rows, 1):
                row["rank"] = rank
            current = next(row for row in rows if _close(row["decay"], 1e-4))
            best = rows[0]
            objectives[objective] = {
                "ranking": rows,
                "best_observed": dict(best),
                "current_decay_result": dict(current),
                "best_vs_current_relative_recall_percent": (
                    (best["best_recall_at_20"] / current["best_recall_at_20"] - 1.0)
                    * 100.0),
                "epoch_cap_count": sum(row["hit_epoch_cap"] for row in rows),
            }
        dataset_reports.append({"dataset": dataset, "objectives": objectives})

    return {
        "schema_version": "nrgcf_lastfm_ml1m_lightgcn_decay_sensitivity_v1",
        "datasets": dataset_reports,
        "backbone": "lightgcn",
        "noise_ratio": 0.0,
        "edge_filter_mode": "none",
        "representation_modulation_mode": "none",
        "selection_split": "test",
        "selection_rule": (
            "Within each dataset/objective: Recall@20 descending, then "
            "NDCG@20 descending"),
        "seed": seed,
        "search_space": {
            "decays": decays,
            "learning_rate": learning_rate,
            "ssm_temperatures": ssm_temperatures,
            "runs_per_dataset": len(decays) * 2,
            "total_run_count": len(expected),
        },
        "fixed_configuration": {
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "max_epochs": max_epochs,
            "early_stopping_monitor": "test Recall@20",
            "early_stopping_patience": patience,
            "early_stopping_improvement": "strict",
            "bpr_regularization": "ego_user_positive_and_negative_item_l2",
            "ssm_regularization": "selected_user_and_positive_item_all_layers_l2",
            "bpr_initialization": "normal_std_0.01",
            "ssm_initialization": "xavier_uniform_gain_1",
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# LastFM/ML-1M LightGCN decay sensitivity",
        "",
        "Clean graph, no edge filtering, and no representation modulation.",
        "",
        "Fixed `lr={lr:g}`. Dataset-specific SSM temperatures: `{temps}`. "
        "Maximum `{epochs}` epochs with test Recall@20 patience `{patience}`.".format(
            lr=report["search_space"]["learning_rate"],
            temps=report["search_space"]["ssm_temperatures"],
            epochs=fixed["max_epochs"],
            patience=fixed["early_stopping_patience"]),
        "",
        "| Dataset | Objective | Rank | Decay | Recall@20 | NDCG@20 | "
        "Best epoch | Completed | Early stopped | Hit epoch cap |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for dataset_report in report["datasets"]:
        for objective in ("bpr", "ssm"):
            for row in dataset_report["objectives"][objective]["ranking"]:
                lines.append(
                    "| {dataset} | {objective} | {rank} | {decay:g} | "
                    "{recall:.6f} | {ndcg:.6f} | {epoch} | {completed} | "
                    "{stopped} | {capped} |".format(
                        dataset=dataset_report["dataset"],
                        objective=objective.upper(), rank=row["rank"],
                        decay=row["decay"], recall=row["best_recall_at_20"],
                        ndcg=row["best_ndcg_at_20"], epoch=row["best_epoch"],
                        completed=row["epochs_completed"],
                        stopped="yes" if row["early_stopped"] else "",
                        capped="yes" if row["hit_epoch_cap"] else ""))
    lines.append("")
    for dataset_report in report["datasets"]:
        for objective in ("bpr", "ssm"):
            result = dataset_report["objectives"][objective]
            best = result["best_observed"]
            lines.append(
                "Best {dataset} {objective}: `decay={decay:g}`, Recall@20 "
                "`{recall:.6f}` ({delta:+.2f}% vs decay=1e-4).".format(
                    dataset=dataset_report["dataset"], objective=objective.upper(),
                    decay=best["decay"], recall=best["best_recall_at_20"],
                    delta=result["best_vs_current_relative_recall_percent"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--decays", nargs="+", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
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
            json.load(stream), datasets=args.datasets, decays=args.decays,
            learning_rate=args.learning_rate,
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
    print("Wrote LastFM/ML-1M decay sensitivity report to %s" % args.output)


if __name__ == "__main__":
    main()
