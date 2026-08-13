"""Validate long-horizon LastFM and ML-1M LightGCN BPR/SSM runs."""

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
        report, datasets, learning_rate, temperatures, seed, decay,
        message_dropout, batch_size, max_epochs, patience):
    datasets = list(datasets)
    learning_rate = float(learning_rate)
    temperatures = sorted(float(value) for value in temperatures)
    seed = int(seed)
    max_epochs = int(max_epochs)
    patience = int(patience)
    expected = {(dataset, "bpr", None) for dataset in datasets}
    expected.update({
        (dataset, "ssm", temperature)
        for dataset in datasets for temperature in temperatures
    })
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
        metadata = row.get("training_objective_metadata") or {}
        temperature = (
            float(metadata.get("tau")) if objective == "ssm" else None)
        identity = (dataset, objective, temperature)
        if identity not in expected:
            continue
        if identity in matched:
            raise ValueError("Duplicate experiment identity: %r" % (identity,))
        if not _close(_finite(row, "train_decay"), decay):
            raise ValueError("Unexpected decay in %s" % row.get("run"))
        if int(_finite(row, "train_batch_size")) != int(batch_size):
            raise ValueError("Unexpected batch size in %s" % row.get("run"))
        expected_init = "normal" if objective == "bpr" else "xavier_uniform"
        if row.get("train_init_method") != expected_init:
            raise ValueError("Unexpected initialization in %s" % row.get("run"))
        if objective == "ssm" and not _close(
                metadata.get("message_dropout", 0.0), message_dropout):
            raise ValueError("Unexpected message dropout in %s" % row.get("run"))
        epochs_completed = int(_finite(row, "epochs_completed"))
        if epochs_completed > max_epochs:
            raise ValueError("Run exceeds max_epochs in %s" % row.get("run"))
        matched[identity] = row

    missing = sorted(expected.difference(matched), key=lambda x: (
        x[0], x[1], -1.0 if x[2] is None else x[2]))
    if missing:
        raise ValueError("Missing experiment identities: %r" % missing)

    dataset_reports = []
    for dataset in datasets:
        objectives = {}
        for objective in ("bpr", "ssm"):
            identities = sorted(
                [identity for identity in expected
                 if identity[0] == dataset and identity[1] == objective],
                key=lambda x: -1.0 if x[2] is None else x[2])
            rows = []
            for identity in identities:
                row = matched[identity]
                epochs_completed = int(_finite(row, "epochs_completed"))
                best_epoch = int(_finite(row, "best_epoch"))
                early_stopped = bool(row.get("early_stopped"))
                rows.append({
                    "objective": objective,
                    "learning_rate": learning_rate,
                    "temperature": identity[2],
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
                -1.0 if row["temperature"] is None else row["temperature"]))
            for rank, row in enumerate(rows, 1):
                row["objective_rank"] = rank
            objectives[objective] = {
                "ranking": rows,
                "best_observed": dict(rows[0]),
                "epoch_cap_count": sum(row["hit_epoch_cap"] for row in rows),
            }
        dataset_reports.append({"dataset": dataset, "objectives": objectives})

    return {
        "schema_version": "nrgcf_lastfm_ml1m_lightgcn_convergence_v1",
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
            "learning_rate": learning_rate,
            "ssm_temperatures": temperatures,
            "runs_per_dataset": 1 + len(temperatures),
            "total_run_count": len(expected),
        },
        "fixed_configuration": {
            "decay": float(decay),
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "max_epochs": max_epochs,
            "early_stopping_monitor": "test Recall@20",
            "early_stopping_patience": patience,
            "early_stopping_improvement": "strict",
            "bpr_initialization": "normal_std_0.01",
            "ssm_initialization": "xavier_uniform_gain_1",
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# LastFM/ML-1M long-horizon LightGCN BPR/SSM search",
        "",
        "Clean graph, no edge filtering, and no representation modulation.",
        "",
        "Fixed `lr={lr:g}`. Maximum `{max_epochs}` epochs; stop after "
        "`{patience}` consecutive epochs without a strict test Recall@20 "
        "improvement. Fixed `decay={decay:g}`, `dropout={dropout:g}`, "
        "`batch={batch}`.".format(
            lr=report["search_space"]["learning_rate"],
            max_epochs=fixed["max_epochs"],
            patience=fixed["early_stopping_patience"],
            decay=fixed["decay"], dropout=fixed["message_dropout"],
            batch=fixed["batch_size"]),
        "",
        "| Dataset | Objective | Rank | Tau | Recall@20 | NDCG@20 | "
        "Best epoch | Completed | Early stopped | Hit epoch cap |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for dataset_report in report["datasets"]:
        for objective in ("bpr", "ssm"):
            for row in dataset_report["objectives"][objective]["ranking"]:
                tau = "--" if row["temperature"] is None else "%g" % row["temperature"]
                lines.append(
                    "| {dataset} | {objective} | {rank} | {tau} | "
                    "{recall:.6f} | {ndcg:.6f} | {epoch} | {completed} | "
                    "{stopped} | {capped} |".format(
                        dataset=dataset_report["dataset"],
                        objective=objective.upper(), rank=row["objective_rank"],
                        tau=tau, recall=row["best_recall_at_20"],
                        ndcg=row["best_ndcg_at_20"], epoch=row["best_epoch"],
                        completed=row["epochs_completed"],
                        stopped="yes" if row["early_stopped"] else "",
                        capped="yes" if row["hit_epoch_cap"] else ""))
    lines.append("")
    for dataset_report in report["datasets"]:
        for objective in ("bpr", "ssm"):
            best = dataset_report["objectives"][objective]["best_observed"]
            tau = "" if best["temperature"] is None else ", tau=%g" % best["temperature"]
            lines.append(
                "Best {dataset} {objective}: `lr={lr:g}{tau}` with Recall@20 "
                "`{recall:.6f}` at epoch `{epoch}`.".format(
                    dataset=dataset_report["dataset"], objective=objective.upper(),
                    lr=best["learning_rate"], tau=tau,
                    recall=best["best_recall_at_20"], epoch=best["best_epoch"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--message-dropout", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream), datasets=args.datasets,
            learning_rate=args.learning_rate,
            temperatures=args.temperatures, seed=args.seed, decay=args.decay,
            message_dropout=args.message_dropout, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote LastFM/ML-1M long-horizon report to %s" % args.output)


if __name__ == "__main__":
    main()
