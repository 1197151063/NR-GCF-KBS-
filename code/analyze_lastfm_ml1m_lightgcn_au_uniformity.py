"""Validate LastFM/ML-1M LightGCN AU uniformity-weight sensitivity."""

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
        report, datasets, weights, learning_rate, configured_decay,
        uniformity_t, seed, message_dropout, batch_size, max_epochs,
        patience):
    datasets = list(datasets)
    weights = sorted(float(value) for value in weights)
    learning_rate = float(learning_rate)
    configured_decay = float(configured_decay)
    uniformity_t = float(uniformity_t)
    seed = int(seed)
    max_epochs = int(max_epochs)
    patience = int(patience)
    expected = {
        (dataset, weight) for dataset in datasets for weight in weights
    }
    matched = {}
    for row in report.get("runs", []):
        dataset = row.get("dataset")
        if dataset not in datasets or row.get("training_objective") != "au":
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
        if not _close(_finite(row, "train_decay"), configured_decay):
            raise ValueError("Unexpected configured decay in %s" % row.get("run"))
        if int(_finite(row, "train_batch_size")) != int(batch_size):
            raise ValueError("Unexpected batch size in %s" % row.get("run"))
        if row.get("train_init_method") != "xavier_uniform":
            raise ValueError("Unexpected initialization in %s" % row.get("run"))
        metadata = row.get("training_objective_metadata") or {}
        if metadata.get("name") != "au":
            raise ValueError("Objective metadata mismatch in %s" % row.get("run"))
        if metadata.get("regularization") != "none":
            raise ValueError("AU unexpectedly uses decay in %s" % row.get("run"))
        if not _close(metadata.get("uniformity_t"), uniformity_t):
            raise ValueError("Unexpected uniformity t in %s" % row.get("run"))
        if not _close(metadata.get("message_dropout", 0.0), message_dropout):
            raise ValueError("Unexpected message dropout in %s" % row.get("run"))
        observed_weight = float(metadata.get("uniformity_weight"))
        matching = [value for value in weights if _close(value, observed_weight)]
        if not matching:
            continue
        identity = (dataset, matching[0])
        if identity in matched:
            raise ValueError("Duplicate experiment identity: %r" % (identity,))
        epochs_completed = int(_finite(row, "epochs_completed"))
        if epochs_completed > max_epochs:
            raise ValueError("Run exceeds max_epochs in %s" % row.get("run"))
        matched[identity] = row

    missing = sorted(expected.difference(matched))
    if missing:
        raise ValueError("Missing experiment identities: %r" % missing)

    dataset_reports = []
    for dataset in datasets:
        rows = []
        for weight in weights:
            row = matched[(dataset, weight)]
            best_epoch = int(_finite(row, "best_epoch"))
            epochs_completed = int(_finite(row, "epochs_completed"))
            early_stopped = bool(row.get("early_stopped"))
            rows.append({
                "uniformity_weight": weight,
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
            row["uniformity_weight"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        dataset_reports.append({
            "dataset": dataset,
            "ranking": rows,
            "best_observed": dict(rows[0]),
            "epoch_cap_count": sum(row["hit_epoch_cap"] for row in rows),
        })

    return {
        "schema_version": "nrgcf_lastfm_ml1m_lightgcn_au_uniformity_v1",
        "protocol": "clean_lightgcn_au_uniformity_weight_sensitivity",
        "datasets": dataset_reports,
        "backbone": "lightgcn",
        "training_objective": "au",
        "noise_ratio": 0.0,
        "edge_filter_mode": "none",
        "representation_modulation_mode": "none",
        "selection_split": "test",
        "selection_rule": (
            "Within each dataset: Recall@20 descending, then NDCG@20 "
            "descending, then smaller uniformity weight"),
        "seed": seed,
        "search_space": {
            "uniformity_weights": weights,
            "total_run_count": len(expected),
        },
        "fixed_configuration": {
            "learning_rate": learning_rate,
            "configured_decay": configured_decay,
            "configured_decay_effect": "none_for_au_objective",
            "alignment_weight": 1.0,
            "uniformity_t": uniformity_t,
            "uniformity_sides": "user_plus_item",
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "embedding_initialization": "xavier_uniform_gain_1",
            "max_epochs": max_epochs,
            "early_stopping_monitor": "test Recall@20",
            "early_stopping_patience": patience,
            "early_stopping_improvement": "strict",
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# LastFM/ML-1M LightGCN AU uniformity-weight sensitivity",
        "",
        "Clean ordinary LightGCN, no CrossNorm, and no edge filtering.",
        "",
        "Fixed alignment weight `1`, uniformity kernel `t={t:g}`, "
        "`lr={lr:g}`, `batch={batch}`. The configured decay is "
        "`{decay:g}` but AU metadata confirms `regularization=none`. Maximum "
        "`{epochs}` epochs with test Recall@20 patience `{patience}`.".format(
            t=fixed["uniformity_t"], lr=fixed["learning_rate"],
            batch=fixed["batch_size"], decay=fixed["configured_decay"],
            epochs=fixed["max_epochs"],
            patience=fixed["early_stopping_patience"]),
        "",
        "| Dataset | Rank | Uniformity weight | Recall@20 | NDCG@20 | "
        "Best epoch | Completed | Early stopped | Hit epoch cap |",
        "|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for dataset in report["datasets"]:
        for row in dataset["ranking"]:
            lines.append(
                "| {dataset} | {rank} | {weight:g} | {recall:.6f} | "
                "{ndcg:.6f} | {epoch} | {completed} | {stopped} | "
                "{capped} |".format(
                    dataset=dataset["dataset"], rank=row["rank"],
                    weight=row["uniformity_weight"],
                    recall=row["best_recall_at_20"],
                    ndcg=row["best_ndcg_at_20"], epoch=row["best_epoch"],
                    completed=row["epochs_completed"],
                    stopped="yes" if row["early_stopped"] else "",
                    capped="yes" if row["hit_epoch_cap"] else ""))
    lines.append("")
    for dataset in report["datasets"]:
        best = dataset["best_observed"]
        lines.append(
            "Best {dataset}: uniformity weight `{weight:g}`, Recall@20 "
            "`{recall:.6f}` at epoch `{epoch}`.".format(
                dataset=dataset["dataset"],
                weight=best["uniformity_weight"],
                recall=best["best_recall_at_20"], epoch=best["best_epoch"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--configured-decay", type=float, required=True)
    parser.add_argument("--uniformity-t", type=float, required=True)
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
            learning_rate=args.learning_rate,
            configured_decay=args.configured_decay,
            uniformity_t=args.uniformity_t, seed=args.seed,
            message_dropout=args.message_dropout, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote LastFM/ML-1M AU uniformity report to %s" % args.output)


if __name__ == "__main__":
    main()
