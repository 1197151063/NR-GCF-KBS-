"""Analyze LastFM/ML-1M AU CrossNorm blend sensitivity."""

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
        report, datasets, modulation_weights, au_weights, learning_rate,
        configured_decay, uniformity_t, seed, message_dropout, batch_size,
        max_epochs, patience):
    datasets = list(datasets)
    modulation_weights = sorted(float(value) for value in modulation_weights)
    au_weights = {
        str(dataset): float(value) for dataset, value in au_weights.items()
    }
    learning_rate = float(learning_rate)
    configured_decay = float(configured_decay)
    uniformity_t = float(uniformity_t)
    seed = int(seed)
    max_epochs = int(max_epochs)
    patience = int(patience)
    if not modulation_weights or not _close(modulation_weights[0], 0.0):
        raise ValueError("Modulation weights must include 0")
    if set(au_weights) != set(datasets):
        raise ValueError("AU weight mapping must match datasets")
    expected = {
        (dataset, weight)
        for dataset in datasets for weight in modulation_weights
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

        mode = row.get("representation_modulation_mode")
        if mode == "none":
            modulation_weight = 0.0
        elif mode == "blend_always":
            observed = _finite(row, "representation_modulation_lambda")
            matching = [
                value for value in modulation_weights
                if _close(value, observed)
            ]
            if not matching:
                continue
            modulation_weight = matching[0]
            if _close(modulation_weight, 0.0):
                raise ValueError("Zero-weight baseline must use mode=none")
        else:
            continue
        identity = (dataset, modulation_weight)
        if identity in matched:
            raise ValueError("Duplicate experiment identity: %r" % (identity,))

        metadata = row.get("training_objective_metadata") or {}
        if metadata.get("name") != "au":
            raise ValueError("Objective metadata mismatch in %s" % row.get("run"))
        if metadata.get("regularization") != "none":
            raise ValueError("AU unexpectedly uses decay in %s" % row.get("run"))
        if not _close(metadata.get("uniformity_weight"), au_weights[dataset]):
            raise ValueError("Unexpected AU weight in %s" % row.get("run"))
        if not _close(metadata.get("uniformity_t"), uniformity_t):
            raise ValueError("Unexpected uniformity t in %s" % row.get("run"))
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
        grid = []
        for modulation_weight in modulation_weights:
            row = matched[(dataset, modulation_weight)]
            best_epoch = int(_finite(row, "best_epoch"))
            epochs_completed = int(_finite(row, "epochs_completed"))
            early_stopped = bool(row.get("early_stopped"))
            grid.append({
                "arm": (
                    "lightgcn" if _close(modulation_weight, 0.0)
                    else "blend_always"),
                "modulation_weight": modulation_weight,
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
        dataset_reports.append({
            "dataset": dataset,
            "au_uniformity_weight": au_weights[dataset],
            "grid": grid,
            "ranking": ranking,
            "best_observed": dict(best),
            "norm_is_helpful": best["modulation_weight"] > 0.0,
            "epoch_cap_count": sum(row["hit_epoch_cap"] for row in grid),
        })

    return {
        "schema_version": "nrgcf_lastfm_ml1m_au_norm_sensitivity_v1",
        "protocol": "clean_au_lightgcn_and_crossnorm_blend_sensitivity",
        "datasets": dataset_reports,
        "backbone": "lightgcn",
        "training_objective": "au",
        "noise_ratio": 0.0,
        "edge_filter_mode": "none",
        "selection_split": "test",
        "selection_rule": (
            "Within each dataset: Recall@20 descending, then NDCG@20 "
            "descending, then smaller modulation weight"),
        "seed": seed,
        "search_space": {
            "modulation_weights": modulation_weights,
            "au_uniformity_weights": au_weights,
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
            "modulation_mode": "blend_always_for_nonzero_weights",
            "blend_definition": (
                "H_next=(1-mu)*propagated_H+mu*CrossNorm(propagated_H)"),
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    lines = [
        "# LastFM/ML-1M AU CrossNorm sensitivity",
        "",
        "Clean graph and no edge filtering. `Mu=0` is ordinary LightGCN; "
        "nonzero values use `blend_always`.",
        "",
        "Fixed `lr={lr:g}`, alignment weight `1`, uniformity kernel "
        "`t={t:g}`. Dataset-specific AU weights: `{weights}`. Maximum "
        "`{epochs}` epochs with test Recall@20 patience `{patience}`.".format(
            lr=fixed["learning_rate"], t=fixed["uniformity_t"],
            weights=report["search_space"]["au_uniformity_weights"],
            epochs=fixed["max_epochs"],
            patience=fixed["early_stopping_patience"]),
    ]
    for dataset in report["datasets"]:
        selected = dataset["best_observed"]["modulation_weight"]
        lines.extend([
            "",
            "## %s" % dataset["dataset"],
            "",
            "| Arm | Mu | Recall@20 | NDCG@20 | Best epoch | Completed | "
            "Recall gain | Selected |",
            "|:---|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for row in dataset["grid"]:
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
    parser.add_argument("--modulation-weights", nargs="+", type=float, required=True)
    parser.add_argument("--lastfm-au-weight", type=float, required=True)
    parser.add_argument("--ml1m-au-weight", type=float, required=True)
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
            json.load(stream), datasets=args.datasets,
            modulation_weights=args.modulation_weights,
            au_weights={
                "lastfm": args.lastfm_au_weight,
                "ml-1m": args.ml1m_au_weight,
            },
            learning_rate=args.learning_rate,
            configured_decay=args.configured_decay,
            uniformity_t=args.uniformity_t, seed=args.seed,
            message_dropout=args.message_dropout, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote LastFM/ML-1M AU CrossNorm report to %s" % args.output)


if __name__ == "__main__":
    main()
