"""Validate and rank a clean LightGCN SSM learning-rate/tau grid."""

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
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    )


def analyze(
        report, learning_rates, temperatures, seed, decay,
        message_dropout, batch_size, dataset="lastfm"):
    dataset = str(dataset)
    learning_rates = sorted(float(value) for value in learning_rates)
    temperatures = sorted(float(value) for value in temperatures)
    seed = int(seed)
    expected = {
        (learning_rate, temperature)
        for learning_rate in learning_rates
        for temperature in temperatures
    }
    matched = {}
    for row in report.get("runs", []):
        if row.get("dataset") != dataset:
            continue
        if row.get("training_objective") != "ssm":
            continue
        if int(row.get("seed")) != seed:
            continue
        if row.get("mode") != "none":
            raise ValueError("Edge filtering is active in %s" % row.get("run"))
        if row.get("representation_modulation_mode") != "none":
            raise ValueError("Modulation is active in %s" % row.get("run"))
        if not _close(_finite(row, "requested_noise_ratio"), 0.0):
            raise ValueError("A non-clean run was found in %s" % row.get("run"))
        learning_rate = _finite(row, "train_learning_rate")
        metadata = row.get("training_objective_metadata") or {}
        temperature = float(metadata.get("tau"))
        matching = [
            (expected_lr, expected_tau)
            for expected_lr, expected_tau in expected
            if _close(learning_rate, expected_lr)
            and _close(temperature, expected_tau)
        ]
        if not matching:
            continue
        identity = matching[0]
        if identity in matched:
            raise ValueError("Duplicate grid identity: %r" % (identity,))
        if not _close(_finite(row, "train_decay"), decay):
            raise ValueError("Unexpected decay in %s" % row.get("run"))
        if int(_finite(row, "train_batch_size")) != int(batch_size):
            raise ValueError("Unexpected batch size in %s" % row.get("run"))
        if row.get("train_init_method") != "xavier_uniform":
            raise ValueError("Unexpected initialization in %s" % row.get("run"))
        if not _close(metadata.get("message_dropout", 0.0), message_dropout):
            raise ValueError("Unexpected message dropout in %s" % row.get("run"))
        matched[identity] = row

    missing = sorted(expected.difference(matched))
    if missing:
        raise ValueError("Missing grid identities: %r" % missing)

    rows = []
    for learning_rate, temperature in sorted(expected):
        row = matched[(learning_rate, temperature)]
        best_epoch = int(_finite(row, "best_epoch"))
        rows.append({
            "learning_rate": learning_rate,
            "temperature": temperature,
            "best_recall_at_20": _finite(row, "best_recall_at_20"),
            "best_ndcg_at_20": _finite(row, "best_ndcg_at_20"),
            "best_epoch": best_epoch,
            "epochs_completed": int(_finite(row, "epochs_completed")),
            "early_stopped": bool(row.get("early_stopped")),
            "early_peak_flag": best_epoch <= 2,
            "final_training_loss": _finite(row, "final_training_loss"),
            "run": row.get("run"),
        })
    rows.sort(key=lambda row: (
        -row["best_recall_at_20"],
        -row["best_ndcg_at_20"],
        row["early_peak_flag"],
        row["learning_rate"],
        row["temperature"],
    ))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    best = rows[0]
    return {
        "schema_version": "nrgcf_ssm_lr_tau_grid_v2",
        "dataset": dataset,
        "training_objective": "ssm",
        "backbone": "lightgcn",
        "noise_ratio": 0.0,
        "edge_filter_mode": "none",
        "representation_modulation_mode": "none",
        "seed": seed,
        "search_space": {
            "learning_rates": learning_rates,
            "temperatures": temperatures,
            "combination_count": len(expected),
        },
        "fixed_configuration": {
            "decay": float(decay),
            "message_dropout": float(message_dropout),
            "batch_size": int(batch_size),
            "embedding_initialization": "xavier_uniform_gain_1",
        },
        "selection_split": "test",
        "selection_rule": (
            "Recall@20 descending, then NDCG@20 descending; early-peak flag "
            "is diagnostic and only breaks exact metric ties"
        ),
        "early_peak_definition": "best_epoch <= 2",
        "early_peak_count": sum(row["early_peak_flag"] for row in rows),
        "ranking": rows,
        "best_observed": {
            key: best[key] for key in (
                "learning_rate", "temperature", "best_recall_at_20",
                "best_ndcg_at_20", "best_epoch", "early_peak_flag"
            )
        },
    }


def markdown(report):
    fixed = report["fixed_configuration"]
    dataset_label = {
        "lastfm": "LastFM",
        "ml-1m": "ML-1M",
    }.get(report["dataset"], report["dataset"])
    lines = [
        "# %s LightGCN SSM learning-rate/temperature grid" % dataset_label,
        "",
        "Clean graph, no edge filtering, and no representation modulation.",
        "",
        "Fixed: `decay={decay:g}`, `dropout={dropout:g}`, `batch={batch}`. "
        "An early peak means `best_epoch <= 2`.".format(
            decay=fixed["decay"], dropout=fixed["message_dropout"],
            batch=fixed["batch_size"]),
        "",
        "| Rank | LR | Tau | Recall@20 | NDCG@20 | Best epoch | "
        "Completed | Early peak |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["ranking"]:
        lines.append(
            "| {rank} | {learning_rate:g} | {temperature:g} | "
            "{best_recall_at_20:.6f} | {best_ndcg_at_20:.6f} | "
            "{best_epoch} | {epochs_completed} | {early} |".format(
                early="yes" if row["early_peak_flag"] else "",
                **row))
    best = report["best_observed"]
    lines.extend([
        "",
        "Best observed: `lr={learning_rate:g}, tau={temperature:g}` with "
        "Recall@20 `{best_recall_at_20:.6f}` at epoch `{best_epoch}`.".format(
            **best),
        "",
        "Early-peak configurations: `{count}/{total}`.".format(
            count=report["early_peak_count"],
            total=report["search_space"]["combination_count"]),
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="lastfm")
    parser.add_argument("--learning-rates", nargs="+", type=float, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--message-dropout", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as stream:
        result = analyze(
            json.load(stream), learning_rates=args.learning_rates,
            temperatures=args.temperatures, seed=args.seed,
            decay=args.decay, message_dropout=args.message_dropout,
            batch_size=args.batch_size, dataset=args.dataset)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    best = result["best_observed"]
    print("Best %s SSM: lr=%g tau=%g Recall@20=%.6f epoch=%d" % (
        result["dataset"], best["learning_rate"], best["temperature"],
        best["best_recall_at_20"], best["best_epoch"]))


if __name__ == "__main__":
    main()
