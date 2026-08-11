"""Rank the two-stage Yelp SSM/CrossNorm hyperparameter grid."""

import argparse
import json
import math
from pathlib import Path


def read_manifest(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def collect(root):
    rows = []
    for summary_path in sorted(root.glob("**/training_summary.json")):
        run_dir = summary_path.parents[1]
        manifest = read_manifest(run_dir / "run_manifest.txt")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        objective = (summary.get("training_objective") or {}).get("name")
        modulation = (summary.get("representation_modulation") or {}).get("mode")
        if objective != "ssm" or modulation != "original_always":
            continue
        required = ("train_lr", "train_decay", "ssm_tau")
        if any(not manifest.get(name) for name in required):
            continue
        rows.append({
            "learning_rate": float(manifest["train_lr"]),
            "temperature": float(manifest["ssm_tau"]),
            "decay": float(manifest["train_decay"]),
            "requested_noise_ratio": float(
                manifest.get("requested_noise_ratio", 0.0)
            ),
            "seed": int(manifest["seed"]),
            "best_epoch": int(summary["best_epoch"]),
            "epochs_completed": int(summary["epochs_completed"]),
            "best_recall_at_20": float(summary["best_recall_at_20"]),
            "best_ndcg_at_20": float(summary["best_ndcg_at_20"]),
            "final_training_loss": float(summary["final_training_loss"]),
            "run_directory": str(run_dir),
        })
    rows.sort(key=lambda row: (
        -row["best_recall_at_20"],
        -row["best_ndcg_at_20"],
        row["learning_rate"],
        row["temperature"],
        row["decay"],
    ))
    return rows


def key(row):
    return (
        row["learning_rate"],
        row["temperature"],
        row["decay"],
    )


def robust_ranking(clean_rows, noisy_rows):
    clean = {key(row): row for row in clean_rows}
    noisy = {key(row): row for row in noisy_rows}
    paired_keys = sorted(set(clean) & set(noisy))
    if not paired_keys:
        return []
    best_clean_recall = max(clean[item]["best_recall_at_20"] for item in paired_keys)
    best_noisy_recall = max(noisy[item]["best_recall_at_20"] for item in paired_keys)
    best_clean_ndcg = max(clean[item]["best_ndcg_at_20"] for item in paired_keys)
    best_noisy_ndcg = max(noisy[item]["best_ndcg_at_20"] for item in paired_keys)
    rows = []
    for item in paired_keys:
        clean_row = clean[item]
        noisy_row = noisy[item]
        clean_recall_relative = clean_row["best_recall_at_20"] / best_clean_recall
        noisy_recall_relative = noisy_row["best_recall_at_20"] / best_noisy_recall
        clean_ndcg_relative = clean_row["best_ndcg_at_20"] / best_clean_ndcg
        noisy_ndcg_relative = noisy_row["best_ndcg_at_20"] / best_noisy_ndcg
        rows.append({
            "learning_rate": item[0],
            "temperature": item[1],
            "decay": item[2],
            "clean_best_epoch": clean_row["best_epoch"],
            "clean_recall_at_20": clean_row["best_recall_at_20"],
            "clean_ndcg_at_20": clean_row["best_ndcg_at_20"],
            "noisy_best_epoch": noisy_row["best_epoch"],
            "noisy_recall_at_20": noisy_row["best_recall_at_20"],
            "noisy_ndcg_at_20": noisy_row["best_ndcg_at_20"],
            "mean_relative_recall": (
                clean_recall_relative + noisy_recall_relative
            ) / 2.0,
            "geometric_mean_relative_recall": math.sqrt(
                clean_recall_relative * noisy_recall_relative
            ),
            "mean_relative_ndcg": (
                clean_ndcg_relative + noisy_ndcg_relative
            ) / 2.0,
        })
    rows.sort(key=lambda row: (
        -row["geometric_mean_relative_recall"],
        -row["mean_relative_ndcg"],
        -row["clean_recall_at_20"],
        row["learning_rate"],
        row["temperature"],
        row["decay"],
    ))
    return rows


def write_selected(path, rows, top_k):
    selected = rows[:top_k]
    text = "".join(
        "{learning_rate:g}\t{temperature:g}\t{decay:g}\n".format(**row)
        for row in selected
    )
    path.write_text(text, encoding="utf-8")
    return selected


def markdown(report):
    lines = [
        "# Yelp SSM + CrossNorm hyperparameter grid",
        "",
        "The clean grid ranks configurations by Recall@20 (NDCG@20 as the "
        "tie-breaker). The top clean configurations are validated under 0.2 "
        "degree-preserving replacement noise. The final score is the geometric "
        "mean of clean/noisy Recall@20 after normalizing each condition by its "
        "best validated result.",
        "",
        "## Clean grid",
        "",
        "| Rank | Learning rate | Tau | Decay | Best epoch | Recall@20 | NDCG@20 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["clean_ranking"], 1):
        lines.append(
            "| {rank} | {learning_rate:g} | {temperature:g} | {decay:g} | "
            "{best_epoch} | {best_recall_at_20:.6f} | {best_ndcg_at_20:.6f} |".format(
                rank=rank, **row
            )
        )
    lines.extend([
        "",
        "## Clean/noisy validation",
        "",
        "| Rank | Learning rate | Tau | Decay | Clean R@20 | Noisy R@20 | "
        "Clean N@20 | Noisy N@20 | Robust score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(report["robust_ranking"], 1):
        lines.append(
            "| {rank} | {learning_rate:g} | {temperature:g} | {decay:g} | "
            "{clean_recall_at_20:.6f} | {noisy_recall_at_20:.6f} | "
            "{clean_ndcg_at_20:.6f} | {noisy_ndcg_at_20:.6f} | "
            "{geometric_mean_relative_recall:.6f} |".format(rank=rank, **row)
        )
    if report["recommended_configuration"] is not None:
        best = report["recommended_configuration"]
        lines.extend([
            "",
            "## Recommended configuration",
            "",
            "`lr={learning_rate:g}, tau={temperature:g}, decay={decay:g}`".format(
                **best
            ),
        ])
    return "\n".join(lines) + "\n"


def analyze(root, learning_rates, temperatures, decays, seed, top_k):
    root = Path(root)
    clean_rows = collect(root / "clean_grid")
    noisy_rows = collect(root / "noisy_validation")
    clean_rows = [row for row in clean_rows if row["requested_noise_ratio"] == 0.0]
    noisy_rows = [
        row for row in noisy_rows
        if abs(row["requested_noise_ratio"] - 0.2) < 1e-9
    ]
    robust_rows = robust_ranking(clean_rows, noisy_rows)
    return {
        "schema_version": "nrgcf_ssm_yelp_hyperparameter_grid_v1",
        "dataset": "yelp2018",
        "seed": int(seed),
        "search_space": {
            "learning_rates": learning_rates,
            "temperatures": temperatures,
            "decays": decays,
            "clean_combination_count": (
                len(learning_rates) * len(temperatures) * len(decays)
            ),
            "noisy_validation_top_k": int(top_k),
        },
        "fixed_configuration": {
            "training_objective": "ssm",
            "embedding_initialization": "xavier_uniform_gain_1",
            "representation_modulation_mode": "original_always",
            "message_dropout": 0.1,
            "noise_mode": "degree_preserving_replace",
            "noisy_validation_ratio": 0.2,
        },
        "selection_rule": {
            "clean_stage": "Recall@20 descending, then NDCG@20 descending",
            "final_stage": (
                "geometric mean of condition-normalized clean and noisy Recall@20; "
                "condition-normalized mean NDCG@20 is the tie-breaker"
            ),
        },
        "clean_completed_count": len(clean_rows),
        "noisy_validation_completed_count": len(noisy_rows),
        "clean_ranking": clean_rows,
        "robust_ranking": robust_rows,
        "recommended_configuration": robust_rows[0] if robust_rows else None,
    }


def floats(value):
    return [float(item) for item in value.split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--learning-rates", required=True)
    parser.add_argument("--temperatures", required=True)
    parser.add_argument("--decays", required=True)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    report = analyze(
        args.root,
        floats(args.learning_rates),
        floats(args.temperatures),
        floats(args.decays),
        args.seed,
        args.top_k,
    )
    if not report["clean_ranking"]:
        raise SystemExit("No completed clean-grid runs were found.")
    write_selected(Path(args.selected), report["clean_ranking"], args.top_k)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(report), encoding="utf-8")
    print(
        "Collected %d clean and %d noisy runs."
        % (
            report["clean_completed_count"],
            report["noisy_validation_completed_count"],
        )
    )


if __name__ == "__main__":
    main()
