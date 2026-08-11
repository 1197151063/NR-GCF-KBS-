"""Summarize the focused SSM/CrossNorm initialization-scale diagnostic."""

import argparse
import json
from pathlib import Path


def read_manifest(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def collect(root):
    rows = []
    for summary_path in sorted(root.glob("**/training_summary.json")):
        run_dir = summary_path.parents[1]
        manifest_path = run_dir / "run_manifest.txt"
        if not manifest_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = read_manifest(manifest_path)
        objective = summary.get("training_objective", {}).get("name")
        modulation = summary.get("representation_modulation", {}).get("mode")
        if objective != "ssm" or modulation != "original_always":
            continue
        rows.append({
            "embedding_init": manifest["train_init_method"],
            "init_weight": float(manifest["train_init_weight"]),
            "decay": float(manifest["train_decay"]),
            "message_dropout": float(
                manifest["objective_message_dropout"]
            ),
            "best_epoch": int(summary["best_epoch"]),
            "best_recall_at_20": float(summary["best_recall_at_20"]),
            "best_ndcg_at_20": float(summary["best_ndcg_at_20"]),
            "final_training_loss": float(summary["final_training_loss"]),
            "epochs_completed": int(summary["epochs_completed"]),
            "run_directory": str(run_dir),
        })
    rows.sort(key=lambda row: (
        -row["best_recall_at_20"],
        -row["best_ndcg_at_20"],
        row["embedding_init"],
        row["decay"],
        row["message_dropout"],
    ))
    return rows


def markdown(rows):
    lines = [
        "# SSM + CrossNorm focused factorial diagnostic",
        "",
        "All runs use Yelp2018, seed 2020, ten epochs, in-batch SSM, "
        "and always-on direct CrossNorm.",
        "",
        "| Rank | Init | Init scale | Decay | Dropout | Best epoch | "
        "Recall@20 | NDCG@20 | Final loss |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | {embedding_init} | {init_weight:g} | {decay:g} | "
            "{message_dropout:g} | {best_epoch} | {best_recall_at_20:.6f} | "
            "{best_ndcg_at_20:.6f} | {final_training_loss:.6f} |".format(
                rank=rank, **row
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect(root)
    if len(rows) != 8:
        raise SystemExit(
            "Expected eight completed factorial runs, found %d" % len(rows)
        )
    result = {
        "schema_version": "nrgcf_ssm_crossnorm_factorial_v1",
        "controlled_factors": {
            "embedding_init": ["xavier_uniform", "normal"],
            "decay": [0.0001, 0.1],
            "message_dropout": [0.0, 0.1],
        },
        "fixed_configuration": {
            "dataset": "yelp2018",
            "seed": 2020,
            "epochs": 10,
            "training_objective": "ssm",
            "negative_sampling": "B-1 in-batch negatives",
            "representation_modulation_mode": "original_always",
        },
        "runs": rows,
        "best_run": rows[0],
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(rows), encoding="utf-8")
    print("Wrote factorial analysis to %s" % args.output)


if __name__ == "__main__":
    main()
