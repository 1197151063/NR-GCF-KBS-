"""Extract the matched clean/noisy SSM + CrossNorm confirmation pair."""

import argparse
import json
import math
from pathlib import Path


def analyze(report, noise_ratios, seed):
    targets = [float(value) for value in noise_ratios]
    rows = []
    for target in targets:
        matched = [
            row for row in report.get("runs", [])
            if row.get("dataset") == "yelp2018"
            and row.get("mode") == "none"
            and row.get("training_objective") == "ssm"
            and row.get("representation_modulation_mode") == "original_always"
            and int(row.get("seed", -1)) == int(seed)
            and math.isclose(
                float(row.get("requested_noise_ratio", -1)), target,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ]
        if len(matched) != 1:
            raise ValueError(
                "Expected one SSM/CrossNorm row at noise %.3g, found %d"
                % (target, len(matched))
            )
        rows.append(matched[0])
    return {
        "schema_version": "nrgcf_ssm_crossnorm_clean_noisy_v1",
        "dataset": "yelp2018",
        "seed": int(seed),
        "configuration": {
            "embedding_init": "xavier_uniform",
            "decay": 1e-5,
            "learning_rate": 1e-5,
            "temperature": 0.09,
            "message_dropout": 0.1,
            "maximum_epochs": 100,
            "early_stopping_patience": 20,
        },
        "runs": rows,
    }


def markdown(report):
    lines = [
        "# SSM + CrossNorm clean/noisy confirmation",
        "",
        "Xavier Uniform, decay `1e-5`, learning rate `1e-5`, tau `0.09`, "
        "message dropout `0.1`, seed `2020`.",
        "",
        "| Requested noise | Actual noise | Best epoch | Recall@20 | NDCG@20 | "
        "Epochs completed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            "| {requested:.2f} | {actual:.4f} | {epoch} | {recall:.6f} | "
            "{ndcg:.6f} | {completed} |".format(
                requested=float(row["requested_noise_ratio"]),
                actual=float(row["actual_noise_ratio"]),
                epoch=int(row["best_epoch"]),
                recall=float(row["best_recall_at_20"]),
                ndcg=float(row["best_ndcg_at_20"]),
                completed=int(row["epochs_completed"]),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--noise-ratios", default="0 0.2")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    report = analyze(
        json.loads(Path(args.input).read_text(encoding="utf-8")),
        args.noise_ratios.split(), args.seed,
    )
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(report), encoding="utf-8")
    print("Wrote clean/noisy confirmation to %s" % args.output)


if __name__ == "__main__":
    main()
