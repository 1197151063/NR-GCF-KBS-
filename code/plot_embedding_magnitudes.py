#!/usr/bin/env python3
"""Compare LightGCN and LightGCN+Norm embedding magnitudes."""

import argparse
import csv
import json
from pathlib import Path


def load_run(summary_path):
    with open(summary_path, encoding="utf-8") as stream:
        summary = json.load(stream)
    modulation = summary.get("representation_modulation", {})
    rows = []
    for epoch_trace in modulation.get("trace", []):
        layers = epoch_trace.get("layer_magnitudes", [])
        if not layers:
            continue
        mean = lambda key: sum(float(x[key]) for x in layers) / len(layers)
        rows.append({
            "epoch": int(epoch_trace["epoch"]),
            "user": mean("after_user_mean_l2"),
            "item": mean("after_item_mean_l2"),
        })
    if not rows:
        raise ValueError("No layer_magnitudes found in the training summary")
    return rows


def compare_rows(lightgcn, norm):
    if [x["epoch"] for x in lightgcn] != [x["epoch"] for x in norm]:
        raise ValueError("LightGCN and Norm traces must contain the same epochs")
    return [
        {
            "epoch": baseline["epoch"],
            "user": baseline["user"],
            "user_norm": calibrated["user"],
            "item": baseline["item"],
            "item_norm": calibrated["item"],
        }
        for baseline, calibrated in zip(lightgcn, norm)
    ]


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required to draw the figure") from error

    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(4.4, 3.0))
    styles = (
        ("user", "User", "#4C78A8", "--"),
        ("user_norm", "User + Norm", "#4C78A8", "-"),
        ("item", "Item", "#E08B35", "--"),
        ("item_norm", "Item + Norm", "#E08B35", "-"),
    )
    for key, label, color, line_style in styles:
        axis.plot(epochs, [row[key] for row in rows], linewidth=2.0,
                  linestyle=line_style, color=color, label=label)
    axis.set(xlabel="Epoch", ylabel="Magnitude")
    axis.set_xticks(epochs)
    axis.set_ylim(bottom=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightgcn", required=True)
    parser.add_argument("--norm", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = compare_rows(load_run(args.lightgcn), load_run(args.norm))
    write_csv(rows, output_dir / "embedding_magnitudes.csv")
    plot(rows, output_dir / "embedding_magnitudes.png")
    plot(rows, output_dir / "embedding_magnitudes.pdf")
    print(f"Wrote magnitude data and figures to {output_dir}")


if __name__ == "__main__":
    main()
