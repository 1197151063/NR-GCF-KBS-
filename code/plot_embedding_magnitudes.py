#!/usr/bin/env python3
"""Plot epoch-wise embedding magnitudes before and after CrossNorm."""

import argparse
import csv
import json
from pathlib import Path


def load_rows(summary_path):
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
            "before_crossnorm": mean("before_all_mean_l2"),
            "pure_crossnorm": mean("pure_crossnorm_all_mean_l2"),
            "after_crossnorm_blend": mean("after_all_mean_l2"),
        })
    if not rows:
        raise ValueError("No layer_magnitudes found in the training summary")
    return rows, modulation.get("lambda")


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path, blend_weight):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required to draw the figure") from error

    epochs = [row["epoch"] for row in rows]
    before = [row["before_crossnorm"] for row in rows]
    after = [row["after_crossnorm_blend"] for row in rows]

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(epochs, before, marker="o", linewidth=2.2,
              color="#4C78A8", label="Before CrossNorm")
    label = "After CrossNorm blend"
    if blend_weight is not None:
        label += f" ($\\mu$={float(blend_weight):g})"
    axis.plot(epochs, after, marker="o", linewidth=2.2,
              color="#D65F4A", label=label)
    axis.set(xlabel="Epoch", ylabel="Mean embedding L2 magnitude",
             title="Yelp2018 Embedding Magnitudes")
    axis.set_xticks(epochs)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="training_summary.json from the experiment")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, blend_weight = load_rows(args.input)
    write_csv(rows, output_dir / "embedding_magnitudes.csv")
    plot(rows, output_dir / "embedding_magnitudes.png", blend_weight)
    plot(rows, output_dir / "embedding_magnitudes.pdf", blend_weight)
    print(f"Wrote magnitude data and figures to {output_dir}")


if __name__ == "__main__":
    main()
