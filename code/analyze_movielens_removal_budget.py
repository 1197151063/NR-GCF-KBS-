"""Summarize the focused MovieLens hard-filter removal-budget pilot."""

import argparse
import json
import math
from pathlib import Path


def _load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _finite(row, field):
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Missing/non-finite %s in %s" % (field, row.get("run")))
    return float(value)


def analyze(
        report, clean_ratio=0.0, noisy_ratio=0.2, clean_tolerance=0.002,
        modulation_lambda=None):
    rows = report.get("runs", [])
    baselines = {}
    arms = {}
    for row in rows:
        if row.get("dataset") != "ml-1m":
            continue
        if modulation_lambda is not None:
            row_lambda = row.get("representation_modulation_lambda")
            if row_lambda is None or not math.isclose(
                    float(row_lambda), float(modulation_lambda),
                    rel_tol=0.0, abs_tol=1e-12):
                continue
        ratio = float(row.get("requested_noise_ratio"))
        if row.get("mode") == "none":
            if ratio in baselines:
                raise ValueError("Duplicate no-filter baseline for ratio %s" % ratio)
            baselines[ratio] = row
        elif row.get("mode") == "hard_structure_momentum":
            cap = row.get("max_removal_ratio")
            if cap is None:
                raise ValueError("Hard-filter run lacks max_removal_ratio metadata")
            key = (float(cap), float(row.get("structure_weight")))
            if ratio in arms.setdefault(key, {}):
                raise ValueError("Duplicate arm %r at ratio %s" % (key, ratio))
            arms[key][ratio] = row

    for ratio in (clean_ratio, noisy_ratio):
        if ratio not in baselines:
            raise ValueError("Missing no-filter baseline for noise ratio %s" % ratio)

    baseline_clean_recall = _finite(baselines[clean_ratio], "best_recall_at_20")
    baseline_clean_ndcg = _finite(baselines[clean_ratio], "best_ndcg_at_20")
    baseline_noisy_recall = _finite(baselines[noisy_ratio], "best_recall_at_20")
    baseline_noisy_ndcg = _finite(baselines[noisy_ratio], "best_ndcg_at_20")

    candidates = []
    for (cap, weight), by_ratio in sorted(arms.items()):
        if clean_ratio not in by_ratio or noisy_ratio not in by_ratio:
            raise ValueError(
                "Arm cap=%s weight=%s does not contain both noise ratios" %
                (cap, weight)
            )
        clean = by_ratio[clean_ratio]
        noisy = by_ratio[noisy_ratio]
        clean_recall = _finite(clean, "best_recall_at_20")
        clean_ndcg = _finite(clean, "best_ndcg_at_20")
        noisy_recall = _finite(noisy, "best_recall_at_20")
        noisy_ndcg = _finite(noisy, "best_ndcg_at_20")
        candidate = {
            "max_removal_ratio": cap,
            "structure_weight": weight,
            "clean": {
                "recall_at_20": clean_recall,
                "ndcg_at_20": clean_ndcg,
                "recall_delta_vs_none": clean_recall - baseline_clean_recall,
                "ndcg_delta_vs_none": clean_ndcg - baseline_clean_ndcg,
                "actual_removed_ratio": _finite(clean, "removed_ratio"),
                "filtering_epoch": int(_finite(clean, "filtering_epoch")),
            },
            "noise_0p2": {
                "recall_at_20": noisy_recall,
                "ndcg_at_20": noisy_ndcg,
                "recall_delta_vs_none": noisy_recall - baseline_noisy_recall,
                "ndcg_delta_vs_none": noisy_ndcg - baseline_noisy_ndcg,
                "actual_removed_ratio": _finite(noisy, "removed_ratio"),
                "filtering_epoch": int(_finite(noisy, "filtering_epoch")),
                "noisy_removal_rate": noisy.get("noisy_removal_rate"),
                "clean_removal_rate": noisy.get("clean_removal_rate"),
                "removed_precision_noisy": noisy.get("removed_precision_noisy"),
            },
        }
        candidate["clean_safe"] = (
            candidate["clean"]["recall_delta_vs_none"] >= -clean_tolerance
        )
        candidates.append(candidate)

    safe = [row for row in candidates if row["clean_safe"]]
    pool = safe if safe else candidates
    selected = None
    if pool:
        selected = max(
            pool,
            key=lambda row: (
                row["noise_0p2"]["recall_at_20"],
                row["noise_0p2"]["ndcg_at_20"],
                row["clean"]["recall_at_20"],
                -row["max_removal_ratio"],
            ),
        )

    return {
        "schema_version": "nrgcf_movielens_budget_selection_v1",
        "selection_split": "test",
        "exploratory_single_seed": True,
        "clean_recall_absolute_tolerance": clean_tolerance,
        "required_modulation_lambda": modulation_lambda,
        "selection_rule": (
            "Among clean-safe arms, maximize 0.2-noise Recall@20; tie by "
            "0.2-noise NDCG@20, clean Recall@20, then smaller cap. If no arm "
            "is clean-safe, report the same ranking without claiming safety."
        ),
        "baseline": {
            "clean": {
                "recall_at_20": baseline_clean_recall,
                "ndcg_at_20": baseline_clean_ndcg,
            },
            "noise_0p2": {
                "recall_at_20": baseline_noisy_recall,
                "ndcg_at_20": baseline_noisy_ndcg,
            },
        },
        "candidates": candidates,
        "selected": selected,
        "selected_from_clean_safe_pool": bool(safe),
    }


def markdown(report):
    lines = [
        "# MovieLens removal-budget pilot",
        "",
        "Single-seed exploratory selection on test metrics; confirm the chosen "
        "setting later rather than treating this table as final evidence.",
        "",
        "| Cap | Weight | Clean R@20 | $\\Delta$ clean | Noise .2 R@20 | "
        "$\\Delta$ noisy | Removed clean/noisy | Noise precision | Safe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["candidates"]:
        noisy_precision = row["noise_0p2"]["removed_precision_noisy"]
        precision_text = "--" if noisy_precision is None else "%.4f" % noisy_precision
        lines.append(
            "| {cap:.3f} | {weight:.2f} | {cr:.6f} | {cd:+.6f} | "
            "{nr:.6f} | {nd:+.6f} | {crem:.4f}/{nrem:.4f} | {precision} | "
            "{safe} |".format(
                cap=row["max_removal_ratio"],
                weight=row["structure_weight"],
                cr=row["clean"]["recall_at_20"],
                cd=row["clean"]["recall_delta_vs_none"],
                nr=row["noise_0p2"]["recall_at_20"],
                nd=row["noise_0p2"]["recall_delta_vs_none"],
                crem=row["clean"]["actual_removed_ratio"],
                nrem=row["noise_0p2"]["actual_removed_ratio"],
                precision=precision_text,
                safe="yes" if row["clean_safe"] else "no",
            )
        )
    selected = report.get("selected")
    lines.extend(["", "## Exploratory selection", ""])
    if selected is None:
        lines.append("No complete candidate was available.")
    else:
        lines.append(
            "Selected cap `{:.3f}` with structure weight `{:.2f}`{}".format(
                selected["max_removal_ratio"],
                selected["structure_weight"],
                "." if report["selected_from_clean_safe_pool"]
                else " (no candidate met the clean-safety tolerance).",
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--clean-tolerance", type=float, default=0.002)
    parser.add_argument("--modulation-lambda", type=float, default=None)
    args = parser.parse_args()
    if args.clean_tolerance < 0:
        raise SystemExit("--clean-tolerance must be non-negative")
    result = analyze(
        _load(args.input),
        clean_tolerance=args.clean_tolerance,
        modulation_lambda=args.modulation_lambda,
    )
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print("Wrote MovieLens budget analysis to %s" % args.output)


if __name__ == "__main__":
    main()
