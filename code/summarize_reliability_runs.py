"""Merge compact NR-GCF reliability run JSONs into one transfer-friendly file."""

import argparse
import json
from pathlib import Path


def _read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _metric(evaluation, name, field):
    if not evaluation:
        return None
    scores = evaluation.get("scores_for_noisy_edge", {})
    metric = scores.get(name)
    if metric is None and name == "momentum_signal":
        metric = scores.get("raw_runtime_momentum")
    return (metric or {}).get(field)


def summarize(root):
    root = Path(root).resolve()
    runs = []
    for reliability_path in sorted(root.glob("**/edge_reliability/reliability_summary.json")):
        reliability = _read(reliability_path)
        training_path = reliability_path.with_name("training_summary.json")
        training = _read(training_path) if training_path.exists() else {}
        evaluation = reliability.get("synthetic_label_evaluation")
        representation_modulation = (
            training.get("representation_modulation")
            or reliability.get("representation_modulation")
            or {}
        )
        runs.append({
            "run": str(reliability_path.parent.parent.relative_to(root)),
            "dataset": reliability.get("dataset"),
            "mode": reliability.get("mode"),
            "seed": reliability.get("seed"),
            "requested_noise_ratio": reliability.get("requested_noise_ratio"),
            "filtering_epoch": reliability.get("filtering_epoch"),
            "momentum_semantics": reliability.get("momentum_semantics"),
            "representation_modulation_mode": representation_modulation.get("mode"),
            "representation_modulation_ramp_epochs": representation_modulation.get("ramp_epochs"),
            "representation_modulation_lambda": representation_modulation.get("lambda"),
            "adaptive_budget_count": reliability.get(
                "adaptive_budget_count_without_connectivity_constraint"
            ),
            "actual_noise_ratio": (
                (reliability.get("noise_validation") or {}).get("actual_noise_ratio")
            ),
            "epochs_completed": training.get("epochs_completed"),
            "completed_requested_epochs": training.get("completed_requested_epochs"),
            "best_epoch": training.get("best_epoch"),
            "best_recall_at_20": training.get("best_recall_at_20"),
            "best_ndcg_at_20": training.get("best_ndcg_at_20"),
            "final_training_loss": training.get("final_training_loss"),
            "retained_edge_count": reliability.get("retained_edge_count"),
            "removed_edge_count": reliability.get("removed_edge_count"),
            "removed_ratio": reliability.get("removed_ratio"),
            "protected_edge_count": reliability.get("protected_edge_count"),
            "clean_removal_rate": evaluation.get("clean_removal_rate") if evaluation else None,
            "noisy_removal_rate": evaluation.get("noisy_removal_rate") if evaluation else None,
            "removed_precision_noisy": evaluation.get("removed_precision_noisy") if evaluation else None,
            "propagation_weight_mean": (
                reliability.get("statistics", {}).get("propagation_weight", {}).get("mean")
            ),
            "momentum_auroc": _metric(evaluation, "momentum_signal", "auroc"),
            "momentum_auprc": _metric(evaluation, "momentum_signal", "average_precision"),
            "structure_auroc": _metric(evaluation, "available_side_structure", "auroc"),
            "structure_auprc": _metric(evaluation, "available_side_structure", "average_precision"),
            "reliability_auroc": _metric(evaluation, "reliability", "auroc"),
            "reliability_auprc": _metric(evaluation, "reliability", "average_precision"),
            "fused_risk_auroc": _metric(evaluation, "fused_risk", "auroc"),
            "fused_risk_auprc": _metric(evaluation, "fused_risk", "average_precision"),
            "gated_soft_risk_auroc": _metric(evaluation, "gated_soft_risk", "auroc"),
            "gated_soft_risk_auprc": _metric(evaluation, "gated_soft_risk", "average_precision"),
        })
    return {
        "schema_version": "nrgcf_reliability_comparison_v4",
        "root": str(root),
        "run_count": len(runs),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Wrote %d runs to %s" % (report["run_count"], output))


if __name__ == "__main__":
    main()
