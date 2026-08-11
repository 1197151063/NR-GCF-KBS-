"""Merge compact NR-GCF reliability run JSONs into one transfer-friendly file."""

import argparse
import json
from pathlib import Path


def _read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _manifest(path):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _number(value, cast=float):
    if value is None or value == "":
        return None
    return cast(value)


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
    training_paths = set(root.glob("**/edge_reliability/training_summary.json"))
    reliability_paths = set(
        root.glob("**/edge_reliability/reliability_summary.json")
    )
    training_paths.update(
        path.with_name("training_summary.json") for path in reliability_paths
    )
    for training_path in sorted(training_paths):
        reliability_path = training_path.with_name("reliability_summary.json")
        reliability = _read(reliability_path) if reliability_path.exists() else {}
        training = _read(training_path) if training_path.exists() else {}
        run_dir = training_path.parent.parent
        manifest = _manifest(run_dir / "run_manifest.txt")
        noise_validation_path = run_dir / "noise_validation.json"
        noise_validation = (
            _read(noise_validation_path) if noise_validation_path.exists()
            else {}
        )
        evaluation = reliability.get("synthetic_label_evaluation")
        representation_modulation = (
            training.get("representation_modulation")
            or reliability.get("representation_modulation")
            or {}
        )
        adaptive_filtering = reliability.get("adaptive_filtering") or {}
        filtering_timing = training.get("filtering_timing") or {}
        early_stopping = training.get("early_stopping") or {}
        training_objective = training.get("training_objective") or {}
        parameters = reliability.get("parameters") or {}
        adaptive_trace = adaptive_filtering.get("trace") or []
        trigger_snapshot = adaptive_trace[-1] if adaptive_trace else {}
        runs.append({
            "run": str(run_dir.relative_to(root)),
            "dataset": reliability.get("dataset") or manifest.get("dataset"),
            "mode": reliability.get("mode") or manifest.get("edge_filter_mode"),
            "training_objective": training_objective.get("name", "bpr"),
            "training_objective_metadata": training_objective,
            "seed": reliability.get("seed") or _number(manifest.get("seed"), int),
            "requested_noise_ratio": (
                reliability.get("requested_noise_ratio")
                if reliability.get("requested_noise_ratio") is not None
                else _number(manifest.get("requested_noise_ratio"))
            ),
            "filtering_epoch": reliability.get("filtering_epoch"),
            "filtering_schedule": (
                adaptive_filtering.get("schedule")
                or filtering_timing.get("schedule")
            ),
            "filtering_trigger_reason": adaptive_filtering.get("trigger_reason"),
            "filtering_trigger_coverage": trigger_snapshot.get("coverage"),
            "filtering_trigger_jaccard": trigger_snapshot.get(
                "removed_set_jaccard"
            ),
            "filtering_trigger_stable_checks": trigger_snapshot.get(
                "consecutive_stable_checks"
            ),
            "momentum_semantics": reliability.get("momentum_semantics"),
            "momentum_quantile": parameters.get("momentum_quantile"),
            "structure_quantile": parameters.get("structure_quantile"),
            "structure_weight": parameters.get("structure_weight"),
            "max_removal_ratio": parameters.get("max_removal_ratio"),
            "representation_modulation_mode": representation_modulation.get("mode"),
            "representation_modulation_ramp_epochs": representation_modulation.get("ramp_epochs"),
            "representation_modulation_lambda": representation_modulation.get("lambda"),
            "adaptive_budget_count": reliability.get(
                "adaptive_budget_count_without_connectivity_constraint"
            ),
            "capped_adaptive_budget_count": reliability.get(
                "capped_adaptive_budget_count"
            ),
            "actual_noise_ratio": (
                (reliability.get("noise_validation") or {}).get(
                    "actual_noise_ratio"
                )
                if reliability else noise_validation.get("actual_noise_ratio")
            ),
            "epochs_completed": training.get("epochs_completed"),
            "completed_requested_epochs": training.get("completed_requested_epochs"),
            "early_stopped": early_stopping.get("stopped_early"),
            "early_stopping_wait": early_stopping.get(
                "consecutive_non_improving_epochs"
            ),
            "best_epoch": training.get("best_epoch"),
            "best_recall_at_20": training.get("best_recall_at_20"),
            "best_ndcg_at_20": training.get("best_ndcg_at_20"),
            "best_post_filter_monitor": training.get(
                "best_post_filter_monitor"
            ),
            "best_post_filter_includes_filtering_epoch": training.get(
                "best_post_filter_includes_filtering_epoch"
            ),
            "best_post_filter_epoch": training.get("best_post_filter_epoch"),
            "best_post_filter_recall_at_20": training.get("best_post_filter_recall_at_20"),
            "best_post_filter_ndcg_at_20": training.get("best_post_filter_ndcg_at_20"),
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
        "schema_version": "nrgcf_reliability_comparison_v9",
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
