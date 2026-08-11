import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from summarize_reliability_runs import summarize


class SummarizeReliabilityRunsTest(unittest.TestCase):
    def test_summarizes_training_only_run_when_filtering_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "case" / "yelp2018" / "run"
            output = run / "edge_reliability"
            output.mkdir(parents=True)
            (run / "run_manifest.txt").write_text(
                "dataset=yelp2018\n"
                "seed=2020\n"
                "requested_noise_ratio=0.2\n"
                "edge_filter_mode=none\n",
                encoding="utf-8",
            )
            (run / "noise_validation.json").write_text(
                json.dumps({"actual_noise_ratio": 0.2}), encoding="utf-8"
            )
            (output / "training_summary.json").write_text(json.dumps({
                "epochs_completed": 12,
                "best_epoch": 3,
                "best_recall_at_20": 0.1,
                "best_ndcg_at_20": 0.08,
                "training_objective": {"name": "ssm"},
                "representation_modulation": {"mode": "original_always"},
            }), encoding="utf-8")
            report = summarize(root)
            self.assertEqual(report["run_count"], 1)
            row = report["runs"][0]
            self.assertEqual(row["dataset"], "yelp2018")
            self.assertEqual(row["mode"], "none")
            self.assertEqual(row["training_objective"], "ssm")
            self.assertEqual(row["requested_noise_ratio"], 0.2)
            self.assertEqual(row["actual_noise_ratio"], 0.2)

    def test_merges_only_compact_json_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "yelp2018" / "soft_run" / "edge_reliability"
            output.mkdir(parents=True)
            (output / "reliability_summary.json").write_text(json.dumps({
                "dataset": "yelp2018",
                "mode": "soft_reliability",
                "seed": 2026,
                "requested_noise_ratio": 0.2,
                "parameters": {
                    "momentum_quantile": 0.8,
                    "structure_quantile": 0.2,
                    "structure_weight": 0.95,
                },
                "filtering_epoch": 7,
                "adaptive_filtering": {
                    "schedule": "adaptive",
                    "trigger_reason": "coverage_and_removed_set_stable",
                    "trace": [{
                        "epoch": 7,
                        "coverage": 1.0,
                        "removed_set_jaccard": 0.95,
                        "consecutive_stable_checks": 2,
                    }],
                },
                "retained_edge_count": 10,
                "removed_edge_count": 0,
                "removed_ratio": 0.0,
                "protected_edge_count": 2,
                "statistics": {"propagation_weight": {"mean": 0.7}},
                "noise_validation": {"actual_noise_ratio": 0.2},
                "synthetic_label_evaluation": {
                    "clean_removal_rate": 0.0,
                    "noisy_removal_rate": 0.0,
                    "removed_precision_noisy": 0.0,
                    "scores_for_noisy_edge": {
                        "reliability": {"auroc": 0.8, "average_precision": 0.5}
                    },
                },
            }), encoding="utf-8")
            (output / "training_summary.json").write_text(json.dumps({
                "epochs_completed": 100,
                "completed_requested_epochs": True,
                "early_stopping": {
                    "stopped_early": False,
                    "consecutive_non_improving_epochs": 3,
                },
                "best_epoch": 93,
                "best_recall_at_20": 0.12,
                "best_ndcg_at_20": 0.08,
                "best_post_filter_monitor": "Recall@20",
                "best_post_filter_includes_filtering_epoch": True,
                "final_training_loss": 0.3,
            }), encoding="utf-8")

            report = summarize(root)
            self.assertEqual(report["run_count"], 1)
            self.assertEqual(report["runs"][0]["epochs_completed"], 100)
            self.assertEqual(report["runs"][0]["reliability_auroc"], 0.8)
            self.assertEqual(report["runs"][0]["filtering_schedule"], "adaptive")
            self.assertEqual(report["runs"][0]["filtering_trigger_coverage"], 1.0)
            self.assertEqual(report["runs"][0]["structure_weight"], 0.95)
            self.assertEqual(
                report["runs"][0]["best_post_filter_monitor"], "Recall@20"
            )
            self.assertTrue(
                report["runs"][0]["best_post_filter_includes_filtering_epoch"]
            )


if __name__ == "__main__":
    unittest.main()
