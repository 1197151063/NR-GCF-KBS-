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
                "best_epoch": 93,
                "best_recall_at_20": 0.12,
                "best_ndcg_at_20": 0.08,
                "final_training_loss": 0.3,
            }), encoding="utf-8")

            report = summarize(root)
            self.assertEqual(report["run_count"], 1)
            self.assertEqual(report["runs"][0]["epochs_completed"], 100)
            self.assertEqual(report["runs"][0]["reliability_auroc"], 0.8)


if __name__ == "__main__":
    unittest.main()
