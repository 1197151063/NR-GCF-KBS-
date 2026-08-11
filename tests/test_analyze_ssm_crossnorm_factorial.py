import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyze_ssm_crossnorm_factorial import collect, markdown


class SsmCrossnormFactorialAnalysisTest(unittest.TestCase):
    def test_collects_manifest_factors_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "case" / "dataset" / "run"
            summary_dir = run_dir / "edge_reliability"
            summary_dir.mkdir(parents=True)
            (run_dir / "run_manifest.txt").write_text(
                "train_init_method=normal\n"
                "train_init_weight=0.01\n"
                "train_decay=0.0001\n"
                "objective_message_dropout=0.1\n",
                encoding="utf-8",
            )
            summary = {
                "training_objective": {"name": "ssm"},
                "representation_modulation": {"mode": "original_always"},
                "best_epoch": 7,
                "best_recall_at_20": 0.07,
                "best_ndcg_at_20": 0.06,
                "final_training_loss": 4.2,
                "epochs_completed": 10,
            }
            (summary_dir / "training_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            rows = collect(Path(temporary))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["embedding_init"], "normal")
            self.assertEqual(rows[0]["message_dropout"], 0.1)
            self.assertIn("0.070000", markdown(rows))


if __name__ == "__main__":
    unittest.main()
