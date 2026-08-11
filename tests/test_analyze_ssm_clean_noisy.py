import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyze_ssm_clean_noisy import analyze, markdown


class SsmCleanNoisyAnalysisTest(unittest.TestCase):
    def test_extracts_matched_pair(self):
        rows = []
        for noise, recall in ((0.0, 0.07), (0.2, 0.06)):
            rows.append({
                "dataset": "yelp2018",
                "mode": "none",
                "training_objective": "ssm",
                "representation_modulation_mode": "original_always",
                "seed": 2020,
                "requested_noise_ratio": noise,
                "actual_noise_ratio": noise,
                "best_epoch": 3,
                "best_recall_at_20": recall,
                "best_ndcg_at_20": recall - 0.01,
                "epochs_completed": 23,
            })
        report = analyze({"runs": rows}, ["0", "0.2"], 2020)
        self.assertEqual(len(report["runs"]), 2)
        self.assertIn("0.060000", markdown(report))


if __name__ == "__main__":
    unittest.main()
