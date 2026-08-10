import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyze_objective_backbones import analyze, markdown


def row(objective, mode, recall, ndcg):
    return {
        "dataset": "yelp2018",
        "mode": "none",
        "seed": 2026,
        "requested_noise_ratio": 0.0,
        "training_objective": objective,
        "representation_modulation_mode": mode,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": ndcg,
        "best_epoch": 10,
    }


class ObjectiveBackboneAnalysisTest(unittest.TestCase):
    def test_computes_matched_relative_gains(self):
        report = {"runs": [
            row("ssm", "none", 0.10, 0.08),
            row("ssm", "original_always", 0.11, 0.09),
            row("au", "none", 0.20, 0.16),
            row("au", "original_always", 0.21, 0.17),
        ]}
        result = analyze(
            report, "yelp2018", ["ssm", "au"], "none",
            "original_always", 0.0, 2026,
        )
        self.assertEqual(len(result["comparisons"]), 2)
        self.assertAlmostEqual(
            result["comparisons"][0]["recall_relative_percent"], 10.0
        )
        self.assertIn("CrossNorm R@20", markdown(result))

    def test_rejects_missing_treatment(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            analyze(
                {"runs": [row("ssm", "none", 0.1, 0.08)]},
                "yelp2018", ["ssm"], "none", "original_always", 0.0, 2026,
            )


if __name__ == "__main__":
    unittest.main()
