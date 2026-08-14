import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyze_full_edge_filter_norm import analyze


class AnalyzeFullEdgeFilterNormTest(unittest.TestCase):
    def test_validates_grid_and_computes_full_gains(self):
        runs = []
        values = {
            "lightgcn": (0.10, 0.08, None),
            "norm_only": (0.11, 0.09, None),
            "filter_only": (0.12, 0.10, 0.02),
            "full": (0.13, 0.11, 0.02),
        }
        for arm, (recall, ndcg, removed) in values.items():
            runs.append(
                {
                    "run": f"yelp2018/{arm}/noise_0p2/seed_2026/yelp2018/run",
                    "dataset": "yelp2018",
                    "requested_noise_ratio": 0.2,
                    "seed": 2026,
                    "best_recall_at_20": recall,
                    "best_ndcg_at_20": ndcg,
                    "best_epoch": 7,
                    "removed_ratio": removed,
                    "noisy_removal_rate": 0.04 if removed is not None else None,
                    "clean_removal_rate": 0.01 if removed is not None else None,
                    "removed_precision_noisy": 0.4 if removed is not None else None,
                }
            )

        result = analyze(
            {"root": "/tmp/results", "runs": runs},
            {"schema_version": "profile-v1"},
            ["yelp2018"],
            ["lightgcn", "norm_only", "filter_only", "full"],
            [0.2],
            [2026],
        )

        self.assertEqual(result["run_count"], 4)
        full = next(row for row in result["rows"] if row["arm"] == "full")
        self.assertAlmostEqual(
            full["gains_over"]["lightgcn"]["recall_percent"], 30.0
        )
        self.assertAlmostEqual(
            full["gains_over"]["norm_only"]["recall_percent"],
            100.0 * (0.13 - 0.11) / 0.11,
        )

    def test_rejects_incomplete_grid(self):
        with self.assertRaisesRegex(ValueError, "grid mismatch"):
            analyze(
                {"runs": []},
                {"schema_version": "profile-v1"},
                ["yelp2018"],
                ["lightgcn", "full"],
                [0.0],
                [2026],
            )


if __name__ == "__main__":
    unittest.main()
