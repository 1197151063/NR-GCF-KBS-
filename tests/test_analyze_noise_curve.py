import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyze_noise_curve import analyze, markdown


def row(noise):
    return {
        "dataset": "lastfm",
        "mode": "hard_structure_momentum",
        "seed": 2026,
        "requested_noise_ratio": noise,
        "actual_noise_ratio": noise,
        "filtering_epoch": 10,
        "filtering_schedule": "fixed",
        "max_removal_ratio": 0.04,
        "representation_modulation_lambda": 0.2,
        "structure_weight": 0.5,
        "best_epoch": 20,
        "best_recall_at_20": 0.2,
        "best_ndcg_at_20": 0.1,
        "removed_ratio": 0.04,
        "noisy_removal_rate": None if noise == 0 else 0.08,
        "removed_precision_noisy": None if noise == 0 else 0.4,
        "momentum_auroc": None if noise == 0 else 0.6,
        "structure_auroc": None if noise == 0 else 0.75,
    }


class NoiseCurveTest(unittest.TestCase):
    def test_extracts_and_sorts_matched_rows(self):
        report = {"runs": [row(0.2), row(0.0), row(0.1)]}
        result = analyze(
            report, "lastfm", 0.04, 0.2, 0.5, [0, 0.1, 0.2], 2026
        )
        self.assertEqual(
            [entry["requested_noise_ratio"] for entry in result["runs"]],
            [0.0, 0.1, 0.2],
        )
        self.assertIn("fixed-time noise curve", markdown(result))

    def test_rejects_missing_ratio(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            analyze(
                {"runs": [row(0.0)]},
                "lastfm", 0.04, 0.2, 0.5, [0, 0.2], 2026,
            )


if __name__ == "__main__":
    unittest.main()
