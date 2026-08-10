import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import analyze_movielens_removal_budget


def row(mode, ratio, recall, ndcg, cap=None, removed=0.0):
    return {
        "run": "%s-%s-%s" % (mode, ratio, cap),
        "dataset": "ml-1m",
        "mode": mode,
        "requested_noise_ratio": ratio,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": ndcg,
        "max_removal_ratio": cap,
        "structure_weight": 0.95,
        "representation_modulation_lambda": 0.2,
        "removed_ratio": removed,
        "filtering_epoch": 7 if mode != "none" else 15,
        "noisy_removal_rate": 0.10 if ratio else 0.0,
        "clean_removal_rate": 0.02,
        "removed_precision_noisy": 0.50 if ratio else 0.0,
    }


class MovieLensBudgetAnalysisTest(unittest.TestCase):
    def test_selects_noisy_gain_subject_to_clean_safety(self):
        report = {"runs": [
            row("none", 0.0, 0.250, 0.240),
            row("none", 0.2, 0.190, 0.170),
            row("hard_structure_momentum", 0.0, 0.249, 0.239, 0.01, 0.01),
            row("hard_structure_momentum", 0.2, 0.194, 0.174, 0.01, 0.01),
            row("hard_structure_momentum", 0.0, 0.247, 0.238, 0.02, 0.02),
            row("hard_structure_momentum", 0.2, 0.196, 0.175, 0.02, 0.02),
        ]}
        result = analyze_movielens_removal_budget.analyze(
            report, clean_tolerance=0.002
        )
        self.assertTrue(result["selected_from_clean_safe_pool"])
        self.assertEqual(result["selected"]["max_removal_ratio"], 0.01)
        self.assertGreater(
            result["selected"]["noise_0p2"]["recall_delta_vs_none"], 0
        )

    def test_can_select_one_matched_modulation_baseline(self):
        report = {"runs": [
            row("none", 0.0, 0.200, 0.190),
            row("none", 0.2, 0.150, 0.140),
            dict(
                row("none", 0.0, 0.250, 0.240),
                representation_modulation_lambda=1.0,
            ),
            dict(
                row("none", 0.2, 0.190, 0.170),
                representation_modulation_lambda=1.0,
            ),
            row("hard_structure_momentum", 0.0, 0.202, 0.192, 0.006, 0.006),
            row("hard_structure_momentum", 0.2, 0.154, 0.144, 0.006, 0.006),
        ]}
        result = analyze_movielens_removal_budget.analyze(
            report, modulation_lambda=0.2
        )
        self.assertEqual(result["baseline"]["clean"]["recall_at_20"], 0.2)
        self.assertEqual(result["selected"]["max_removal_ratio"], 0.006)


if __name__ == "__main__":
    unittest.main()
