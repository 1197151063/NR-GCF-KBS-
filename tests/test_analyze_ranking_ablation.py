import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import analyze_ranking_ablation


def row(ratio, weight):
    return {
        "dataset": "lastfm",
        "mode": "hard_structure_momentum",
        "max_removal_ratio": 0.02,
        "representation_modulation_lambda": 0.4,
        "structure_weight": weight,
        "requested_noise_ratio": ratio,
        "seed": 2026,
        "filtering_epoch": 7,
        "best_recall_at_20": 0.2,
        "best_ndcg_at_20": 0.1,
        "removed_ratio": 0.02,
        "noisy_removal_rate": 0.04 if ratio else 0.0,
        "removed_precision_noisy": 0.4 if ratio else 0.0,
    }


class RankingAblationAnalysisTest(unittest.TestCase):
    def test_extracts_requested_cap_modulation_and_weights(self):
        report = {"runs": [
            row(0.0, 0.0), row(0.2, 0.0),
            row(0.0, 0.95), row(0.2, 0.95),
            dict(row(0.0, 0.95), max_removal_ratio=0.03),
        ]}
        result = analyze_ranking_ablation.analyze(
            report,
            dataset="lastfm",
            removal_cap=0.02,
            modulation_lambda=0.4,
            weights=[0.0, 0.95],
        )
        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(result["selected_max_removal_ratio"], 0.02)


if __name__ == "__main__":
    unittest.main()
