import os
import sys
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import analyze_common_modulation


def run(weight, ratio, recall, ndcg):
    return {
        "dataset": "lastfm",
        "mode": "none",
        "representation_modulation_lambda": weight,
        "requested_noise_ratio": ratio,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": ndcg,
        "best_epoch": 5,
        "epochs_completed": 25,
        "early_stopped": True,
    }


class CommonModulationAnalysisTest(unittest.TestCase):
    def test_selects_one_weight_across_clean_and_noisy_cases(self):
        report = {"runs": [
            run(0.2, 0.0, 0.30, 0.29),
            run(0.2, 0.2, 0.24, 0.22),
            run(1.0, 0.0, 0.31, 0.30),
            run(1.0, 0.2, 0.20, 0.18),
        ]}
        result = analyze_common_modulation.analyze(report, dataset="lastfm")
        self.assertEqual(result["selected"]["modulation_lambda"], 0.2)
        self.assertEqual(len(result["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
