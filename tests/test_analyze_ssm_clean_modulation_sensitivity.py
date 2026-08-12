import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_ssm_clean_modulation_sensitivity as analysis


def run(weight, seed, recall, ndcg):
    return {
        "run": "mu_%s_seed_%s" % (weight, seed),
        "dataset": "yelp2018",
        "mode": "none",
        "training_objective": "ssm",
        "training_objective_metadata": {
            "tau": 0.14,
            "message_dropout": 0.1,
        },
        "train_learning_rate": 0.0001,
        "train_decay": 0.0001,
        "train_batch_size": 2048,
        "train_init_method": "xavier_uniform",
        "seed": seed,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": "blend_always",
        "representation_modulation_lambda": weight,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": ndcg,
        "best_epoch": 8,
        "epochs_completed": 28,
        "early_stopped": True,
    }


class CleanSsmModulationAnalysisTest(unittest.TestCase):
    def test_aggregates_and_selects_best_weight(self):
        report = {"runs": [
            run(0.0, 2020, 0.060, 0.050),
            run(0.0, 2021, 0.062, 0.051),
            run(0.5, 2020, 0.070, 0.058),
            run(0.5, 2021, 0.072, 0.059),
            run(1.0, 2020, 0.069, 0.057),
            run(1.0, 2021, 0.068, 0.056),
        ]}
        result = analysis.analyze(
            report, dataset="yelp2018",
            expected_weights=[0.0, 0.5, 1.0],
            expected_seeds=[2020, 2021], learning_rate=1e-4,
            temperature=0.14, decay=1e-4, message_dropout=0.1,
            batch_size=2048)
        self.assertEqual(result["best_observed"]["modulation_weight"], 0.5)
        self.assertAlmostEqual(result["grid"][1]["recall_at_20"]["mean"], 0.071)
        self.assertGreater(result["grid"][1]["recall_at_20"]["sample_std"], 0)

    def test_rejects_incomplete_grid(self):
        with self.assertRaisesRegex(ValueError, "Expected modulation weights"):
            analysis.analyze(
                {"runs": [run(0.0, 2020, 0.06, 0.05)]},
                dataset="yelp2018", expected_weights=[0.0, 1.0],
                expected_seeds=[2020], learning_rate=1e-4,
                temperature=0.14, decay=1e-4, message_dropout=0.1,
                batch_size=2048)


if __name__ == "__main__":
    unittest.main()
