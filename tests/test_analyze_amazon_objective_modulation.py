import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_amazon_objective_modulation as analysis


def run(objective, mode, weight, seed, recall, ndcg):
    metadata = {
        "name": objective,
        "message_dropout": 0.0,
    }
    if objective == "ssm":
        metadata["tau"] = 0.1
    else:
        metadata["uniformity_weight"] = 5.0
        metadata["uniformity_t"] = 2.0
    return {
        "run": "%s_%s_%s_%s" % (objective, mode, weight, seed),
        "dataset": "amazon-book",
        "mode": "none",
        "training_objective": objective,
        "training_objective_metadata": metadata,
        "train_learning_rate": 0.0005,
        "train_decay": 0.0001,
        "train_batch_size": 2048,
        "train_init_method": "xavier_uniform",
        "seed": seed,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": mode,
        "representation_modulation_lambda": weight,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": ndcg,
        "best_epoch": 10,
    }


class AmazonObjectiveModulationTest(unittest.TestCase):
    def test_compares_lightgcn_and_selects_weight_per_objective(self):
        rows = []
        for objective in ("ssm", "au"):
            rows.extend([
                run(objective, "none", None, 2020, 0.10, 0.08),
                run(objective, "blend_always", 0.5, 2020, 0.11, 0.09),
                run(objective, "blend_always", 1.0, 2020, 0.105, 0.085),
            ])
        result = analysis.analyze(
            {"runs": rows}, dataset="amazon-book",
            objectives=["ssm", "au"], weights=[0.5, 1.0], seeds=[2020],
            learning_rate=0.0005, decay=0.0001, message_dropout=0.0,
            batch_size=2048, ssm_tau=0.1, au_uniformity_weight=5.0,
            au_uniformity_t=2.0)
        self.assertEqual(len(result["objectives"]), 2)
        for objective in result["objectives"]:
            self.assertEqual(
                objective["best_observed_modulation"]["modulation_weight"],
                0.5)
            self.assertAlmostEqual(
                objective["modulation_grid"][0][
                    "recall_gain_over_lightgcn_percent"], 10.0)

    def test_rejects_missing_weight(self):
        rows = [
            run("ssm", "none", None, 2020, 0.10, 0.08),
            run("au", "none", None, 2020, 0.10, 0.08),
        ]
        with self.assertRaisesRegex(ValueError, "Missing modulation run"):
            analysis.analyze(
                {"runs": rows}, dataset="amazon-book",
                objectives=["ssm", "au"], weights=[0.5], seeds=[2020],
                learning_rate=0.0005, decay=0.0001, message_dropout=0.0,
                batch_size=2048, ssm_tau=0.1,
                au_uniformity_weight=5.0, au_uniformity_t=2.0)


if __name__ == "__main__":
    unittest.main()
