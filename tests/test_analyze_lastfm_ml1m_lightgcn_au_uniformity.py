import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_lastfm_ml1m_lightgcn_au_uniformity as analysis


def run(dataset, weight, recall):
    return {
        "run": "%s_%s" % (dataset, weight),
        "dataset": dataset,
        "mode": "none",
        "training_objective": "au",
        "training_objective_metadata": {
            "name": "au", "uniformity_weight": weight,
            "uniformity_t": 2.0, "uniformity_sides": "user_plus_item",
            "regularization": "none", "message_dropout": 0.0,
        },
        "train_learning_rate": 0.0005,
        "train_decay": 0.001,
        "train_batch_size": 2048,
        "train_init_method": "xavier_uniform",
        "seed": 2026,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": "none",
        "best_recall_at_20": recall,
        "best_ndcg_at_20": recall * 0.9,
        "best_epoch": 20,
        "epochs_completed": 40,
        "early_stopped": True,
        "final_training_loss": -1.0,
    }


class CrossDatasetAUUniformityTest(unittest.TestCase):
    def rows(self):
        rows = []
        for offset, dataset in enumerate(("lastfm", "ml-1m")):
            rows.extend([
                run(dataset, 0.1, 0.10 + offset * 0.01),
                run(dataset, 1.0, 0.12 + offset * 0.01),
                run(dataset, 5.0, 0.09 + offset * 0.01),
            ])
        return rows

    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, datasets=["lastfm", "ml-1m"],
            weights=[0.1, 1.0, 5.0], learning_rate=0.0005,
            configured_decay=0.001, uniformity_t=2.0, seed=2026,
            message_dropout=0.0, batch_size=2048, max_epochs=500,
            patience=20)

    def test_selects_best_weight(self):
        result = self.analyze(self.rows())
        self.assertEqual(result["search_space"]["total_run_count"], 6)
        for dataset in result["datasets"]:
            self.assertEqual(
                dataset["best_observed"]["uniformity_weight"], 1.0)
        self.assertIn("AU metadata confirms", analysis.markdown(result))

    def test_rejects_missing_combination(self):
        with self.assertRaisesRegex(ValueError, "Missing experiment identities"):
            self.analyze(self.rows()[:-1])

    def test_rejects_active_modulation(self):
        rows = self.rows()
        rows[0]["representation_modulation_mode"] = "blend_always"
        with self.assertRaisesRegex(ValueError, "Modulation is active"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
