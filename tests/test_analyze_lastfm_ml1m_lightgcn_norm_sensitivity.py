import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_lastfm_ml1m_lightgcn_norm_sensitivity as analysis


def run(dataset, objective, weight, recall, temperature=None):
    metadata = {"name": objective}
    if objective == "ssm":
        metadata.update({"tau": temperature, "message_dropout": 0.0})
    return {
        "run": "%s_%s_%s" % (dataset, objective, weight),
        "dataset": dataset,
        "mode": "none",
        "training_objective": objective,
        "training_objective_metadata": metadata,
        "train_learning_rate": 0.0005,
        "train_decay": 0.001,
        "train_batch_size": 2048,
        "train_init_method": (
            "normal" if objective == "bpr" else "xavier_uniform"),
        "seed": 2026,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": (
            "none" if weight == 0.0 else "blend_always"),
        "representation_modulation_lambda": weight,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": recall * 0.9,
        "best_epoch": 20,
        "epochs_completed": 40,
        "early_stopped": True,
        "final_training_loss": 1.0,
    }


class CrossDatasetNormSensitivityTest(unittest.TestCase):
    def rows(self):
        rows = []
        for offset, dataset in enumerate(("lastfm", "ml-1m")):
            tau = 0.5 if dataset == "lastfm" else 0.1
            for objective in ("bpr", "ssm"):
                rows.extend([
                    run(dataset, objective, 0.0, 0.10 + offset * 0.01, tau),
                    run(dataset, objective, 0.5, 0.12 + offset * 0.01, tau),
                    run(dataset, objective, 1.0, 0.09 + offset * 0.01, tau),
                ])
        return rows

    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, datasets=["lastfm", "ml-1m"],
            weights=[0.0, 0.5, 1.0], learning_rate=0.0005, decay=0.001,
            ssm_temperatures={"lastfm": 0.5, "ml-1m": 0.1}, seed=2026,
            message_dropout=0.0, batch_size=2048, max_epochs=500,
            patience=20)

    def test_selects_helpful_norm(self):
        result = self.analyze(self.rows())
        self.assertEqual(result["search_space"]["total_run_count"], 12)
        for dataset in result["datasets"]:
            for objective in dataset["objectives"]:
                self.assertEqual(
                    objective["best_observed"]["modulation_weight"], 0.5)
                self.assertTrue(objective["norm_is_helpful"])
        self.assertIn("`Mu=0` is ordinary LightGCN", analysis.markdown(result))

    def test_rejects_missing_combination(self):
        with self.assertRaisesRegex(ValueError, "Missing experiment identities"):
            self.analyze(self.rows()[:-1])

    def test_rejects_wrong_temperature(self):
        rows = self.rows()
        next(row for row in rows if row["training_objective"] == "ssm")[
            "training_objective_metadata"]["tau"] = 0.2
        with self.assertRaisesRegex(ValueError, "Unexpected SSM temperature"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
