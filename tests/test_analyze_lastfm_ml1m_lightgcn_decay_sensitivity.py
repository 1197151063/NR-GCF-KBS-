import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_lastfm_ml1m_lightgcn_decay_sensitivity as analysis


def run(dataset, objective, decay, recall, temperature=None):
    metadata = {"name": objective}
    if objective == "ssm":
        metadata.update({"tau": temperature, "message_dropout": 0.0})
    return {
        "run": "%s_%s_%s" % (dataset, objective, decay),
        "dataset": dataset,
        "mode": "none",
        "training_objective": objective,
        "training_objective_metadata": metadata,
        "train_learning_rate": 0.001,
        "train_decay": decay,
        "train_batch_size": 2048,
        "train_init_method": (
            "normal" if objective == "bpr" else "xavier_uniform"),
        "seed": 2026,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": "none",
        "best_recall_at_20": recall,
        "best_ndcg_at_20": recall * 0.9,
        "best_epoch": 20,
        "epochs_completed": 40,
        "early_stopped": True,
        "final_training_loss": 1.0,
    }


class CrossDatasetDecaySensitivityTest(unittest.TestCase):
    def rows(self):
        rows = []
        for offset, dataset in enumerate(("lastfm", "ml-1m")):
            tau = 0.5 if dataset == "lastfm" else 0.1
            for objective in ("bpr", "ssm"):
                rows.extend([
                    run(dataset, objective, 1e-5, 0.12 + offset * 0.01, tau),
                    run(dataset, objective, 1e-4, 0.10 + offset * 0.01, tau),
                ])
        return rows

    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, datasets=["lastfm", "ml-1m"],
            decays=[1e-5, 1e-4], learning_rate=0.001,
            ssm_temperatures={"lastfm": 0.5, "ml-1m": 0.1}, seed=2026,
            message_dropout=0.0, batch_size=2048, max_epochs=500,
            patience=20)

    def test_ranks_and_compares_current_decay(self):
        result = self.analyze(self.rows())
        self.assertEqual(result["search_space"]["total_run_count"], 8)
        for dataset in result["datasets"]:
            for objective in ("bpr", "ssm"):
                report = dataset["objectives"][objective]
                self.assertEqual(report["best_observed"]["decay"], 1e-5)
                self.assertGreater(
                    report["best_vs_current_relative_recall_percent"], 0)
        self.assertIn("vs decay=1e-4", analysis.markdown(result))

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
