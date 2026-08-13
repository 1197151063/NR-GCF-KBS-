import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_lastfm_ml1m_lightgcn_convergence as analysis


def run(dataset, objective, recall, epoch, temperature=None,
        epochs_completed=None, early_stopped=True):
    metadata = {"name": objective}
    if objective == "ssm":
        metadata.update({"tau": temperature, "message_dropout": 0.0})
    return {
        "run": "%s_%s_%s" % (dataset, objective, temperature),
        "dataset": dataset,
        "mode": "none",
        "training_objective": objective,
        "training_objective_metadata": metadata,
        "train_learning_rate": 0.001,
        "train_decay": 0.0001,
        "train_batch_size": 2048,
        "train_init_method": (
            "normal" if objective == "bpr" else "xavier_uniform"),
        "seed": 2026,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": "none",
        "best_recall_at_20": recall,
        "best_ndcg_at_20": recall * 0.9,
        "best_epoch": epoch,
        "epochs_completed": epochs_completed or epoch + 20,
        "early_stopped": early_stopped,
        "final_training_loss": 1.0,
    }


class CrossDatasetLightGCNConvergenceTest(unittest.TestCase):
    def rows(self):
        rows = []
        for offset, dataset in enumerate(("lastfm", "ml-1m")):
            rows.extend([
                run(dataset, "bpr", 0.10 + offset * 0.01, 100),
                run(dataset, "ssm", 0.13 + offset * 0.01, 90, 0.1),
                run(dataset, "ssm", 0.14 + offset * 0.01, 480, 0.2, 500, False),
            ])
        return rows

    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, datasets=["lastfm", "ml-1m"],
            learning_rate=0.001, temperatures=[0.1, 0.2], seed=2026,
            decay=0.0001, message_dropout=0.0, batch_size=2048,
            max_epochs=500, patience=20)

    def test_ranks_each_dataset_and_marks_cap(self):
        result = self.analyze(self.rows())
        self.assertEqual(result["search_space"]["total_run_count"], 6)
        for dataset in result["datasets"]:
            self.assertEqual(
                dataset["objectives"]["ssm"]["best_observed"]["temperature"],
                0.2)
            self.assertEqual(dataset["objectives"]["ssm"]["epoch_cap_count"], 1)
        self.assertIn("Fixed `lr=0.001`", analysis.markdown(result))

    def test_rejects_missing_combination(self):
        with self.assertRaisesRegex(ValueError, "Missing experiment identities"):
            self.analyze(self.rows()[:-1])

    def test_rejects_wrong_learning_rate(self):
        rows = self.rows()
        rows[0]["train_learning_rate"] = 0.0005
        with self.assertRaisesRegex(ValueError, "Unexpected learning rate"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
