import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_cross_dataset_lightgcn_objectives as analysis


def run(dataset, objective, seed, recall, ndcg):
    metadata = {
        "name": objective,
        "message_dropout": 0.0,
    }
    if objective == "ssm":
        metadata["tau"] = 0.1
    else:
        metadata["uniformity_weight"] = 1.0
        metadata["uniformity_t"] = 2.0
    return {
        "run": "%s_%s_%s" % (dataset, objective, seed),
        "dataset": dataset,
        "mode": "none",
        "training_objective": objective,
        "training_objective_metadata": metadata,
        "train_learning_rate": 0.0005,
        "train_decay": 0.0001,
        "train_batch_size": 2048,
        "train_init_method": "xavier_uniform",
        "seed": seed,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": "none",
        "representation_modulation_lambda": None,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": ndcg,
        "best_epoch": 12,
        "epochs_completed": 32,
        "early_stopped": True,
    }


class CrossDatasetLightGCNObjectivesTest(unittest.TestCase):
    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, datasets=["lastfm", "ml-1m"],
            objectives=["ssm", "au"], seeds=[2026],
            learning_rate=0.0005, decay=0.0001,
            message_dropout=0.0, batch_size=2048, ssm_tau=0.1,
            au_uniformity_weight=1.0, au_uniformity_t=2.0)

    def test_validates_and_summarizes_four_runs(self):
        rows = []
        for dataset in ("lastfm", "ml-1m"):
            rows.append(run(dataset, "ssm", 2026, 0.10, 0.08))
            rows.append(run(dataset, "au", 2026, 0.09, 0.07))
        result = self.analyze(rows)
        self.assertEqual(len(result["datasets"]), 2)
        self.assertEqual(
            result["datasets"][0]["objectives"][0]["recall_at_20"]["mean"],
            0.10)
        self.assertIn("| lastfm | SSM |", analysis.markdown(result))

    def test_rejects_missing_objective(self):
        rows = [
            run("lastfm", "ssm", 2026, 0.10, 0.08),
            run("lastfm", "au", 2026, 0.09, 0.07),
            run("ml-1m", "ssm", 2026, 0.10, 0.08),
        ]
        with self.assertRaisesRegex(ValueError, "Missing experiment identities"):
            self.analyze(rows)

    def test_rejects_active_modulation(self):
        rows = []
        for dataset in ("lastfm", "ml-1m"):
            rows.append(run(dataset, "ssm", 2026, 0.10, 0.08))
            rows.append(run(dataset, "au", 2026, 0.09, 0.07))
        rows[0]["representation_modulation_mode"] = "blend_always"
        with self.assertRaisesRegex(ValueError, "modulation is active"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
