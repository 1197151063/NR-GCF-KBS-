import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_cross_dataset_objective_modulation as analysis


def run(dataset, objective, mode, weight, recall, seed=2026):
    metadata = {"name": objective, "message_dropout": 0.0}
    if objective == "ssm":
        metadata["tau"] = 0.1
    else:
        metadata.update({"uniformity_weight": 1.0, "uniformity_t": 2.0})
    return {
        "run": "%s_%s_%s_%s" % (dataset, objective, mode, weight),
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
        "representation_modulation_mode": mode,
        "representation_modulation_lambda": weight,
        "best_recall_at_20": recall,
        "best_ndcg_at_20": recall * 0.8,
        "best_epoch": 10,
        "epochs_completed": 30,
        "early_stopped": True,
    }


def complete_rows():
    rows = []
    for dataset in ("lastfm", "ml-1m"):
        for objective in ("ssm", "au"):
            rows.append(run(dataset, objective, "none", None, 0.10))
            rows.append(run(dataset, objective, "blend_always", 0.5, 0.11))
            rows.append(run(dataset, objective, "blend_always", 1.0, 0.09))
    return rows


class CrossDatasetObjectiveModulationTest(unittest.TestCase):
    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, datasets=["lastfm", "ml-1m"],
            objectives=["ssm", "au"], weights=[0.5, 1.0], seeds=[2026],
            learning_rate=0.0005, decay=0.0001,
            message_dropout=0.0, batch_size=2048, ssm_tau=0.1,
            au_uniformity_weight=1.0, au_uniformity_t=2.0)

    def test_includes_lightgcn_in_best_selection(self):
        result = self.analyze(complete_rows())
        for dataset in result["datasets"]:
            for objective in dataset["objectives"]:
                self.assertEqual(
                    objective["best_observed"]["modulation_weight"], 0.5)
                self.assertEqual(
                    len(objective["grid_including_lightgcn"]), 3)

    def test_selects_zero_when_every_nonzero_weight_is_worse(self):
        rows = complete_rows()
        for row in rows:
            if row["representation_modulation_mode"] == "blend_always":
                row["best_recall_at_20"] = 0.08
                row["best_ndcg_at_20"] = 0.06
        result = self.analyze(rows)
        for dataset in result["datasets"]:
            for objective in dataset["objectives"]:
                self.assertEqual(
                    objective["best_observed"]["modulation_weight"], 0.0)

    def test_rejects_missing_weight(self):
        rows = complete_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "Missing experiment identities"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
