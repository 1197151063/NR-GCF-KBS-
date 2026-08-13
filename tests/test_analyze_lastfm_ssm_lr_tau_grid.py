import pathlib
import sys
import unittest


CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import analyze_lastfm_ssm_lr_tau_grid as analysis


def run(learning_rate, temperature, recall, epoch):
    return {
        "run": "lr_%s_tau_%s" % (learning_rate, temperature),
        "dataset": "lastfm",
        "mode": "none",
        "training_objective": "ssm",
        "training_objective_metadata": {
            "name": "ssm", "tau": temperature, "message_dropout": 0.0,
        },
        "train_learning_rate": learning_rate,
        "train_decay": 0.0001,
        "train_batch_size": 2048,
        "train_init_method": "xavier_uniform",
        "seed": 2026,
        "requested_noise_ratio": 0.0,
        "representation_modulation_mode": "none",
        "best_recall_at_20": recall,
        "best_ndcg_at_20": recall * 0.8,
        "best_epoch": epoch,
        "epochs_completed": epoch + 20,
        "early_stopped": True,
        "final_training_loss": 4.0,
    }


class LastFMSSMLrTauGridTest(unittest.TestCase):
    def analyze(self, rows):
        return analysis.analyze(
            {"runs": rows}, learning_rates=[0.0001, 0.001],
            temperatures=[0.1, 0.5], seed=2026, decay=0.0001,
            message_dropout=0.0, batch_size=2048)

    def test_ranks_and_flags_early_peak(self):
        rows = [
            run(0.0001, 0.1, 0.10, 10),
            run(0.0001, 0.5, 0.11, 8),
            run(0.001, 0.1, 0.12, 1),
            run(0.001, 0.5, 0.09, 2),
        ]
        result = self.analyze(rows)
        self.assertEqual(result["best_observed"]["learning_rate"], 0.001)
        self.assertTrue(result["best_observed"]["early_peak_flag"])
        self.assertEqual(result["early_peak_count"], 2)
        self.assertIn("Early-peak configurations: `2/4`", analysis.markdown(result))

    def test_rejects_missing_combination(self):
        rows = [
            run(0.0001, 0.1, 0.10, 10),
            run(0.0001, 0.5, 0.11, 8),
            run(0.001, 0.1, 0.12, 1),
        ]
        with self.assertRaisesRegex(ValueError, "Missing grid identities"):
            self.analyze(rows)

    def test_rejects_active_modulation(self):
        rows = [
            run(0.0001, 0.1, 0.10, 10),
            run(0.0001, 0.5, 0.11, 8),
            run(0.001, 0.1, 0.12, 1),
            run(0.001, 0.5, 0.09, 2),
        ]
        rows[0]["representation_modulation_mode"] = "blend_always"
        with self.assertRaisesRegex(ValueError, "Modulation is active"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
