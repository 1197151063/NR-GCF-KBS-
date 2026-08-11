import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyze_ssm_yelp_hyperparameter_grid import (
    analyze,
    collect,
    robust_ranking,
    write_selected,
)


def make_run(root, stage, name, ratio, lr, tau, decay, recall, ndcg):
    run_dir = Path(root) / stage / name / "yelp2018" / "run"
    summary_dir = run_dir / "edge_reliability"
    summary_dir.mkdir(parents=True)
    (run_dir / "run_manifest.txt").write_text(
        "dataset=yelp2018\n"
        "training_objective=ssm\n"
        "seed=2020\n"
        "requested_noise_ratio=%s\n"
        "train_lr=%s\n"
        "train_decay=%s\n"
        "ssm_tau=%s\n" % (ratio, lr, decay, tau),
        encoding="utf-8",
    )
    (summary_dir / "training_summary.json").write_text(
        json.dumps({
            "training_objective": {"name": "ssm"},
            "representation_modulation": {"mode": "original_always"},
            "best_epoch": 3,
            "epochs_completed": 23,
            "best_recall_at_20": recall,
            "best_ndcg_at_20": ndcg,
            "final_training_loss": 4.0,
        }),
        encoding="utf-8",
    )


class SsmYelpHyperparameterGridTest(unittest.TestCase):
    def test_collect_rank_select_and_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_run(
                temporary, "clean_grid", "a", 0, 0.0001, 0.09,
                0.00001, 0.070, 0.060,
            )
            make_run(
                temporary, "clean_grid", "b", 0, 0.0005, 0.10,
                0.0001, 0.071, 0.059,
            )
            make_run(
                temporary, "noisy_validation", "a", 0.2, 0.0001, 0.09,
                0.00001, 0.060, 0.050,
            )
            make_run(
                temporary, "noisy_validation", "b", 0.2, 0.0005, 0.10,
                0.0001, 0.055, 0.046,
            )
            clean = collect(Path(temporary) / "clean_grid")
            noisy = collect(Path(temporary) / "noisy_validation")
            self.assertEqual(clean[0]["learning_rate"], 0.0005)
            robust = robust_ranking(clean, noisy)
            self.assertEqual(robust[0]["learning_rate"], 0.0001)
            selected_path = Path(temporary) / "selected.tsv"
            write_selected(selected_path, clean, 1)
            self.assertEqual(selected_path.read_text().split()[0], "0.0005")

            report = analyze(
                temporary,
                [0.0001, 0.0005],
                [0.09, 0.10],
                [0.00001, 0.0001],
                2020,
                2,
            )
            self.assertEqual(report["clean_completed_count"], 2)
            self.assertEqual(report["noisy_validation_completed_count"], 2)
            self.assertEqual(
                report["recommended_configuration"]["learning_rate"],
                0.0001,
            )


if __name__ == "__main__":
    unittest.main()
