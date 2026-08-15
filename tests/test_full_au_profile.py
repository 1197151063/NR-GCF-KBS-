import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
PROFILE = os.path.join(ROOT, "configs", "full_au_edge_filter_norm.json")


class FullAuProfileTest(unittest.TestCase):
    def setUp(self):
        with open(PROFILE, encoding="utf-8") as stream:
            self.profile = json.load(stream)

    def test_default_grid_has_48_cases(self):
        self.assertEqual(self.profile["objective"], "au")
        self.assertEqual(len(self.profile["datasets"]) * 6 * 2, 48)

    def test_requested_learning_rate_split_and_initialization(self):
        common = self.profile["common"]
        self.assertEqual(common["non_crossnorm_train_lr"], 0.001)
        self.assertEqual(common["train_init_method"], "xavier_uniform")
        for dataset in self.profile["datasets"].values():
            self.assertEqual(dataset["crossnorm_train_lr"], 0.0005)

    def test_selected_au_and_modulation_weights(self):
        datasets = self.profile["datasets"]
        expected = {
            "yelp2018": (1.0, 1.0),
            "amazon-book": (5.0, 0.2),
            "lastfm": (0.1, 0.2),
            "ml-1m": (0.5, 1.0),
        }
        for dataset, (au_weight, modulation) in expected.items():
            self.assertEqual(
                datasets[dataset]["au_uniformity_weight"], au_weight
            )
            self.assertEqual(
                datasets[dataset]["modulation_weight"], modulation
            )


if __name__ == "__main__":
    unittest.main()
