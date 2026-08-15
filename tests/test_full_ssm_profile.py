import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
PROFILE = os.path.join(ROOT, "configs", "full_ssm_edge_filter_norm.json")


class FullSsmProfileTest(unittest.TestCase):
    def setUp(self):
        with open(PROFILE, encoding="utf-8") as stream:
            self.profile = json.load(stream)

    def test_default_grid_has_48_cases(self):
        self.assertEqual(self.profile["objective"], "ssm")
        self.assertEqual(len(self.profile["datasets"]) * 6 * 2, 48)

    def test_requested_learning_rate_split_and_ssm_initialization(self):
        common = self.profile["common"]
        self.assertEqual(common["non_crossnorm_train_lr"], 0.001)
        self.assertEqual(common["train_init_method"], "xavier_uniform")
        for dataset in self.profile["datasets"].values():
            self.assertEqual(dataset["crossnorm_train_lr"], 0.0005)

    def test_selected_temperatures(self):
        datasets = self.profile["datasets"]
        self.assertEqual(datasets["yelp2018"]["ssm_tau"], 0.14)
        self.assertEqual(datasets["amazon-book"]["ssm_tau"], 0.1)
        self.assertEqual(datasets["lastfm"]["ssm_tau"], 0.5)
        self.assertEqual(datasets["ml-1m"]["ssm_tau"], 0.1)


if __name__ == "__main__":
    unittest.main()
