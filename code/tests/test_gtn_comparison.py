import pathlib
import sys
import unittest

import torch


CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from model import ObjectiveGTN


def config(objective):
    return {
        "init": "normal",
        "init_weight": 0.1,
        "dim": 4,
        "decay": 1e-4,
        "K": 3,
        "lambda": 0.0,
        "training_objective": objective,
        "num_neg": 2,
        "tau": 0.1,
        "objective_message_dropout": 0.0,
        "adap_tau_mode": "weight_mean",
        "adap_tau_temperature_2": 1.5,
        "adap_tau_loss_quantile": 1.0,
        "adap_tau_recalibration_epoch": 100,
        "adap_tau_degree_quantile": 0.2,
        "adap_tau_initial_positive_gap": 0.7,
        "au_uniformity_weight": 1.0,
        "au_uniformity_t": 2.0,
        "gtn_lambda": 1.0,
        "gtn_prop_dropout": 0.1,
        "representation_modulation_mode": "none",
        "representation_modulation_ramp_epochs": 0,
    }


class GTNComparisonTest(unittest.TestCase):
    def setUp(self):
        self.edges = torch.tensor(
            [[0, 0, 1], [0, 1, 1]], dtype=torch.long
        )

    def test_uses_sparse_edge_node_incidence_and_one_embedding_pair(self):
        model = ObjectiveGTN(2, 2, config("bpr"), self.edges)
        self.assertEqual(model.incident_matrix.sizes(), (3, 4))
        self.assertEqual(model.incident_matrix.nnz(), 6)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()), 16
        )
        metadata = model.objective_metadata()
        self.assertEqual(metadata["backbone"], "gtn")
        self.assertFalse(metadata["dense_node_node_matrix"])
        self.assertFalse(metadata["cross_type_normalization"])

    def test_all_three_objectives_have_finite_gradients(self):
        for objective in ("bpr", "ssm", "au"):
            with self.subTest(objective=objective):
                model = ObjectiveGTN(2, 2, config(objective), self.edges)
                if objective == "bpr":
                    labels = torch.tensor(
                        [[0, 1], [0, 1], [1, 0]], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [[0, 1], [0, 1]], dtype=torch.long
                    )
                loss = model.get_loss(labels)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_evaluation_is_deterministic_and_returns_full_embeddings(self):
        model = ObjectiveGTN(2, 2, config("bpr"), self.edges).eval()
        first_users, first_items = model(model.edge_index)
        second_users, second_items = model(model.edge_index)
        self.assertEqual(first_users.shape, (2, 4))
        self.assertEqual(first_items.shape, (2, 4))
        self.assertTrue(torch.equal(first_users, second_users))
        self.assertTrue(torch.equal(first_items, second_items))


if __name__ == "__main__":
    unittest.main()
