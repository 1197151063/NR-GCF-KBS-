import math
import os
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

if np is not None:
    import edge_reliability
else:
    edge_reliability = None


@unittest.skipUnless(np is not None, "NumPy unavailable")
class EdgeReliabilityMathTest(unittest.TestCase):
    def test_percentile_ranks_average_exact_ties(self):
        ranks = edge_reliability.percentile_ranks(
            np.array([3.0, 1.0, 1.0, np.nan, 2.0])
        )
        np.testing.assert_allclose(
            ranks[[0, 1, 2, 4]],
            np.array([1.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
        )
        self.assertTrue(math.isnan(ranks[3]))

    def test_available_side_mean_uses_single_finite_side(self):
        score, count = edge_reliability._available_side_mean(
            np.array([0.2, np.nan, 0.4, np.nan]),
            np.array([0.6, 0.3, np.nan, np.nan]),
        )
        np.testing.assert_allclose(score[:3], np.array([0.4, 0.3, 0.4]))
        self.assertTrue(math.isnan(score[3]))
        np.testing.assert_array_equal(count, np.array([2, 1, 1, 0]))

    def test_binary_metrics_has_expected_direction(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        score = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = edge_reliability._binary_metrics(labels, score, True)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)


@unittest.skipUnless(
    edge_reliability is not None and edge_reliability.torch is not None,
    "NumPy or PyTorch unavailable",
)
class EdgeReliabilityPolicyTest(unittest.TestCase):
    def test_soft_policy_keeps_bpr_edges_and_protects_degree_one(self):
        torch = edge_reliability.torch
        edges = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 1, 2],
        ], dtype=torch.long)
        momentum = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.4, 0.7])
        policy = edge_reliability.build_reliability_policy(
            edge_index=edges,
            raw_momentum=momentum,
            num_users=3,
            num_items=3,
            mode="soft_reliability",
            topk=2,
            chunk_size=3,
            min_degree=1,
            momentum_quantile=0.8,
            structure_quantile=0.2,
            structure_weight=0.95,
            minimum_weight=0.1,
        )
        self.assertTrue(bool(policy["retained_mask"].all()))
        weights = policy["propagation_weight"].numpy()
        self.assertTrue(bool(np.all(weights >= 0.1)))
        self.assertTrue(bool(np.all(weights <= 1.0)))
        # Item 2 has degree one, so its only incident edge must have full weight.
        self.assertEqual(float(weights[5]), 1.0)


if __name__ == "__main__":
    unittest.main()
