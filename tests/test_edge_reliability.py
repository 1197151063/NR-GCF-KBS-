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
    def test_early_pilot_triggers_after_one_stable_comparison(self):
        trigger = edge_reliability.AdaptiveFilteringTrigger(
            min_epoch=2,
            max_epoch=4,
            min_coverage=0.99,
            jaccard_threshold=0.90,
            stable_checks=1,
        )
        retained = np.array([False, True, True, False])
        fired, _ = trigger.observe(2, 1.0, retained)
        self.assertFalse(fired)
        fired, row = trigger.observe(3, 1.0, retained)
        self.assertTrue(fired)
        self.assertEqual(row["trigger_reason"], "coverage_and_removed_set_stable")
        self.assertEqual(trigger.trigger_epoch, 3)

    def test_adaptive_trigger_requires_two_stable_checks(self):
        trigger = edge_reliability.AdaptiveFilteringTrigger(
            min_epoch=5,
            max_epoch=10,
            min_coverage=0.99,
            jaccard_threshold=0.90,
            stable_checks=2,
        )
        retained = np.array([False, True, True, False])
        fired, _ = trigger.observe(5, 1.0, retained)
        self.assertFalse(fired)
        fired, _ = trigger.observe(6, 1.0, retained)
        self.assertFalse(fired)
        fired, row = trigger.observe(7, 1.0, retained)
        self.assertTrue(fired)
        self.assertEqual(row["trigger_reason"], "coverage_and_removed_set_stable")
        self.assertEqual(trigger.trigger_epoch, 7)

    def test_adaptive_trigger_forces_at_max_epoch(self):
        trigger = edge_reliability.AdaptiveFilteringTrigger(
            min_epoch=5,
            max_epoch=6,
            min_coverage=0.99,
            jaccard_threshold=0.90,
            stable_checks=2,
        )
        fired, _ = trigger.observe(
            5, 0.5, np.array([False, True, True])
        )
        self.assertFalse(fired)
        fired, row = trigger.observe(
            6, 0.5, np.array([True, False, True])
        )
        self.assertTrue(fired)
        self.assertEqual(row["trigger_reason"], "maximum_epoch_reached")

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

    def test_structure_only_matches_count_and_protects_edges(self):
        retained = edge_reliability._structure_only_retained_mask(
            structure=np.array([0.1, 0.1, 0.2, np.nan, 0.0]),
            protected=np.array([False, False, False, False, True]),
            target_remove_count=2,
        )
        # Equal scores are resolved by stable edge_id; protected edge 4 remains.
        np.testing.assert_array_equal(
            retained, np.array([False, False, True, True, True])
        )

    def test_gated_soft_risk_is_zero_outside_consensus_tail(self):
        risk = edge_reliability._gated_soft_risk(
            momentum_rank=np.array([0.7, 0.9, 1.0, 0.9]),
            structure_rank=np.array([0.1, 0.3, 0.0, 0.1]),
            momentum_quantile=0.8,
            structure_quantile=0.2,
        )
        self.assertEqual(float(risk[0]), 0.0)
        self.assertEqual(float(risk[1]), 0.0)
        self.assertEqual(float(risk[2]), 1.0)
        self.assertGreater(float(risk[3]), 0.0)
        self.assertLess(float(risk[3]), 1.0)

    def test_top_fused_risk_matches_budget_with_stable_ties(self):
        retained = edge_reliability._top_risk_retained_mask(
            risk=np.array([0.9, 0.9, 0.2, np.nan, 0.8]),
            target_remove_count=3,
        )
        np.testing.assert_array_equal(
            retained, np.array([False, False, True, True, False])
        )


@unittest.skipUnless(
    edge_reliability is not None and edge_reliability.torch is not None,
    "NumPy or PyTorch unavailable",
)
class EdgeReliabilityPolicyTest(unittest.TestCase):
    def test_stable_edge_momentum_initializes_then_updates_ema(self):
        torch = edge_reliability.torch
        tracker = edge_reliability.StableEdgeMomentum(
            edge_count=3, decay=0.9, device=torch.device("cpu")
        )
        tracker.update(torch.tensor([0, 2]), torch.tensor([1.0, 3.0]))
        tracker.update(torch.tensor([0, 1, 2]), torch.tensor([2.0, 4.0, 1.0]))
        np.testing.assert_allclose(
            tracker.snapshot().numpy(), np.array([1.1, 4.0, 2.8]), rtol=1e-6
        )

    def test_stable_edge_momentum_exposes_coverage_and_neutral_snapshot(self):
        torch = edge_reliability.torch
        tracker = edge_reliability.StableEdgeMomentum(
            edge_count=3, decay=0.9, device=torch.device("cpu")
        )
        tracker.update(torch.tensor([0, 2]), torch.tensor([1.0, 3.0]))
        self.assertAlmostEqual(tracker.coverage(), 2.0 / 3.0, places=6)
        snapshot = tracker.snapshot(require_all=False)
        self.assertTrue(bool(torch.isnan(snapshot[1])))
        self.assertEqual(tracker.observation_counts().tolist(), [1, 0, 1])

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

    def test_unobserved_momentum_is_neutral_and_excluded_from_budget(self):
        torch = edge_reliability.torch
        edges = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 1, 2],
        ], dtype=torch.long)
        momentum = torch.tensor([0.9, float("nan"), 0.8, 0.1, 0.4, 0.7])
        observed = torch.tensor([True, False, True, True, True, True])
        policy = edge_reliability.build_reliability_policy(
            edge_index=edges,
            raw_momentum=momentum,
            num_users=3,
            num_items=3,
            mode="hard_structure_momentum",
            topk=2,
            chunk_size=3,
            min_degree=1,
            momentum_quantile=0.8,
            structure_quantile=0.2,
            structure_weight=0.95,
            minimum_weight=0.1,
            momentum_observed_mask=observed,
        )
        self.assertEqual(float(policy["momentum_rank"][1]), 0.5)
        self.assertFalse(bool(policy["consensus_candidate"][1]))
        self.assertEqual(policy["momentum_unobserved_count"], 1)


if __name__ == "__main__":
    unittest.main()
