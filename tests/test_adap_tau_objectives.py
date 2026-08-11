import math
import os
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

if torch is not None:
    from adap_tau_objectives import (
        adap_tau_in_batch_instance_loss,
        adap_tau_inverse_temperature,
        initial_adap_tau_inverse_temperature,
        principal_lambert_w,
        ssm_in_batch_instance_loss,
    )


@unittest.skipUnless(torch is not None, "PyTorch unavailable")
class AdapTauObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.users = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        self.items = torch.tensor([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.6, 0.8],
        ])

    def test_ssm_matches_reference_swap_construction(self):
        tau = 0.2
        users = torch.nn.functional.normalize(self.users, dim=-1)
        items = torch.nn.functional.normalize(self.items, dim=-1)
        scores = users @ items.t()
        rows = torch.arange(scores.size(0))
        reference = []
        for row in rows:
            negatives = scores[row, rows != row]
            reference.append(
                torch.logsumexp(negatives / tau, dim=0)
                - scores[row, row] / tau
            )
        actual = ssm_in_batch_instance_loss(self.users, self.items, tau)
        torch.testing.assert_close(actual, torch.stack(reference))

    def test_constant_adap_tau_equals_ssm(self):
        inverse_temperature = 5.0
        adap, base = adap_tau_in_batch_instance_loss(
            self.users,
            self.items,
            torch.full((3,), inverse_temperature),
            inverse_temperature,
        )
        ssm = ssm_in_batch_instance_loss(
            self.users, self.items, 1.0 / inverse_temperature
        )
        torch.testing.assert_close(adap, ssm)
        self.assertFalse(base.requires_grad)

    def test_lambert_w_principal_branch(self):
        values = torch.tensor([-0.3, 0.0, 0.5, 10.0, 1000.0])
        result = principal_lambert_w(values)
        torch.testing.assert_close(
            result * result.exp(), values, rtol=2e-5, atol=2e-5
        )

    def test_weight_mean_is_centered_around_base(self):
        losses = torch.tensor([1.0, 2.0, 3.0])
        inverse_temperature = adap_tau_inverse_temperature(
            losses, 8.0, mode="weight_mean", temperature_2=1.5
        )
        self.assertAlmostEqual(float(inverse_temperature[1]), 8.0, places=5)
        self.assertGreater(float(inverse_temperature[0]), 8.0)
        self.assertLess(float(inverse_temperature[2]), 8.0)

    def test_initial_inverse_temperature_is_positive(self):
        value = initial_adap_tau_inverse_temperature(
            high_degree_user_count=100,
            high_degree_interaction_count=1000,
            num_items=500,
            assumed_positive_gap=0.7,
        )
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
