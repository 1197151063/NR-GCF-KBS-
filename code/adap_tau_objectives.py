"""Numerically stable objective primitives matching the Adap-tau LGN code.

The reference LightGCN configuration uses ``sampling_method=no_sample``.
Consequently, every other positive item in a batch is treated as a negative;
the command-line ``n_negs`` value is not used on that path.
"""

import math

import torch
import torch.nn.functional as F


def _normalized_in_batch_scores(user_embedding, positive_item_embedding):
    if user_embedding.dim() != 2 or positive_item_embedding.dim() != 2:
        raise ValueError("in-batch embeddings must both be rank-two")
    if user_embedding.shape != positive_item_embedding.shape:
        raise ValueError("user and positive-item batches must have equal shape")
    if user_embedding.size(0) < 2:
        raise ValueError("in-batch SSM requires at least two interactions")
    user_embedding = F.normalize(user_embedding, dim=-1)
    positive_item_embedding = F.normalize(positive_item_embedding, dim=-1)
    scores = user_embedding @ positive_item_embedding.t()
    positive = scores.diagonal()
    negative_mask = ~torch.eye(
        scores.size(0), dtype=torch.bool, device=scores.device
    )
    negatives = scores.masked_fill(~negative_mask, float("-inf"))
    return positive, negatives


def ssm_in_batch_instance_loss(
        user_embedding, positive_item_embedding, temperature):
    """Reference SSM with B-1 in-batch negatives and no positive denominator."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("SSM temperature must be finite and positive")
    positive, negatives = _normalized_in_batch_scores(
        user_embedding, positive_item_embedding
    )
    return (
        torch.logsumexp(negatives / temperature, dim=1)
        - positive / temperature
    )


def adap_tau_in_batch_instance_loss(
        user_embedding,
        positive_item_embedding,
        user_inverse_temperature,
        positive_inverse_temperature):
    """Reference Adap-tau loss and its detached unit-temperature statistic."""
    positive, negatives = _normalized_in_batch_scores(
        user_embedding, positive_item_embedding
    )
    user_inverse_temperature = user_inverse_temperature.to(
        device=positive.device, dtype=positive.dtype
    )
    if user_inverse_temperature.shape != positive.shape:
        raise ValueError("one adaptive inverse temperature is required per row")
    positive_inverse_temperature = torch.as_tensor(
        positive_inverse_temperature,
        device=positive.device,
        dtype=positive.dtype,
    )
    optimized = (
        torch.logsumexp(
            negatives * user_inverse_temperature.unsqueeze(1), dim=1
        )
        - positive * positive_inverse_temperature
    )
    unit_temperature = (
        torch.logsumexp(negatives, dim=1) - positive
    ).detach()
    return optimized, unit_temperature


@torch.no_grad()
def principal_lambert_w(value, iterations=12):
    """Real principal Lambert W for x >= -1/e using Halley iterations.

    The reference repository uses a roughly 40 MB SciPy lookup table.  This
    evaluates the same principal branch directly on the active device and
    avoids both the table and a SciPy runtime dependency.
    """
    value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.to(torch.float32)
    branch_point = -1.0 / math.e
    eps = torch.finfo(value.dtype).eps
    x = value.clamp(min=branch_point + 4.0 * eps, max=1000.0)
    branch_initial = -1.0 + torch.sqrt(
        (2.0 * (math.e * x + 1.0)).clamp_min(0.0)
    )
    regular_initial = torch.where(x < 1.0, x, torch.log1p(x))
    w = torch.where(x < -0.25, branch_initial, regular_initial)
    for _ in range(int(iterations)):
        exp_w = torch.exp(w)
        residual = w * exp_w - x
        denominator = (
            exp_w * (w + 1.0)
            - (w + 2.0) * residual
            / (2.0 * (w + 1.0)).clamp_min(8.0 * eps)
        )
        step = residual / denominator.clamp_min(8.0 * eps)
        w = w - step
    return w


@torch.no_grad()
def adap_tau_inverse_temperature(
        previous_user_loss,
        base_inverse_temperature,
        mode="weight_mean",
        temperature_2=1.5,
        loss_quantile=1.0):
    """Map the previous epoch's per-user loss to Adap-tau multipliers."""
    if mode not in {"weight_v0", "weight_mean", "weight_ratio"}:
        raise ValueError("unsupported Adap-tau mode: " + str(mode))
    base = torch.as_tensor(
        base_inverse_temperature,
        device=previous_user_loss.device,
        dtype=previous_user_loss.dtype,
    )
    if mode == "weight_v0":
        return torch.ones_like(previous_user_loss) * base
    temperature_2 = float(temperature_2)
    if not math.isfinite(temperature_2) or temperature_2 <= 0.0:
        raise ValueError("Adap-tau temperature_2 must be finite and positive")
    if mode == "weight_mean":
        center = previous_user_loss.mean()
    else:
        loss_quantile = float(loss_quantile)
        if not 0.0 <= loss_quantile <= 1.0:
            raise ValueError("Adap-tau loss quantile must be within [0, 1]")
        center = torch.quantile(previous_user_loss, loss_quantile)
    normalized = ((previous_user_loss - center) / temperature_2).clamp(
        min=-1.0 / math.e, max=1000.0
    )
    return (base * torch.exp(-principal_lambert_w(normalized))).detach()


def initial_adap_tau_inverse_temperature(
        high_degree_user_count,
        high_degree_interaction_count,
        num_items,
        assumed_positive_gap=0.7):
    """Stable equivalent of the reference code's cancellation-prone formula."""
    high_degree_user_count = int(high_degree_user_count)
    high_degree_interaction_count = int(high_degree_interaction_count)
    num_items = int(num_items)
    assumed_positive_gap = float(assumed_positive_gap)
    if min(high_degree_user_count, high_degree_interaction_count, num_items) < 1:
        raise ValueError("Adap-tau calibration sets must be non-empty")
    if not math.isfinite(assumed_positive_gap) or assumed_positive_gap <= 0.0:
        raise ValueError("assumed positive gap must be finite and positive")
    c_value = 2.0 * (
        math.log(0.5)
        + math.log(high_degree_user_count * num_items)
        - math.log(high_degree_interaction_count)
    )
    # Reference: (-b - sqrt(b^2-a*c))/a with b=-gap and a=1e-10.
    # Rationalization gives the stable expression below.
    a_value = 1e-10
    discriminant = max(
        assumed_positive_gap ** 2 - a_value * c_value, 0.0
    )
    return c_value / (
        assumed_positive_gap + math.sqrt(discriminant)
    )
