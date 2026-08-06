"""Compact, training-side edge reliability policies for NR-GCF pilots.

The module is deliberately separate from the loss implementation.  It reads
the epoch-15 momentum values and the pre-filter training graph, computes a
deterministic structural score, and returns either a retained-edge mask or
propagation-only edge weights.  Synthetic labels are read only after the
decision has been made and are used exclusively for JSON summary statistics.
"""

from __future__ import print_function

import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

try:
    import torch
except ImportError:  # Keep lightweight rank/statistic tests importable.
    torch = None


SCHEMA_VERSION = "nrgcf_edge_reliability_pilot_v7"


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for edge reliability policies")


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _average_ranks(values):
    """Return deterministic average ranks, including exact tie handling."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    repeated = np.repeat(0.5 * (starts + ends - 1), ends - starts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = repeated
    return ranks


def percentile_ranks(values):
    """Finite values map to [0, 1]; missing values remain NaN."""
    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(array)
    count = int(finite.sum())
    if count == 0:
        return result
    if count == 1:
        result[finite] = 0.5
    else:
        result[finite] = _average_ranks(array[finite]) / float(count - 1)
    return result


def _stats(values, mask=None):
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    population = int(array.size)
    if mask is not None:
        selected_mask = np.asarray(mask, dtype=bool)
        valid &= selected_mask
        population = int(selected_mask.sum())
    selected = array[valid]
    if selected.size == 0:
        return {
            "count": 0, "missing_count": population, "mean": None,
            "std": None, "min": None, "q20": None, "median": None,
            "q80": None, "max": None,
        }
    return {
        "count": int(selected.size),
        "missing_count": int(population - selected.size),
        "mean": float(selected.mean()),
        "std": float(selected.std()),
        "min": float(selected.min()),
        "q20": float(np.quantile(selected, 0.20)),
        "median": float(np.quantile(selected, 0.50)),
        "q80": float(np.quantile(selected, 0.80)),
        "max": float(selected.max()),
    }


def _binary_metrics(labels, raw_score, higher_is_noisy=True):
    labels = np.asarray(labels, dtype=np.int8)
    score = np.asarray(raw_score, dtype=np.float64)
    if not higher_is_noisy:
        score = -score
    valid = np.isfinite(score) & (labels >= 0)
    y = labels[valid]
    score = score[valid]
    positive = int(y.sum())
    negative = int(y.size - positive)
    empty = {
        "count": int(y.size), "positive_count": positive,
        "auroc": None, "average_precision": None,
    }
    if positive == 0 or negative == 0:
        return empty

    ranks = _average_ranks(score) + 1.0
    auc = (float(ranks[y == 1].sum()) - positive * (positive + 1) / 2.0) / (
        positive * negative
    )

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_score.size]
    cumulative_positive = np.cumsum(sorted_y, dtype=np.int64)
    positives_at_end = cumulative_positive[ends - 1]
    positives_before = np.r_[0, positives_at_end[:-1]]
    group_positive = positives_at_end - positives_before
    group_precision = positives_at_end / ends.astype(np.float64)
    return {
        "count": int(y.size),
        "positive_count": positive,
        "auroc": float(auc),
        "average_precision": float(np.sum(group_precision * group_positive) / positive),
    }


def _parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes"):
        return 1
    if normalized in ("0", "false", "no"):
        return 0
    raise ValueError("Invalid boolean label: %r" % value)


def _load_labels(path, edge_index_cpu):
    if not path:
        return None
    edge_count = int(edge_index_cpu.size(1))
    labels = np.full(edge_count, -1, dtype=np.int8)
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"edge_id", "user_id_internal", "item_id_internal", "synthetic_is_noisy"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Synthetic label CSV is missing required identity columns")
        for expected, row in enumerate(reader):
            if expected >= edge_count:
                raise ValueError("Synthetic label CSV contains extra rows")
            edge_id = int(row["edge_id"])
            user = int(row["user_id_internal"])
            item = int(row["item_id_internal"])
            if edge_id != expected:
                raise ValueError("Synthetic label edge_id mismatch at row %d" % expected)
            if user != int(edge_index_cpu[0, expected]) or item != int(edge_index_cpu[1, expected]):
                raise ValueError("Synthetic label endpoint mismatch at edge_id %d" % expected)
            labels[expected] = _parse_bool(row["synthetic_is_noisy"])
    if bool((labels < 0).any()):
        raise ValueError("Synthetic label CSV row count does not match training edges")
    return labels


def _git_commit(repo_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _available_side_mean(user_side, item_side):
    user_side = np.asarray(user_side, dtype=np.float64)
    item_side = np.asarray(item_side, dtype=np.float64)
    user_valid = np.isfinite(user_side)
    item_valid = np.isfinite(item_side)
    count = user_valid.astype(np.int8) + item_valid.astype(np.int8)
    total = np.where(user_valid, user_side, 0.0) + np.where(item_valid, item_side, 0.0)
    result = np.full(user_side.shape, np.nan, dtype=np.float64)
    valid = count > 0
    result[valid] = total[valid] / count[valid]
    return result, count


class StableEdgeMomentum(object):
    """Per-edge EMA updated from already-computed detached instance losses."""

    def __init__(self, edge_count, decay, device):
        _require_torch()
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and within [0, 1)")
        self.decay = decay
        self.values = torch.zeros(int(edge_count), dtype=torch.float32, device=device)
        self.seen = torch.zeros(int(edge_count), dtype=torch.bool, device=device)

    @torch.no_grad() if torch is not None else (lambda function: function)
    def update(self, edge_ids, losses):
        edge_ids = edge_ids.detach().to(device=self.values.device, dtype=torch.long)
        losses = losses.detach().to(device=self.values.device, dtype=torch.float32)
        first = ~self.seen[edge_ids]
        updated = (
            self.decay * self.values[edge_ids]
            + (1.0 - self.decay) * losses
        )
        self.values[edge_ids] = torch.where(first, losses, updated)
        self.seen[edge_ids] = True

    @torch.no_grad() if torch is not None else (lambda function: function)
    def snapshot(self):
        missing = int((~self.seen).sum().detach().cpu().item())
        if missing:
            raise RuntimeError(
                "Stable edge momentum has %d unobserved training edges" % missing
            )
        return self.values.detach().clone()


def _structure_only_retained_mask(structure, protected, target_remove_count):
    """Remove the lowest-structure eligible edges with stable edge-ID ties."""
    structure = np.asarray(structure, dtype=np.float64)
    protected = np.asarray(protected, dtype=bool)
    target_remove_count = int(target_remove_count)
    if target_remove_count < 0:
        raise ValueError("target_remove_count cannot be negative")
    eligible = np.isfinite(structure) & ~protected
    eligible_ids = np.flatnonzero(eligible)
    if target_remove_count > eligible_ids.size:
        raise ValueError(
            "Not enough finite, unprotected structural scores for matched filtering"
        )
    retained = np.ones(structure.size, dtype=bool)
    if target_remove_count == 0:
        return retained
    order = np.lexsort((eligible_ids, structure[eligible_ids]))
    removed_ids = eligible_ids[order[:target_remove_count]]
    retained[removed_ids] = False
    return retained


def _top_risk_retained_mask(risk, target_remove_count):
    """Remove a fixed number of highest-risk finite edges; no graph constraint."""
    risk = np.asarray(risk, dtype=np.float64)
    target_remove_count = int(target_remove_count)
    if target_remove_count < 0:
        raise ValueError("target_remove_count cannot be negative")
    eligible_ids = np.flatnonzero(np.isfinite(risk))
    if target_remove_count > eligible_ids.size:
        raise ValueError("Not enough finite fused risks for the adaptive budget")
    retained = np.ones(risk.size, dtype=bool)
    if target_remove_count == 0:
        return retained
    order = np.lexsort((eligible_ids, -risk[eligible_ids]))
    retained[eligible_ids[order[:target_remove_count]]] = False
    return retained


def node_confidence_from_edge_reliability(
        edge_index_cpu, reliability, retained_mask, num_users, num_items):
    """Aggregate frozen retained-edge confidence into deterministic nodes.

    Missing edge reliability is treated as neutral confidence one.  Nodes with
    no retained incident edge receive zero and therefore cannot determine the
    reliability-weighted global scale.  No labels, embeddings, or future epoch
    information are used.
    """
    users = edge_index_cpu[0].numpy()
    items = edge_index_cpu[1].numpy()
    reliability = np.asarray(reliability, dtype=np.float64)
    retained = np.asarray(retained_mask, dtype=bool)
    if reliability.size != users.size or retained.size != users.size:
        raise ValueError('edge reliability identity does not match edge_index')
    confidence = np.where(np.isfinite(reliability), reliability, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)
    active_confidence = confidence[retained]
    active_users = users[retained]
    active_items = items[retained]
    user_count = np.bincount(active_users, minlength=int(num_users))
    item_count = np.bincount(active_items, minlength=int(num_items))
    user_sum = np.bincount(
        active_users, weights=active_confidence, minlength=int(num_users)
    )
    item_sum = np.bincount(
        active_items, weights=active_confidence, minlength=int(num_items)
    )
    user_confidence = np.zeros(int(num_users), dtype=np.float32)
    item_confidence = np.zeros(int(num_items), dtype=np.float32)
    user_valid = user_count > 0
    item_valid = item_count > 0
    user_confidence[user_valid] = (
        user_sum[user_valid] / user_count[user_valid]
    ).astype(np.float32)
    item_confidence[item_valid] = (
        item_sum[item_valid] / item_count[item_valid]
    ).astype(np.float32)
    return user_confidence, item_confidence


def _gated_soft_risk(momentum_rank, structure_rank,
                     momentum_quantile, structure_quantile):
    """Continuous risk restricted to the hard-consensus tail quadrant."""
    momentum_rank = np.asarray(momentum_rank, dtype=np.float64)
    structure_rank = np.asarray(structure_rank, dtype=np.float64)
    momentum_denominator = max(1.0 - float(momentum_quantile), 1e-12)
    structure_denominator = max(float(structure_quantile), 1e-12)
    momentum_tail = np.clip(
        (momentum_rank - float(momentum_quantile)) / momentum_denominator,
        0.0, 1.0,
    )
    structure_tail = np.clip(
        (float(structure_quantile) - structure_rank) / structure_denominator,
        0.0, 1.0,
    )
    return np.sqrt(momentum_tail * structure_tail)


@torch.no_grad() if torch is not None else (lambda function: function)
def build_reliability_policy(
        edge_index, raw_momentum, num_users, num_items, mode, topk,
        chunk_size, min_degree, momentum_quantile, structure_quantile,
        structure_weight, minimum_weight,
        momentum_semantics="legacy_runtime_momentum"):
    """Compute a frozen filtering policy without reading evaluation labels."""
    _require_torch()
    if mode not in (
            "none", "hard_consensus", "hard_structure_only",
            "soft_reliability", "gated_soft_reliability",
            "hard_structure_momentum"):
        raise ValueError("Unsupported reliability mode: %s" % mode)
    for name, value in (
            ("momentum_quantile", momentum_quantile),
            ("structure_quantile", structure_quantile),
            ("structure_weight", structure_weight),
            ("minimum_weight", minimum_weight)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("%s must be finite and within [0, 1]" % name)
    if int(chunk_size) <= 0 or int(topk) <= 0:
        raise ValueError("topk and chunk_size must be positive")
    if int(min_degree) < 1:
        raise ValueError("min_degree must be at least 1")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    edge_count = int(edge_index.size(1))
    if raw_momentum.numel() != edge_count:
        raise ValueError("raw_momentum must contain one value per training edge")

    # Reuse the diagnostics estimator, but retain only two side means in CPU
    # memory.  No full node-by-node matrix or edge-pair Cartesian product is
    # constructed.
    from edge_diagnostics import TwoHopMinHash
    engine = TwoHopMinHash(
        edge_index=edge_index.detach(),
        num_users=num_users,
        num_items=num_items,
        topk=topk,
        structural_mode="two_hop_minhash",
    )
    user_side = np.full(edge_count, np.nan, dtype=np.float32)
    item_side = np.full(edge_count, np.nan, dtype=np.float32)
    for start in range(0, edge_count, int(chunk_size)):
        end = min(start + int(chunk_size), edge_count)
        chunk = engine.compute_chunk(start, end)
        user_side[start:end] = chunk["user_side_structure_mean"].numpy()
        item_side[start:end] = chunk["item_side_structure_mean"].numpy()
    del engine

    structure, valid_side_count = _available_side_mean(user_side, item_side)
    momentum = raw_momentum.detach().to(device="cpu", dtype=torch.float64).numpy()
    momentum_rank = percentile_ranks(momentum)
    structure_rank = percentile_ranks(structure)
    clean_loss_rank = 1.0 - momentum_rank
    reliability = (
        float(structure_weight) * structure_rank
        + (1.0 - float(structure_weight)) * clean_loss_rank
    )

    users = edge_index_cpu[0].numpy()
    items = edge_index_cpu[1].numpy()
    user_degree_all = np.bincount(users, minlength=int(num_users))
    item_degree_all = np.bincount(items, minlength=int(num_items))
    user_degree = user_degree_all[users]
    item_degree = item_degree_all[items]
    protected = (
        (user_degree - 1 < int(min_degree))
        | (item_degree - 1 < int(min_degree))
    )

    finite_momentum = momentum[np.isfinite(momentum)]
    finite_structure = structure[np.isfinite(structure)]
    momentum_threshold = (
        float(np.quantile(finite_momentum, float(momentum_quantile)))
        if finite_momentum.size else None
    )
    structure_threshold = (
        float(np.quantile(finite_structure, float(structure_quantile)))
        if finite_structure.size else None
    )
    high_momentum = (
        np.isfinite(momentum) & (momentum >= momentum_threshold)
        if momentum_threshold is not None else np.zeros(edge_count, dtype=bool)
    )
    low_structure = (
        np.isfinite(structure) & (structure <= structure_threshold)
        if structure_threshold is not None else np.zeros(edge_count, dtype=bool)
    )
    consensus_candidate = high_momentum & low_structure
    consensus_remove_count = int((consensus_candidate & ~protected).sum())
    adaptive_budget_count = int(consensus_candidate.sum())
    fused_risk = 1.0 - reliability

    retained = np.ones(edge_count, dtype=bool)
    propagation_weight = np.ones(edge_count, dtype=np.float32)
    gated_risk = _gated_soft_risk(
        momentum_rank, structure_rank,
        momentum_quantile=momentum_quantile,
        structure_quantile=structure_quantile,
    )
    if mode == "hard_consensus":
        retained = ~(consensus_candidate & ~protected)
        propagation_weight = retained.astype(np.float32)
    elif mode == "hard_structure_only":
        retained = _structure_only_retained_mask(
            structure=structure,
            protected=protected,
            target_remove_count=consensus_remove_count,
        )
        propagation_weight = retained.astype(np.float32)
    elif mode == "hard_structure_momentum":
        retained = _top_risk_retained_mask(
            risk=fused_risk,
            target_remove_count=adaptive_budget_count,
        )
        propagation_weight = retained.astype(np.float32)
    elif mode == "soft_reliability":
        scorable = np.isfinite(reliability)
        propagation_weight[scorable] = (
            float(minimum_weight)
            + (1.0 - float(minimum_weight)) * reliability[scorable]
        ).astype(np.float32)
        # Unscorable and vulnerable long-tail edges receive full weight.
        propagation_weight[~scorable | protected] = 1.0
    elif mode == "gated_soft_reliability":
        scorable = np.isfinite(gated_risk)
        propagation_weight[scorable] = (
            1.0 - (1.0 - float(minimum_weight)) * gated_risk[scorable]
        ).astype(np.float32)
        # Only the high-momentum/low-structure tail is attenuated.  All other,
        # unscorable, and long-tail-protected edges keep unit propagation weight.
        propagation_weight[~scorable | protected] = 1.0

    user_node_confidence, item_node_confidence = (
        node_confidence_from_edge_reliability(
            edge_index_cpu=edge_index_cpu,
            reliability=reliability,
            retained_mask=retained,
            num_users=num_users,
            num_items=num_items,
        )
    )

    decision = {
        "mode": mode,
        "frozen_at_filter_epoch": True,
        "uses_synthetic_labels": False,
    }
    if mode == "hard_consensus":
        decision.update({
            "rule": "high_raw_momentum AND low_structure AND not degree_protected",
            "target_remove_count": consensus_remove_count,
        })
    elif mode == "hard_structure_only":
        decision.update({
            "rule": "lowest_structure among finite unprotected edges",
            "matched_to": "hard_consensus removal count on the same epoch-15 graph",
            "target_remove_count": consensus_remove_count,
            "actual_remove_count": int((~retained).sum()),
            "tie_break": "ascending stable edge_id",
        })
    elif mode == "gated_soft_reliability":
        decision.update({
            "rule": "unit weight outside consensus tail; smooth risk weight inside",
            "bpr_positive_sampling": "all edges, uniform",
            "edge_weight_scope": "LightGCN propagation only",
        })
    elif mode == "hard_structure_momentum":
        decision.update({
            "rule": "remove highest structure-dominant fused risks",
            "fused_risk": "structure_weight*(1-structure_rank) + (1-structure_weight)*momentum_rank",
            "budget": "count(high_momentum_quantile AND low_structure_quantile)",
            "target_remove_count": adaptive_budget_count,
            "actual_remove_count": int((~retained).sum()),
            "connectivity_constraint": "none by design",
            "tie_break": "ascending stable edge_id",
            "momentum_semantics": str(momentum_semantics),
        })

    return {
        "mode": mode,
        "edge_index_cpu": edge_index_cpu,
        "retained_mask": torch.from_numpy(retained),
        "propagation_weight": torch.from_numpy(propagation_weight),
        "momentum": momentum,
        "momentum_rank": momentum_rank,
        "user_side_structure": user_side,
        "item_side_structure": item_side,
        "structure": structure,
        "valid_side_count": valid_side_count,
        "structure_rank": structure_rank,
        "reliability": reliability,
        "fused_risk": fused_risk,
        "user_node_confidence": user_node_confidence,
        "item_node_confidence": item_node_confidence,
        "gated_soft_risk": gated_risk,
        "protected": protected,
        "consensus_candidate": consensus_candidate,
        "consensus_remove_count_after_protection": consensus_remove_count,
        "adaptive_budget_count_without_connectivity_constraint": adaptive_budget_count,
        "user_degree": user_degree,
        "item_degree": item_degree,
        "momentum_threshold": momentum_threshold,
        "structure_threshold": structure_threshold,
        "decision": decision,
        "momentum_semantics": str(momentum_semantics),
        "parameters": {
            "topk": int(topk),
            "chunk_size": int(chunk_size),
            "min_degree": int(min_degree),
            "momentum_quantile": float(momentum_quantile),
            "structure_quantile": float(structure_quantile),
            "structure_weight": float(structure_weight),
            "minimum_weight": float(minimum_weight),
        },
    }


def write_reliability_summary(
        output_dir, policy, dataset, seed, requested_noise_ratio,
        filtering_epoch, labels_path=None, noise_validation_path=None,
        repo_dir=None):
    """Write one compact JSON report; labels cannot affect the policy."""
    os.makedirs(output_dir, exist_ok=True)
    labels = _load_labels(labels_path, policy["edge_index_cpu"])
    retained = policy["retained_mask"].numpy().astype(bool)
    weights = policy["propagation_weight"].numpy().astype(np.float64)
    protected = policy["protected"]
    consensus = policy["consensus_candidate"]
    structure = policy["structure"]
    momentum = policy["momentum"]
    reliability = policy["reliability"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "seed": int(seed),
        "requested_noise_ratio": requested_noise_ratio,
        "filtering_epoch": int(filtering_epoch),
        "warmup_epoch_count": int(
            policy.get("warmup_epoch_count", filtering_epoch - 1)
        ),
        "mode": policy["mode"],
        "edge_count": int(retained.size),
        "retained_edge_count": int(retained.sum()),
        "removed_edge_count": int((~retained).sum()),
        "removed_ratio": float((~retained).mean()),
        "protected_edge_count": int(protected.sum()),
        "degree_protection_applied": policy["mode"] in (
            "hard_consensus", "hard_structure_only",
            "soft_reliability", "gated_soft_reliability",
        ),
        "consensus_candidate_count_before_protection": int(consensus.sum()),
        "consensus_candidate_protected_count": int((consensus & protected).sum()),
        "consensus_remove_count_after_protection": int(
            policy["consensus_remove_count_after_protection"]
        ),
        "adaptive_budget_count_without_connectivity_constraint": int(
            policy["adaptive_budget_count_without_connectivity_constraint"]
        ),
        "momentum_semantics": policy.get("momentum_semantics"),
        "representation_modulation": policy.get("representation_modulation"),
        "thresholds": {
            "momentum_high": policy["momentum_threshold"],
            "available_side_structure_low": policy["structure_threshold"],
        },
        "parameters": policy["parameters"],
        "decision": policy.get("decision"),
        "policies": {
            "none": "All training edges remain unweighted.",
            "current": "Exact current-code min-max/beta filtering; compact reporting does not alter the decision.",
            "hard_consensus": "Remove only high-momentum AND low-structure edges unless removal would make either endpoint degree fall below min_degree.",
            "hard_structure_only": "Remove the same number as protected hard_consensus, selecting only the lowest-structure finite unprotected edges with edge_id tie-breaking.",
            "hard_structure_momentum": "Use stable EMA loss to calibrate an adaptive consensus budget, then remove that many highest structure-dominant fused-risk edges without connectivity constraints.",
            "soft_reliability": "Keep all BPR positives; use frozen reliability only as LightGCN propagation edge weights. Unscorable/degree-protected edges have weight 1.",
            "gated_soft_reliability": "Keep all BPR positives and unit propagation weights outside the high-momentum/low-structure tail; smoothly attenuate only that tail.",
            "bpr_objective": "Original unweighted NR-GCF BPR plus L2; no reliability-weighted loss.",
            "representation_modulation": "Configured separately: original_always applies direct cross_norm from epoch one; reliability_weighted_always preserves that operator and changes only its frozen RMS estimator after filtering.",
        },
        "feature_definitions": {
            "structure": "Arithmetic mean of the finite user-side/item-side leave-one-edge-out MinHash two-hop scores; a single available side is used on sparse edges.",
            "reliability": "structure_weight * percentile_rank(structure) + (1-structure_weight) * (1-percentile_rank(momentum_signal)).",
            "fused_risk": "structure_weight * (1-percentile_rank(structure)) + (1-structure_weight) * percentile_rank(momentum).",
            "gated_soft_risk": "sqrt(clip((momentum_rank-q_m)/(1-q_m),0,1) * clip((q_s-structure_rank)/q_s,0,1)); exactly zero outside the consensus tail.",
            "node_confidence": "Mean frozen reliability of retained incident edges. Missing retained-edge reliability is neutral one; nodes without retained edges receive zero weight.",
            "protection": "user_degree_after_if_removed < min_degree OR item_degree_after_if_removed < min_degree.",
            "label_usage": "Synthetic labels are loaded only after the frozen decision and are used only below for evaluation statistics.",
        },
        "statistics": {
            "momentum_signal": _stats(momentum),
            "user_side_structure": _stats(policy["user_side_structure"]),
            "item_side_structure": _stats(policy["item_side_structure"]),
            "available_side_structure": _stats(structure),
            "reliability": _stats(reliability),
            "fused_risk": _stats(policy["fused_risk"]),
            "user_node_confidence": _stats(policy["user_node_confidence"]),
            "item_node_confidence": _stats(policy["item_node_confidence"]),
            "gated_soft_risk": _stats(policy["gated_soft_risk"]),
            "propagation_weight": _stats(weights),
            "user_degree": _stats(policy["user_degree"]),
            "item_degree": _stats(policy["item_degree"]),
            "valid_structural_side_count": {
                "zero": int((policy["valid_side_count"] == 0).sum()),
                "one": int((policy["valid_side_count"] == 1).sum()),
                "two": int((policy["valid_side_count"] == 2).sum()),
            },
        },
        "synthetic_label_evaluation": None,
        "noise_validation": None,
        "command_line": " ".join(sys.argv),
        "code_commit_hash": _git_commit(repo_dir) if repo_dir else None,
    }

    if noise_validation_path:
        with open(noise_validation_path, encoding="utf-8") as stream:
            report["noise_validation"] = json.load(stream)
    if labels is not None:
        clean = labels == 0
        noisy = labels == 1
        removed = ~retained
        removed_count = int(removed.sum())
        report["synthetic_label_evaluation"] = {
            "clean_edge_count": int(clean.sum()),
            "noisy_edge_count": int(noisy.sum()),
            "clean_removal_rate": float((removed & clean).sum() / max(int(clean.sum()), 1)),
            "noisy_removal_rate": float((removed & noisy).sum() / max(int(noisy.sum()), 1)),
            "removed_precision_noisy": float((removed & noisy).sum() / max(removed_count, 1)),
            "scores_for_noisy_edge": {
                "momentum_signal": _binary_metrics(labels, momentum, True),
                "available_side_structure": _binary_metrics(labels, structure, False),
                "reliability": _binary_metrics(labels, reliability, False),
                "fused_risk": _binary_metrics(
                    labels, policy["fused_risk"], True
                ),
                "gated_soft_risk": _binary_metrics(
                    labels, policy["gated_soft_risk"], True
                ),
                "propagation_weight": _binary_metrics(labels, weights, False),
            },
            "clean_group": {
                "momentum": _stats(momentum, clean),
                "structure": _stats(structure, clean),
                "reliability": _stats(reliability, clean),
                "fused_risk": _stats(policy["fused_risk"], clean),
                "gated_soft_risk": _stats(policy["gated_soft_risk"], clean),
                "propagation_weight": _stats(weights, clean),
            },
            "noisy_group": {
                "momentum": _stats(momentum, noisy),
                "structure": _stats(structure, noisy),
                "reliability": _stats(reliability, noisy),
                "fused_risk": _stats(policy["fused_risk"], noisy),
                "gated_soft_risk": _stats(policy["gated_soft_risk"], noisy),
                "propagation_weight": _stats(weights, noisy),
            },
        }

    _write_json(os.path.join(output_dir, "reliability_summary.json"), report)
    _write_json(os.path.join(output_dir, "schema.json"), {
        "schema_version": SCHEMA_VERSION,
        "artifact": "summary_only",
        "per_edge_table_written": False,
        "files": [
            "reliability_summary.json", "training_summary.json", "schema.json"
        ],
    })
    return report


def write_training_summary(
        output_dir, mode, requested_epochs, epochs_completed, best_epoch,
        best_recall, best_ndcg, final_loss, propagation_edge_count,
        bpr_positive_edge_count, representation_modulation_mode,
        representation_modulation_ramp_epochs, representation_modulation_lambda,
        representation_modulation_trace, best_post_filter_epoch,
        best_post_filter_recall, best_post_filter_ndcg,
        early_stopping_patience, early_stopped, early_stopping_wait):
    """Write the compact outcome for a completed or early-stopped run."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "requested_epochs": int(requested_epochs),
        "epochs_completed": int(epochs_completed),
        "completed_requested_epochs": int(epochs_completed) == int(requested_epochs),
        "early_stopping": {
            "monitor": "Recall@20",
            "mode": "max",
            "strict_improvement_required": True,
            "patience": int(early_stopping_patience),
            "stopped_early": bool(early_stopped),
            "consecutive_non_improving_epochs": int(early_stopping_wait),
        },
        "best_epoch": int(best_epoch),
        "best_recall_at_20": float(best_recall),
        "best_ndcg_at_20": float(best_ndcg),
        "best_post_filter_epoch": (
            int(best_post_filter_epoch)
            if best_post_filter_epoch is not None else None
        ),
        "best_post_filter_recall_at_20": (
            float(best_post_filter_recall)
            if best_post_filter_recall is not None else None
        ),
        "best_post_filter_ndcg_at_20": (
            float(best_post_filter_ndcg)
            if best_post_filter_ndcg is not None else None
        ),
        "final_training_loss": float(final_loss),
        "propagation_edge_count": int(propagation_edge_count),
        "bpr_positive_edge_count": int(bpr_positive_edge_count),
        "objective": "Original NR-GCF BPR plus L2; reliability does not weight the loss.",
        "representation_modulation": {
            "mode": str(representation_modulation_mode),
            "ramp_epochs": int(representation_modulation_ramp_epochs),
            "lambda": None,
            "lambda_note": "Ignored by NRGCF direct cross_norm modes; retained only as a legacy CLI/config field.",
            "stage_one": (
                "ordinary LightGCN propagation without modulation"
                if representation_modulation_mode in (
                    "none", "original_stage_two", "paper_stage_two",
                    "reliability_weighted_stage_two"
                ) else "direct unweighted cross_norm from epoch one"
            ),
            "stage_two_scale_estimator": (
                "disabled"
                if representation_modulation_mode == "none"
                else (
                    "frozen retained-edge-reliability-weighted cross-type RMS"
                    if representation_modulation_mode in (
                        "reliability_weighted_always",
                        "reliability_weighted_stage_two",
                    )
                    else "unweighted cross-type RMS"
                )
            ),
            "scale_definition": (
                "sqrt(weighted_mean(node_embedding_squared_l2_norm) + 1e-6); "
                "uncapped and applied directly after every propagation layer, "
                "matching the released cross_norm implementation"
            ),
            "trace": representation_modulation_trace,
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "training_summary.json"), report)
    return report
