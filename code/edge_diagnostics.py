"""Side-channel edge diagnostics for the current NR-GCF implementation.

This module never changes model parameters, optimizer state, edge scores, or
filtering decisions.  It records already-computed per-edge losses and exports
features after the current code has made its filtering decision.

The production structural path uses PyTorch sparse matrix multiplication.  A
small dependency-free reference implementation is included for CPU unit tests
and for documenting the exact leave-one-edge-out feature definition.
"""

from __future__ import print_function

import csv
import gzip
import json
import logging
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime

try:
    import torch
except ImportError:  # Allows schema/reference tests in lightweight environments.
    torch = None


def _no_grad():
    if torch is None:
        return lambda function: function
    return torch.no_grad()


SCHEMA_VERSION = "nrgcf_edge_diagnostics_v3"
MINHASH_DIM = 128
MINHASH_PRIME = 2147483647
MINHASH_BUILD_DIM_CHUNK = 8
QUANTILE_LOW = 0.20
QUANTILE_HIGH = 0.80


FIELD_SPECS = [
    ("edge_id", "int64", False, "Stable original column position in dataset.train_edge_index."),
    ("user_id_internal", "int64", False, "User ID used directly by NR-GCF."),
    ("item_id_internal", "int64", False, "Item ID used directly by NR-GCF."),
    ("user_id_raw", "string", True, "Original user ID loaded from user_list.txt when the mapping is available."),
    ("item_id_raw", "string", True, "Original item ID loaded from item_list.txt when the mapping is available."),
    ("edge_position_in_training_graph", "int64", False, "Stable original training-edge column position."),
    ("is_original_observed_edge", "bool", True, "Optional validated sidecar label; null when no label file is supplied."),
    ("synthetic_is_noisy", "bool", True, "Optional post-export evaluation label; never used by feature computation or filtering."),
    ("synthetic_noise_type", "string", True, "Optional validated synthetic-noise protocol name."),
    ("train_split_identifier", "string", False, "Dataset train split identifier."),
    ("graph_version_identifier", "string", False, "Graph version used for edge identity and structural features."),
    ("raw_edge_loss_at_filter_epoch", "float64", True, "Null: current code does not compute per-edge loss at epoch 15."),
    ("current_edge_loss", "float64", True, "Last observed per-edge instance BPR loss (normally epoch 14)."),
    ("historical_or_momentum_loss", "float64", False, "Raw runtime model.momentum_loss before min-max normalization."),
    ("normalized_edge_score", "float64", True, "Min-max normalized raw runtime momentum value before threshold zeroing."),
    ("threshold_value", "float64", False, "Current code threshold beta; not the paper's adaptive threshold."),
    ("score_minus_threshold", "float64", True, "normalized_edge_score minus current code beta."),
    ("nr_gcf_removed", "bool", False, "Exact current-code decision: not(post-threshold score > 0)."),
    ("nr_gcf_retained", "bool", False, "Exact current-code decision: post-threshold score > 0."),
    ("filtering_epoch", "int64", False, "Epoch at which the current entry filters edges."),
    ("warmup_epoch_count", "int64", False, "Number of epochs that can update per-edge momentum in current loop."),
    ("edge_seen_count", "int64", False, "Number of already-computed per-edge losses observed by diagnostics."),
    ("loss_first_observed", "float64", True, "First observed per-edge instance loss."),
    ("loss_last_observed", "float64", True, "Last observed per-edge instance loss."),
    ("loss_mean", "float64", True, "Mean of observed per-edge instance losses."),
    ("loss_std", "float64", True, "Population standard deviation of observed per-edge instance losses."),
    ("user_degree_before", "int64", False, "Edge-occurrence degree in the pre-filter training graph."),
    ("item_degree_before", "int64", False, "Edge-occurrence degree in the pre-filter training graph."),
    ("user_degree_after_if_removed", "int64", False, "user_degree_before - 1 for this edge occurrence."),
    ("item_degree_after_if_removed", "int64", False, "item_degree_before - 1 for this edge occurrence."),
    ("user_becomes_isolated_if_removed", "bool", False, "True when user degree would become zero."),
    ("item_becomes_isolated_if_removed", "bool", False, "True when item degree would become zero."),
    ("user_below_min_degree_if_removed", "bool", False, "True when post-removal user degree is below configured minimum."),
    ("item_below_min_degree_if_removed", "bool", False, "True when post-removal item degree is below configured minimum."),
    ("min_endpoint_degree", "int64", False, "Minimum of pre-filter user and item degree."),
    ("max_endpoint_degree", "int64", False, "Maximum of pre-filter user and item degree."),
    ("inverse_user_degree", "float64", False, "Fraction of the user's edge-occurrence degree represented by this edge."),
    ("inverse_item_degree", "float64", False, "Fraction of the item's edge-occurrence degree represented by this edge."),
    ("normalized_degree_product", "float64", False, "1/sqrt(user_degree_before * item_degree_before)."),
    ("user_side_structure_mean", "float64", True, "Mean LOO degree-normalized item-item overlap estimated by MinHash over deterministic bounded neighbors."),
    ("user_side_structure_max", "float64", True, "Maximum estimated LOO item-item overlap over deterministic bounded neighbors."),
    ("user_side_structure_topk_mean", "float64", True, "Top-k mean estimated LOO item-item overlap over deterministic bounded neighbors."),
    ("user_side_valid_neighbor_count", "int64", False, "All valid other item neighbors available on the user side."),
    ("user_side_sampled_neighbor_count", "int64", False, "Deterministic bounded item neighbors actually evaluated."),
    ("item_side_structure_mean", "float64", True, "Mean LOO degree-normalized user-user overlap estimated by MinHash over deterministic bounded neighbors."),
    ("item_side_structure_max", "float64", True, "Maximum estimated LOO user-user overlap over deterministic bounded neighbors."),
    ("item_side_structure_topk_mean", "float64", True, "Top-k mean estimated LOO user-user overlap over deterministic bounded neighbors."),
    ("item_side_valid_neighbor_count", "int64", False, "All valid other user neighbors available on the item side."),
    ("item_side_sampled_neighbor_count", "int64", False, "Deterministic bounded user neighbors actually evaluated."),
    ("bilateral_structure_min", "float64", True, "Minimum of the two side means when both are available."),
    ("bilateral_structure_max", "float64", True, "Maximum of the two side means when both are available."),
    ("bilateral_structure_mean", "float64", True, "Arithmetic mean of the two side means when both are available."),
    ("bilateral_structure_geometric_mean", "float64", True, "Geometric mean of non-negative side means."),
    ("bilateral_disagreement", "float64", True, "Absolute difference between side means."),
    ("high_loss_high_structure_flag", "bool", True, "Diagnostic-only flag using saved train-set quantile thresholds."),
    ("high_loss_low_structure_flag", "bool", True, "Diagnostic-only flag using saved train-set quantile thresholds."),
    ("low_loss_low_structure_flag", "bool", True, "Diagnostic-only flag using saved train-set quantile thresholds."),
    ("removed_but_high_structure_flag", "bool", True, "Removed by current code and above saved high-structure threshold."),
    ("retained_but_low_structure_flag", "bool", True, "Retained by current code and below saved low-structure threshold."),
]

FIELD_NAMES = [spec[0] for spec in FIELD_SPECS]
FIELD_TYPES = dict((spec[0], spec[1]) for spec in FIELD_SPECS)


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for NR-GCF tensor diagnostics. "
            "The schema and dependency-free reference tests remain available."
        )


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return dict((str(k), _json_safe(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_raw_id_mapping(path, expected_count):
    """Load an explicit org_id/remap_id mapping without inferring identities."""
    if not os.path.exists(path):
        return None, "mapping file not found"
    mapping = [None] * int(expected_count)
    try:
        with open(path, encoding="utf-8") as stream:
            header = stream.readline().strip().split()
            if "org_id" not in header or "remap_id" not in header:
                return None, "mapping header lacks org_id/remap_id"
            raw_column = header.index("org_id")
            internal_column = header.index("remap_id")
            for line_number, line in enumerate(stream, start=2):
                values = line.rstrip("\n").split()
                if not values:
                    continue
                if max(raw_column, internal_column) >= len(values):
                    return None, "malformed mapping row %d" % line_number
                internal_id = int(values[internal_column])
                if internal_id < 0 or internal_id >= len(mapping):
                    return None, "mapping ID out of range at row %d" % line_number
                if mapping[internal_id] is not None:
                    return None, "duplicate remap_id at row %d" % line_number
                mapping[internal_id] = values[raw_column]
    except (OSError, UnicodeError, ValueError) as exc:
        return None, "mapping read failed: %s" % exc
    missing = sum(value is None for value in mapping)
    if missing:
        return mapping, "partial mapping: %d IDs missing" % missing
    return mapping, None


def diagnostics_schema():
    return {
        "diagnostics_schema_version": SCHEMA_VERSION,
        "fields": [
            {
                "name": name,
                "dtype": dtype,
                "nullable": nullable,
                "description": description,
            }
            for name, dtype, nullable, description in FIELD_SPECS
        ],
        "structural_approximation": {
            "estimator": "MinHash estimates leave-one-edge-out Jaccard overlap; observed degrees transform it to cosine-normalized co-occurrence.",
            "mean_max_and_topk": "Computed over a deterministic bounded edge-ID-hash representative set, not all neighbor pairs.",
            "target_edge_exclusion": "Per-hash first/second minima remove the target endpoint exactly from the candidate neighborhood; the target edge is also excluded from representatives.",
            "degree_bias_control": "MinHash match probability is Jaccard similarity and does not grow merely because a node has high degree.",
            "remaining_approximation": "Finite MinHash dimensions and bounded neighbor sampling introduce sampling variance; sampled-neighbor counts are exported.",
        },
    }


class EdgeLossHistory(object):
    """CPU accumulators updated only from already-computed detached losses."""

    def __init__(self, edge_count):
        _require_torch()
        self.edge_count = int(edge_count)
        self.seen = torch.zeros(self.edge_count, dtype=torch.int32)
        self.first = torch.full((self.edge_count,), float("nan"), dtype=torch.float32)
        self.last = torch.full((self.edge_count,), float("nan"), dtype=torch.float32)
        self.total = torch.zeros(self.edge_count, dtype=torch.float64)
        self.total_sq = torch.zeros(self.edge_count, dtype=torch.float64)

    @_no_grad()
    def observe(self, edge_ids, losses):
        edge_ids_cpu = edge_ids.detach().to(device="cpu", dtype=torch.long)
        losses_cpu = losses.detach().to(device="cpu", dtype=torch.float32)
        first_mask = self.seen[edge_ids_cpu] == 0
        if bool(first_mask.any()):
            self.first[edge_ids_cpu[first_mask]] = losses_cpu[first_mask]
        self.last[edge_ids_cpu] = losses_cpu
        self.total[edge_ids_cpu] += losses_cpu.to(torch.float64)
        self.total_sq[edge_ids_cpu] += losses_cpu.to(torch.float64).pow(2)
        self.seen[edge_ids_cpu] += 1

    def chunk(self, start, end):
        seen = self.seen[start:end].to(torch.int64)
        valid = seen > 0
        denominator = seen.clamp(min=1).to(torch.float64)
        mean = self.total[start:end] / denominator
        variance = self.total_sq[start:end] / denominator - mean.pow(2)
        std = variance.clamp(min=0).sqrt()
        nan = torch.full_like(mean, float("nan"))
        return {
            "edge_seen_count": seen,
            "loss_first_observed": self.first[start:end].to(torch.float64),
            "loss_last_observed": self.last[start:end].to(torch.float64),
            "loss_mean": torch.where(valid, mean, nan),
            "loss_std": torch.where(valid, std, nan),
            "current_edge_loss": self.last[start:end].to(torch.float64),
        }


def _stable_representative_edges(group_ids, num_groups, support_limit):
    """Select bounded edge occurrences per node without touching RNG state."""
    edge_count = int(group_ids.numel())
    edge_ids = torch.arange(edge_count, device=group_ids.device, dtype=torch.long)
    # The key groups by node then orders by a deterministic edge-ID hash.
    hash_mod = 2147483647
    hashed = torch.remainder(edge_ids * 1103515245 + 12345, hash_mod)
    key = group_ids.to(torch.long) * hash_mod + hashed
    order = torch.argsort(key)
    sorted_groups = group_ids[order]
    degree = torch.bincount(sorted_groups, minlength=int(num_groups))
    starts = torch.cumsum(degree, dim=0) - degree
    repeated_starts = torch.repeat_interleave(starts, degree)
    within_group_rank = torch.arange(edge_count, device=group_ids.device) - repeated_starts
    keep = within_group_rank < int(support_limit)
    representatives = torch.full(
        (int(num_groups), int(support_limit)),
        -1,
        dtype=torch.long,
        device=group_ids.device,
    )
    representatives[sorted_groups[keep], within_group_rank[keep]] = order[keep]
    return representatives


def _splitmix64(value):
    """Small deterministic mixer used only to derive hash coefficients."""
    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _minhash_values(node_ids, dimension_start, dimension_end, salt):
    """Deterministic injective affine hashes for the requested dimensions."""
    coefficients = []
    offsets = []
    for dimension in range(int(dimension_start), int(dimension_end)):
        base = (int(salt) << 32) + dimension
        coefficients.append(_splitmix64(base) % (MINHASH_PRIME - 1) + 1)
        offsets.append(_splitmix64(base ^ 0xD1B54A32D192ED03) % MINHASH_PRIME)
    coefficients = torch.tensor(
        coefficients, device=node_ids.device, dtype=torch.int64
    )
    offsets = torch.tensor(offsets, device=node_ids.device, dtype=torch.int64)
    values = torch.remainder(
        node_ids.to(torch.int64).unsqueeze(1) * coefficients.unsqueeze(0)
        + offsets.unsqueeze(0),
        MINHASH_PRIME,
    )
    return values.to(torch.int32)


def _neighbor_minhash(source_ids, group_ids, num_groups, salt):
    """Return first and second MinHash values for every bipartite node."""
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "two_hop_minhash requires torch.Tensor.scatter_reduce_ "
            "(available in the server's PyTorch 2.x runtime)"
        )
    device = source_ids.device
    first = torch.full(
        (int(num_groups), MINHASH_DIM),
        MINHASH_PRIME,
        device=device,
        dtype=torch.int32,
    )
    second = torch.full_like(first, MINHASH_PRIME)
    for start in range(0, MINHASH_DIM, MINHASH_BUILD_DIM_CHUNK):
        end = min(MINHASH_DIM, start + MINHASH_BUILD_DIM_CHUNK)
        edge_hashes = _minhash_values(source_ids, start, end, salt)
        group_index = group_ids.to(torch.long).unsqueeze(1).expand_as(edge_hashes)
        block_first = torch.full(
            (int(num_groups), end - start),
            MINHASH_PRIME,
            device=device,
            dtype=torch.int32,
        )
        block_first.scatter_reduce_(
            0, group_index, edge_hashes, reduce="amin", include_self=True
        )
        is_first = edge_hashes == block_first[group_ids]
        first_count = torch.zeros_like(block_first)
        first_count.scatter_add_(0, group_index, is_first.to(torch.int32))
        without_first = torch.where(
            is_first,
            torch.full_like(edge_hashes, MINHASH_PRIME),
            edge_hashes,
        )
        block_second = torch.full_like(block_first, MINHASH_PRIME)
        block_second.scatter_reduce_(
            0, group_index, without_first, reduce="amin", include_self=True
        )
        # Duplicate coordinates or an extremely unlikely equal hash leave the
        # first minimum present after removing one occurrence.
        block_second = torch.where(first_count > 1, block_first, block_second)
        first[:, start:end] = block_first
        second[:, start:end] = block_second
    return first, second


class TwoHopMinHash(object):
    """Bounded, degree-calibrated two-hop structural diagnostics.

    Each node stores MinHash signatures of its one-hop neighbor set.  For a
    target edge, first/second minima analytically remove the opposite endpoint
    from the candidate signature.  Similarity to a deterministic bounded set
    of same-type neighbors is estimated as Jaccard and transformed to the
    cosine-normalized co-occurrence implied by the two observed degrees.
    """

    def __init__(self, edge_index, num_users, num_items, topk, structural_mode):
        _require_torch()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        self.edge_index = edge_index.detach()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.edge_count = int(edge_index.size(1))
        self.topk = int(topk)
        self.mode = structural_mode
        self.device = edge_index.device
        self.user_degree = torch.bincount(
            self.edge_index[0], minlength=self.num_users
        ).to(torch.float32)
        self.item_degree = torch.bincount(
            self.edge_index[1], minlength=self.num_items
        ).to(torch.float32)
        self.support_limit = max(self.topk + 1, 16)

        self.enabled = structural_mode == "two_hop_minhash"
        if structural_mode not in ("two_hop_minhash", "none"):
            raise ValueError("Unsupported structural mode: %s" % structural_mode)
        if not self.enabled:
            return

        values = torch.ones(self.edge_count, device=self.device, dtype=torch.float32)
        interaction = torch.sparse_coo_tensor(
            self.edge_index,
            values,
            size=(self.num_users, self.num_items),
            device=self.device,
        ).coalesce()
        # Structural overlap uses the coalesced simple graph. Edge identity and
        # connectivity fields continue to use the original edge occurrences.
        self.structural_edge_index = interaction.indices()
        structural_users = self.structural_edge_index[0]
        structural_items = self.structural_edge_index[1]
        self.structural_user_degree = torch.bincount(
            structural_users, minlength=self.num_users
        ).to(torch.float32)
        self.structural_item_degree = torch.bincount(
            structural_items, minlength=self.num_items
        ).to(torch.float32)
        self.item_first, self.item_second = _neighbor_minhash(
            structural_users,
            structural_items,
            self.num_items,
            salt=17,
        )
        self.user_first, self.user_second = _neighbor_minhash(
            structural_items,
            structural_users,
            self.num_users,
            salt=53,
        )
        self.user_representative_edges = _stable_representative_edges(
            structural_users, self.num_users, self.support_limit
        )
        self.item_representative_edges = _stable_representative_edges(
            structural_items, self.num_items, self.support_limit
        )

    def _loo_signature(self, first, second, node_ids, removed_neighbor_ids, salt):
        full = first[node_ids]
        removed_hash = _minhash_values(
            removed_neighbor_ids, 0, MINHASH_DIM, salt
        )
        return torch.where(full == removed_hash, second[node_ids], full)

    def _bounded_statistics(
        self,
        candidate_signature,
        candidate_degree,
        representatives,
        target_neighbor_ids,
        neighbor_node_row,
        neighbor_signatures,
        neighbor_degree,
        candidate_valid,
    ):
        valid = representatives >= 0
        safe_edges = representatives.clamp(min=0)
        neighbor_ids = self.structural_edge_index[neighbor_node_row][safe_edges]
        valid = valid & (neighbor_ids != target_neighbor_ids.unsqueeze(1))
        valid = valid & candidate_valid.unsqueeze(1)
        neighbor_degrees = neighbor_degree[neighbor_ids]
        valid = valid & (neighbor_degrees > 0)

        signatures = neighbor_signatures[neighbor_ids]
        jaccard = (
            candidate_signature.unsqueeze(1) == signatures
        ).to(torch.float32).mean(dim=2)
        candidate_degrees = candidate_degree.unsqueeze(1).to(torch.float32)
        intersection = (
            jaccard * (candidate_degrees + neighbor_degrees)
            / (1.0 + jaccard)
        )
        intersection = torch.minimum(
            intersection, torch.minimum(candidate_degrees, neighbor_degrees)
        )
        similarities = intersection / torch.sqrt(
            candidate_degrees.clamp(min=1.0) * neighbor_degrees.clamp(min=1.0)
        )
        similarities = similarities.clamp(min=0.0, max=1.0)

        sampled_count = valid.sum(dim=1)
        finite_values = torch.where(valid, similarities, torch.zeros_like(similarities))
        mean = finite_values.sum(dim=1) / sampled_count.clamp(min=1)
        nan = torch.full_like(mean, float("nan"))
        mean = torch.where(sampled_count > 0, mean, nan)

        masked = similarities.masked_fill(~valid, float("-inf"))
        maximum = masked.max(dim=1).values
        maximum = torch.where(sampled_count > 0, maximum, nan)
        k = min(self.topk, self.support_limit)
        top_values = torch.topk(masked, k=k, dim=1).values
        finite = torch.isfinite(top_values)
        top_count = finite.sum(dim=1)
        top_sum = torch.where(finite, top_values, torch.zeros_like(top_values)).sum(dim=1)
        top_mean = top_sum / top_count.clamp(min=1)
        top_mean = torch.where(top_count > 0, top_mean, nan)
        return mean, maximum, top_mean, sampled_count

    @_no_grad()
    def compute_chunk(self, start, end):
        size = int(end - start)
        if not self.enabled:
            nan = torch.full((size,), float("nan"), dtype=torch.float64)
            zero = torch.zeros(size, dtype=torch.int64)
            return {
                "user_side_structure_mean": nan.clone(),
                "user_side_structure_max": nan.clone(),
                "user_side_structure_topk_mean": nan.clone(),
                "user_side_valid_neighbor_count": zero.clone(),
                "user_side_sampled_neighbor_count": zero.clone(),
                "item_side_structure_mean": nan.clone(),
                "item_side_structure_max": nan.clone(),
                "item_side_structure_topk_mean": nan.clone(),
                "item_side_valid_neighbor_count": zero.clone(),
                "item_side_sampled_neighbor_count": zero.clone(),
            }

        users = self.edge_index[0, start:end]
        items = self.edge_index[1, start:end]
        user_degree = self.structural_user_degree[users]
        item_degree = self.structural_item_degree[items]
        user_side_valid = (user_degree > 1) & (item_degree > 1)
        item_side_valid = user_side_valid

        candidate_item = self._loo_signature(
            self.item_first, self.item_second, items, users, salt=17
        )
        user_mean, user_max, user_topk, user_sampled = self._bounded_statistics(
            candidate_signature=candidate_item,
            candidate_degree=item_degree - 1,
            representatives=self.user_representative_edges[users],
            target_neighbor_ids=items,
            neighbor_node_row=1,
            neighbor_signatures=self.item_first,
            neighbor_degree=self.structural_item_degree,
            candidate_valid=user_side_valid,
        )
        del candidate_item

        candidate_user = self._loo_signature(
            self.user_first, self.user_second, users, items, salt=53
        )
        item_mean, item_max, item_topk, item_sampled = self._bounded_statistics(
            candidate_signature=candidate_user,
            candidate_degree=user_degree - 1,
            representatives=self.item_representative_edges[items],
            target_neighbor_ids=users,
            neighbor_node_row=0,
            neighbor_signatures=self.user_first,
            neighbor_degree=self.structural_user_degree,
            candidate_valid=item_side_valid,
        )
        result = {
            "user_side_structure_mean": user_mean,
            "user_side_structure_max": user_max,
            "user_side_structure_topk_mean": user_topk,
            "user_side_valid_neighbor_count": torch.where(
                user_side_valid,
                (user_degree - 1).to(torch.int64),
                torch.zeros_like(users),
            ),
            "user_side_sampled_neighbor_count": user_sampled.to(torch.int64),
            "item_side_structure_mean": item_mean,
            "item_side_structure_max": item_max,
            "item_side_structure_topk_mean": item_topk,
            "item_side_valid_neighbor_count": torch.where(
                item_side_valid,
                (item_degree - 1).to(torch.int64),
                torch.zeros_like(items),
            ),
            "item_side_sampled_neighbor_count": item_sampled.to(torch.int64),
        }
        return dict((name, value.detach().cpu()) for name, value in result.items())


def _finite_quantile(values, q):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(torch.quantile(finite.to(torch.float64), float(q)).item())


def _to_python_list(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _to_arrow_values(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


class PartWriter(object):
    """Write chunks into one streaming artifact instead of one file per chunk."""

    def __init__(self, output_dir, requested_format, logger):
        self.output_dir = output_dir
        self.requested_format = requested_format
        self.actual_format = requested_format
        self.fallback_reason = None
        self.logger = logger
        self.part_index = 0
        self.pa = None
        self.pq = None
        self.parquet_writer = None
        self.csv_stream = None
        self.csv_writer = None
        self.path = None
        self.parquet_compression = None
        if requested_format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                self.pa = pa
                self.pq = pq
                codec = getattr(pa, "Codec", None)
                if codec is not None and codec.is_available("zstd"):
                    self.parquet_compression = "zstd"
                elif codec is not None and codec.is_available("snappy"):
                    self.parquet_compression = "snappy"
            except Exception as exc:
                self.actual_format = "csv_gzip"
                self.fallback_reason = "pyarrow unavailable: %s" % exc
                self.logger.warning(
                    "Parquet requested but pyarrow is unavailable; falling back to one gzip CSV."
                )

        if self.actual_format == "parquet":
            self.path = os.path.join(self.output_dir, "edge_diagnostics.parquet")
        elif self.actual_format == "csv_gzip":
            self.path = os.path.join(self.output_dir, "edge_diagnostics.csv.gz")
            self.csv_stream = gzip.open(
                self.path, "wt", newline="", encoding="utf-8", compresslevel=6
            )
            self.csv_writer = csv.writer(self.csv_stream)
            self.csv_writer.writerow(FIELD_NAMES)
        else:
            self.path = os.path.join(self.output_dir, "edge_diagnostics.csv")
            self.csv_stream = open(
                self.path, "w", newline="", encoding="utf-8"
            )
            self.csv_writer = csv.writer(self.csv_stream)
            self.csv_writer.writerow(FIELD_NAMES)

    def _arrow_type(self, dtype):
        if dtype == "int64":
            return self.pa.int64()
        if dtype == "float64":
            return self.pa.float64()
        if dtype == "bool":
            return self.pa.bool_()
        return self.pa.string()

    def write(self, columns):
        if self.actual_format == "parquet":
            arrays = []
            for name in FIELD_NAMES:
                values = _to_arrow_values(columns[name])
                arrays.append(self.pa.array(values, type=self._arrow_type(FIELD_TYPES[name])))
            table = self.pa.Table.from_arrays(arrays, names=FIELD_NAMES)
            if self.parquet_writer is None:
                self.parquet_writer = self.pq.ParquetWriter(
                    self.path, table.schema, compression=self.parquet_compression
                )
            self.parquet_writer.write_table(table)
        else:
            python_columns = dict(
                (name, _to_python_list(columns[name])) for name in FIELD_NAMES
            )
            row_count = len(python_columns[FIELD_NAMES[0]])
            for row_index in range(row_count):
                self.csv_writer.writerow(
                    [python_columns[name][row_index] for name in FIELD_NAMES]
                )
        self.part_index += 1
        return self.path

    def close(self):
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
        if self.csv_stream is not None:
            self.csv_stream.close()
            self.csv_stream = None


def _parse_nullable_bool(value, field_name, edge_id):
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ValueError(
        "Invalid %s=%r for synthetic label edge_id=%d"
        % (field_name, value, edge_id)
    )


class SyntheticLabelReader(object):
    """Stream and validate a label sidecar without exposing it to features."""

    REQUIRED_FIELDS = {
        "edge_id",
        "user_id_internal",
        "item_id_internal",
        "is_original_observed_edge",
        "synthetic_is_noisy",
        "synthetic_noise_type",
    }

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.stream = open(self.path, newline="", encoding="utf-8")
        self.reader = csv.DictReader(self.stream)
        missing = self.REQUIRED_FIELDS - set(self.reader.fieldnames or [])
        if missing:
            self.close()
            raise ValueError("Synthetic label CSV is missing fields: %s" % sorted(missing))
        self.clean_count = 0
        self.noisy_count = 0
        self.clean_removed_count = 0
        self.noisy_removed_count = 0

    def read_chunk(self, start, end, users, items, removed):
        original = []
        noisy = []
        noise_type = []
        for offset, expected_edge_id in enumerate(range(int(start), int(end))):
            try:
                row = next(self.reader)
            except StopIteration:
                raise ValueError(
                    "Synthetic label CSV ended before edge_id=%d" % expected_edge_id
                )
            edge_id = int(row["edge_id"])
            user = int(row["user_id_internal"])
            item = int(row["item_id_internal"])
            if edge_id != expected_edge_id:
                raise ValueError(
                    "Synthetic label edge_id mismatch: expected=%d actual=%d"
                    % (expected_edge_id, edge_id)
                )
            if user != int(users[offset]) or item != int(items[offset]):
                raise ValueError(
                    "Synthetic label endpoint mismatch at edge_id=%d" % edge_id
                )
            is_original = _parse_nullable_bool(
                row["is_original_observed_edge"],
                "is_original_observed_edge",
                edge_id,
            )
            is_noisy = _parse_nullable_bool(
                row["synthetic_is_noisy"], "synthetic_is_noisy", edge_id
            )
            if is_original is not None and is_noisy is not None:
                if is_original == is_noisy:
                    raise ValueError(
                        "Synthetic clean/noisy labels are inconsistent at edge_id=%d"
                        % edge_id
                    )
                if is_noisy:
                    self.noisy_count += 1
                    self.noisy_removed_count += int(bool(removed[offset]))
                else:
                    self.clean_count += 1
                    self.clean_removed_count += int(bool(removed[offset]))
            original.append(is_original)
            noisy.append(is_noisy)
            noise_type.append(row["synthetic_noise_type"] or None)
        return {
            "is_original_observed_edge": original,
            "synthetic_is_noisy": noisy,
            "synthetic_noise_type": noise_type,
        }

    def verify_complete(self):
        extra = next(self.reader, None)
        if extra is not None:
            raise ValueError(
                "Synthetic label CSV contains extra edge_id=%s" % extra.get("edge_id")
            )

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class RunningStats(object):
    def __init__(self, sample_stride, sample_limit=50000):
        self.sample_stride = max(1, int(sample_stride))
        self.sample_limit = int(sample_limit)
        self.count = 0
        self.missing_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = None
        self.maximum = None
        self.sample = []

    def update(self, values, edge_ids):
        if torch is not None and torch.is_tensor(values):
            tensor = values.detach().cpu()
            if tensor.dtype == torch.bool:
                tensor = tensor.to(torch.float64)
            else:
                tensor = tensor.to(torch.float64)
            finite = torch.isfinite(tensor)
            self.nan_count += int(torch.isnan(tensor).sum().item())
            self.inf_count += int(torch.isinf(tensor).sum().item())
            finite_values = tensor[finite]
            if finite_values.numel() > 0:
                self.count += int(finite_values.numel())
                self.total += float(finite_values.sum().item())
                self.total_sq += float(finite_values.pow(2).sum().item())
                local_min = float(finite_values.min().item())
                local_max = float(finite_values.max().item())
                self.minimum = local_min if self.minimum is None else min(self.minimum, local_min)
                self.maximum = local_max if self.maximum is None else max(self.maximum, local_max)
            if len(self.sample) < self.sample_limit:
                ids = edge_ids.detach().cpu()
                sample_mask = finite & (torch.remainder(ids, self.sample_stride) == 0)
                remaining = self.sample_limit - len(self.sample)
                self.sample.extend(tensor[sample_mask][:remaining].tolist())
            return

        for value, edge_id in zip(values, edge_ids):
            if value is None:
                self.missing_count += 1
                continue
            numeric = float(value)
            if math.isnan(numeric):
                self.nan_count += 1
                continue
            if math.isinf(numeric):
                self.inf_count += 1
                continue
            self.count += 1
            self.total += numeric
            self.total_sq += numeric * numeric
            self.minimum = numeric if self.minimum is None else min(self.minimum, numeric)
            self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)
            if edge_id % self.sample_stride == 0 and len(self.sample) < self.sample_limit:
                self.sample.append(numeric)

    def result(self):
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "median": None,
                "max": None,
                "missing_count": self.missing_count,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        ordered = sorted(self.sample)
        if not ordered:
            median = None
        elif len(ordered) % 2:
            median = ordered[len(ordered) // 2]
        else:
            middle = len(ordered) // 2
            median = 0.5 * (ordered[middle - 1] + ordered[middle])
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "median": median,
            "max": self.maximum,
            "missing_count": self.missing_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


SUMMARY_GROUP_FIELDS = [
    "normalized_edge_score",
    "historical_or_momentum_loss",
    "current_edge_loss",
    "user_degree_before",
    "item_degree_before",
    "min_endpoint_degree",
    "normalized_degree_product",
    "user_side_structure_mean",
    "item_side_structure_mean",
    "bilateral_structure_mean",
    "bilateral_disagreement",
]


class SummaryBuilder(object):
    def __init__(self, edge_count):
        stride = max(1, int(math.ceil(float(max(edge_count, 1)) / 50000.0)))
        numeric_names = [
            name for name, dtype, _, _ in FIELD_SPECS
            if dtype in ("int64", "float64", "bool")
        ]
        self.overall = dict((name, RunningStats(stride)) for name in numeric_names)
        self.grouped = defaultdict(dict)
        self.stride = stride
        self.synthetic_clean_count = 0
        self.synthetic_noisy_count = 0
        self.synthetic_clean_removed_count = 0
        self.synthetic_noisy_removed_count = 0

    def _group_stats(self, group_name):
        if not self.grouped[group_name]:
            self.grouped[group_name] = dict(
                (name, RunningStats(self.stride)) for name in SUMMARY_GROUP_FIELDS
            )
        return self.grouped[group_name]

    def update(self, columns):
        edge_ids = columns["edge_id"]
        for name, stats in self.overall.items():
            stats.update(columns[name], edge_ids)

        retained = columns["nr_gcf_retained"]
        removed = columns["nr_gcf_removed"]
        min_degree = columns["min_endpoint_degree"]
        groups = {
            "retained": retained,
            "removed": removed,
            "degree_1": min_degree == 1,
            "degree_2_5": (min_degree >= 2) & (min_degree <= 5),
            "degree_6_20": (min_degree >= 6) & (min_degree <= 20),
            "degree_21_100": (min_degree >= 21) & (min_degree <= 100),
            "degree_gt_100": min_degree > 100,
        }
        synthetic_labels = columns.get("synthetic_is_noisy")
        if synthetic_labels and all(value is not None for value in synthetic_labels):
            noisy_mask = torch.tensor(synthetic_labels, dtype=torch.bool)
            clean_mask = ~noisy_mask
            groups["synthetic_clean"] = clean_mask
            groups["synthetic_noisy"] = noisy_mask
            self.synthetic_clean_count += int(clean_mask.sum().item())
            self.synthetic_noisy_count += int(noisy_mask.sum().item())
            self.synthetic_clean_removed_count += int(
                (clean_mask & removed).sum().item()
            )
            self.synthetic_noisy_removed_count += int(
                (noisy_mask & removed).sum().item()
            )
        for group_name, mask in groups.items():
            group_edge_ids = edge_ids[mask]
            stats_by_name = self._group_stats(group_name)
            for name, stats in stats_by_name.items():
                stats.update(columns[name][mask], group_edge_ids)

    def result(self):
        return {
            "median_method": "deterministic edge-ID stride sample (bounded to 50,000 values per field)",
            "field_statistics": dict((name, stats.result()) for name, stats in self.overall.items()),
            "removed_retained_and_degree_bucket_statistics": {
                group: dict((name, stats.result()) for name, stats in fields.items())
                for group, fields in self.grouped.items()
            },
        }


class DiagnosticsInvarianceGuard(object):
    """Exact optional guard around exporter execution."""

    def __init__(self, model, named_tensors):
        _require_torch()
        self.model = model
        self.model_edge_index_object_id = id(getattr(model, "edge_index", None))
        self.tensor_snapshots = dict(
            (name, tensor.detach().clone()) for name, tensor in named_tensors.items()
        )
        self.tensors = named_tensors
        self.parameter_snapshots = [parameter.detach().clone() for parameter in model.parameters()]
        self.python_rng_state = random.getstate()
        self.numpy_module = None
        self.numpy_rng_state = None
        try:
            import numpy as np
            self.numpy_module = np
            self.numpy_rng_state = np.random.get_state()
        except ImportError:
            pass
        self.cpu_rng_state = torch.get_rng_state().clone()
        self.cuda_rng_states = None
        if torch.cuda.is_available():
            self.cuda_rng_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    def verify(self):
        def exact_equal(before, after):
            if before.dtype.is_floating_point:
                return bool(torch.allclose(
                    before, after, rtol=0.0, atol=0.0, equal_nan=True
                ))
            return bool(torch.equal(before, after))

        tensor_results = dict(
            (name, exact_equal(self.tensor_snapshots[name], tensor.detach()))
            for name, tensor in self.tensors.items()
        )
        parameter_unchanged = all(
            exact_equal(snapshot, parameter.detach())
            for snapshot, parameter in zip(self.parameter_snapshots, self.model.parameters())
        )
        model_edge_index_identity_unchanged = (
            self.model_edge_index_object_id == id(getattr(self.model, "edge_index", None))
        )
        python_rng_unchanged = self.python_rng_state == random.getstate()
        numpy_rng_unchanged = True
        if self.numpy_module is not None:
            current_numpy_state = self.numpy_module.random.get_state()
            numpy_rng_unchanged = (
                self.numpy_rng_state[0] == current_numpy_state[0]
                and bool((self.numpy_rng_state[1] == current_numpy_state[1]).all())
                and self.numpy_rng_state[2:] == current_numpy_state[2:]
            )
        cpu_rng_unchanged = bool(torch.equal(self.cpu_rng_state, torch.get_rng_state()))
        cuda_rng_unchanged = True
        if self.cuda_rng_states is not None:
            current = torch.cuda.get_rng_state_all()
            cuda_rng_unchanged = len(current) == len(self.cuda_rng_states) and all(
                torch.equal(before, after)
                for before, after in zip(self.cuda_rng_states, current)
            )
        result = {
            "tensor_unchanged": tensor_results,
            "model_parameters_unchanged": parameter_unchanged,
            "model_edge_index_object_identity_unchanged": model_edge_index_identity_unchanged,
            "python_rng_state_unchanged": python_rng_unchanged,
            "numpy_rng_state_unchanged": numpy_rng_unchanged,
            "cpu_rng_state_unchanged": cpu_rng_unchanged,
            "cuda_rng_state_unchanged": cuda_rng_unchanged,
        }
        result["passed"] = (
            all(tensor_results.values())
            and parameter_unchanged
            and model_edge_index_identity_unchanged
            and python_rng_unchanged
            and numpy_rng_unchanged
            and cpu_rng_unchanged
            and cuda_rng_unchanged
        )
        return result


def _git_info(repo_dir):
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8").strip()
        # Compare tracked content only. Run logs and diagnostics parts are
        # intentionally untracked artifacts and must not create a false dirty
        # provenance flag merely because they were written inside the repo.
        dirty = subprocess.call(
            ["git", "-C", repo_dir, "diff", "--quiet", "HEAD", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


class EdgeDiagnosticsExporter(object):
    def __init__(self, args, model_config, output_dir, repo_dir):
        _require_torch()
        self.args = args
        self.model_config = model_config
        self.output_dir = os.path.abspath(output_dir)
        self.repo_dir = repo_dir
        self.code_commit_hash, self.code_tracked_worktree_dirty = _git_info(repo_dir)
        self.user_raw_mapping = None
        self.item_raw_mapping = None
        self.user_raw_mapping_error = None
        self.item_raw_mapping_error = None
        os.makedirs(self.output_dir, exist_ok=True)
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.logger = logging.getLogger("nrgcf.edge_diagnostics.%s" % id(self))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(
            os.path.join(logs_dir, "diagnostics.log"), mode="w", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.handlers = [handler]

    def _basic_columns(self, start, end, edge_index, history, raw_momentum,
                       normalized_score, retained_mask, threshold, filtering_epoch,
                       warmup_epoch_count, user_degree, item_degree, min_degree):
        edge_ids = torch.arange(start, end, dtype=torch.int64)
        users = edge_index[0, start:end].detach().cpu().to(torch.int64)
        items = edge_index[1, start:end].detach().cpu().to(torch.int64)
        user_deg = user_degree[users].to(torch.int64)
        item_deg = item_degree[items].to(torch.int64)
        user_after = user_deg - 1
        item_after = item_deg - 1
        retained = retained_mask[start:end].detach().cpu().to(torch.bool)
        score = normalized_score[start:end].detach().cpu().to(torch.float64)
        size = int(end - start)
        user_ids = users.tolist()
        item_ids = items.tolist()
        user_raw = (
            [self.user_raw_mapping[value] for value in user_ids]
            if self.user_raw_mapping is not None else [None] * size
        )
        item_raw = (
            [self.item_raw_mapping[value] for value in item_ids]
            if self.item_raw_mapping is not None else [None] * size
        )
        columns = {
            "edge_id": edge_ids,
            "user_id_internal": users,
            "item_id_internal": items,
            "user_id_raw": user_raw,
            "item_id_raw": item_raw,
            "edge_position_in_training_graph": edge_ids.clone(),
            "is_original_observed_edge": [None] * size,
            "synthetic_is_noisy": [None] * size,
            "synthetic_noise_type": [None] * size,
            "train_split_identifier": ["%s/train.txt" % self.args.dataset] * size,
            "graph_version_identifier": [
                "%s:train:pre_filter:E=%d" % (self.args.dataset, history.edge_count)
            ] * size,
            "raw_edge_loss_at_filter_epoch": [None] * size,
            "historical_or_momentum_loss": raw_momentum[start:end].detach().cpu().to(torch.float64),
            "normalized_edge_score": score,
            "threshold_value": torch.full((size,), float(threshold), dtype=torch.float64),
            "score_minus_threshold": score - float(threshold),
            "nr_gcf_removed": ~retained,
            "nr_gcf_retained": retained,
            "filtering_epoch": torch.full((size,), int(filtering_epoch), dtype=torch.int64),
            "warmup_epoch_count": torch.full((size,), int(warmup_epoch_count), dtype=torch.int64),
            "user_degree_before": user_deg,
            "item_degree_before": item_deg,
            "user_degree_after_if_removed": user_after,
            "item_degree_after_if_removed": item_after,
            "user_becomes_isolated_if_removed": user_after == 0,
            "item_becomes_isolated_if_removed": item_after == 0,
            "user_below_min_degree_if_removed": user_after < int(min_degree),
            "item_below_min_degree_if_removed": item_after < int(min_degree),
            "min_endpoint_degree": torch.minimum(user_deg, item_deg),
            "max_endpoint_degree": torch.maximum(user_deg, item_deg),
            "inverse_user_degree": 1.0 / user_deg.to(torch.float64),
            "inverse_item_degree": 1.0 / item_deg.to(torch.float64),
            "normalized_degree_product": 1.0 / torch.sqrt(
                user_deg.to(torch.float64) * item_deg.to(torch.float64)
            ),
        }
        columns.update(history.chunk(start, end))
        return columns

    @staticmethod
    def _joint_columns(structure, normalized_score, retained, thresholds):
        user = structure["user_side_structure_mean"].to(torch.float64)
        item = structure["item_side_structure_mean"].to(torch.float64)
        both = torch.isfinite(user) & torch.isfinite(item)
        nan = torch.full_like(user, float("nan"))
        minimum = torch.where(both, torch.minimum(user, item), nan)
        maximum = torch.where(both, torch.maximum(user, item), nan)
        mean = torch.where(both, 0.5 * (user + item), nan)
        geometric = torch.where(both, torch.sqrt(user.clamp(min=0) * item.clamp(min=0)), nan)
        disagreement = torch.where(both, torch.abs(user - item), nan)

        loss_valid = torch.isfinite(normalized_score)
        structure_valid = torch.isfinite(mean)
        high_loss = loss_valid & (normalized_score >= thresholds["loss_high"])
        low_loss = loss_valid & (normalized_score <= thresholds["loss_low"])
        high_structure = structure_valid & (mean >= thresholds["structure_high"])
        low_structure = structure_valid & (mean <= thresholds["structure_low"])

        def nullable_flag(value, valid):
            result = []
            for flag, is_valid in zip(value.tolist(), valid.tolist()):
                result.append(bool(flag) if is_valid else None)
            return result

        joint_valid = loss_valid & structure_valid
        return {
            "bilateral_structure_min": minimum,
            "bilateral_structure_max": maximum,
            "bilateral_structure_mean": mean,
            "bilateral_structure_geometric_mean": geometric,
            "bilateral_disagreement": disagreement,
            "high_loss_high_structure_flag": nullable_flag(high_loss & high_structure, joint_valid),
            "high_loss_low_structure_flag": nullable_flag(high_loss & low_structure, joint_valid),
            "low_loss_low_structure_flag": nullable_flag(low_loss & low_structure, joint_valid),
            "removed_but_high_structure_flag": nullable_flag((~retained) & high_structure, structure_valid),
            "retained_but_low_structure_flag": nullable_flag(retained & low_structure, structure_valid),
        }

    @_no_grad()
    def export(self, edge_index, num_users, num_items, history, raw_momentum,
               normalized_score, post_threshold_score, retained_mask,
               filtering_epoch, warmup_epoch_count, threshold):
        edge_index = edge_index.detach()
        edge_count = int(edge_index.size(1))
        chunk_size = int(self.args.edge_diagnostics_chunk_size)
        topk = int(self.args.edge_diagnostics_topk)
        min_degree = int(self.args.edge_diagnostics_min_degree)
        if chunk_size <= 0 or topk <= 0 or min_degree < 0:
            raise ValueError("diagnostics chunk size/topk must be positive and min degree non-negative")
        if history.edge_count != edge_count:
            raise ValueError("loss-history edge count does not match original training graph")
        if not torch.equal(retained_mask, post_threshold_score > 0):
            raise ValueError("retained mask does not match the current post-threshold score tensor")

        self.logger.info("Starting edge diagnostics export for %d edges", edge_count)
        dataset_dir = os.path.join(self.repo_dir, "data", self.args.dataset)
        self.user_raw_mapping, self.user_raw_mapping_error = load_raw_id_mapping(
            os.path.join(dataset_dir, "user_list.txt"), num_users
        )
        self.item_raw_mapping, self.item_raw_mapping_error = load_raw_id_mapping(
            os.path.join(dataset_dir, "item_list.txt"), num_items
        )
        if self.user_raw_mapping_error:
            self.logger.warning("User raw-ID mapping: %s", self.user_raw_mapping_error)
        if self.item_raw_mapping_error:
            self.logger.warning("Item raw-ID mapping: %s", self.item_raw_mapping_error)
        engine = TwoHopMinHash(
            edge_index=edge_index,
            num_users=num_users,
            num_items=num_items,
            topk=topk,
            structural_mode=self.args.edge_diagnostics_structural_mode,
        )
        user_degree_cpu = engine.user_degree.detach().cpu()
        item_degree_cpu = engine.item_degree.detach().cpu()

        # Exact structure quantiles require only one temporary E-length scalar,
        # not a resident full diagnostics table.
        bilateral_mean = torch.full((edge_count,), float("nan"), dtype=torch.float32)
        if engine.enabled:
            for start in range(0, edge_count, chunk_size):
                end = min(edge_count, start + chunk_size)
                structure = engine.compute_chunk(start, end)
                user = structure["user_side_structure_mean"].to(torch.float32)
                item = structure["item_side_structure_mean"].to(torch.float32)
                valid = torch.isfinite(user) & torch.isfinite(item)
                bilateral_mean[start:end] = torch.where(
                    valid,
                    0.5 * (user + item),
                    torch.full_like(user, float("nan")),
                )

        loss_low = _finite_quantile(normalized_score.detach().cpu(), QUANTILE_LOW)
        loss_high = _finite_quantile(normalized_score.detach().cpu(), QUANTILE_HIGH)
        structure_low = _finite_quantile(bilateral_mean, QUANTILE_LOW)
        structure_high = _finite_quantile(bilateral_mean, QUANTILE_HIGH)
        # Nullable thresholds are represented by NaN internally; flags become null.
        thresholds = {
            "loss_low": float("nan") if loss_low is None else loss_low,
            "loss_high": float("nan") if loss_high is None else loss_high,
            "structure_low": float("nan") if structure_low is None else structure_low,
            "structure_high": float("nan") if structure_high is None else structure_high,
            "low_quantile": QUANTILE_LOW,
            "high_quantile": QUANTILE_HIGH,
        }

        writer = PartWriter(
            self.output_dir, self.args.edge_diagnostics_format, self.logger
        )
        labels_path = getattr(self.args, "edge_diagnostics_labels_file", None)
        label_reader = SyntheticLabelReader(labels_path) if labels_path else None
        summary_builder = SummaryBuilder(edge_count)
        try:
            for start in range(0, edge_count, chunk_size):
                end = min(edge_count, start + chunk_size)
                columns = self._basic_columns(
                    start=start,
                    end=end,
                    edge_index=edge_index,
                    history=history,
                    raw_momentum=raw_momentum,
                    normalized_score=normalized_score,
                    retained_mask=retained_mask,
                    threshold=threshold,
                    filtering_epoch=filtering_epoch,
                    warmup_epoch_count=warmup_epoch_count,
                    user_degree=user_degree_cpu,
                    item_degree=item_degree_cpu,
                    min_degree=min_degree,
                )
                structure = engine.compute_chunk(start, end)
                columns.update(structure)
                columns.update(self._joint_columns(
                    structure=structure,
                    normalized_score=columns["normalized_edge_score"],
                    retained=columns["nr_gcf_retained"],
                    thresholds=thresholds,
                ))
                if label_reader is not None:
                    columns.update(label_reader.read_chunk(
                        start,
                        end,
                        columns["user_id_internal"].tolist(),
                        columns["item_id_internal"].tolist(),
                        columns["nr_gcf_removed"].tolist(),
                    ))
                missing = set(FIELD_NAMES) - set(columns)
                extra = set(columns) - set(FIELD_NAMES)
                if missing or extra:
                    raise RuntimeError("diagnostics schema mismatch; missing=%s extra=%s" % (
                        sorted(missing), sorted(extra)
                    ))
                summary_builder.update(columns)
                path = writer.write(columns)
                self.logger.info("Wrote edges [%d, %d) to %s", start, end, path)
            if label_reader is not None:
                label_reader.verify_complete()
        finally:
            writer.close()
            if label_reader is not None:
                label_reader.close()

        retained_count = int(retained_mask.to(torch.int64).sum().item())
        removed_count = edge_count - retained_count
        metadata_thresholds = dict(
            (name, value if isinstance(value, (int, float)) and math.isfinite(value) else None)
            for name, value in thresholds.items()
        )
        noise_validation = None
        noise_validation_path = getattr(
            self.args, "edge_diagnostics_noise_validation_file", None
        )
        if noise_validation_path:
            with open(noise_validation_path, encoding="utf-8") as stream:
                noise_validation = json.load(stream)
        actual_noise_ratio = (
            noise_validation.get("actual_noise_ratio")
            if isinstance(noise_validation, dict) else None
        )
        metadata = {
            "dataset": self.args.dataset,
            "seed_argument": self.args.seed,
            "seed_is_applied_by_current_nrgcf_code": True,
            "seed_reproducibility_scope": "Python, NumPy, torch CPU, and all visible CUDA generators are seeded before data/model construction; sparse CUDA kernels may still have platform-specific nondeterminism.",
            "requested_noise_ratio": self.args.requested_noise_ratio,
            "actual_noise_ratio": actual_noise_ratio,
            "number_of_users": int(num_users),
            "number_of_items": int(num_items),
            "train_edge_count": edge_count,
            "retained_edge_count": retained_count,
            "removed_edge_count": removed_count,
            "removed_ratio": float(removed_count) / max(edge_count, 1),
            "filtering_epoch": int(filtering_epoch),
            "warmup_epoch_count": int(warmup_epoch_count),
            "threshold": float(threshold),
            "threshold_semantics": "Current code uses beta, zeros normalized scores > beta, then retains post-threshold scores > 0; normalized minima are also removed.",
            "quantile_thresholds": metadata_thresholds,
            "command_line": list(sys.argv),
            "config": {
                "model_config": _json_safe(self.model_config),
                "arguments": _json_safe(vars(self.args)),
            },
            "diagnostics_schema_version": SCHEMA_VERSION,
            "requested_format": writer.requested_format,
            "actual_format": writer.actual_format,
            "format_fallback_reason": writer.fallback_reason,
            "parquet_compression": writer.parquet_compression,
            "edge_table_path": os.path.basename(writer.path),
            "streaming_write_count": writer.part_index,
            "structural_mode": self.args.edge_diagnostics_structural_mode,
            "structural_signature_dim": MINHASH_DIM if engine.enabled else None,
            "topk": topk,
            "representative_neighbor_limit": engine.support_limit,
            "chunk_size": chunk_size,
            "minimum_degree_for_risk_flag": min_degree,
            "code_commit_hash": self.code_commit_hash,
            "code_worktree_dirty": self.code_tracked_worktree_dirty,
            "code_worktree_dirty_semantics": "Tracked files compared with HEAD; untracked run artifacts are intentionally ignored.",
            "export_timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "paper_code_difference_summary": [
                "Current loop starts at epoch 1, so the epoch==0 momentum initialization branch is not used.",
                "Current momentum coefficient is epoch/10 and exceeds one at epochs 11-14.",
                "Current code uses fixed beta=0.8 rather than the paper's mean-adaptive threshold.",
                "Current mask also removes the minimum normalized score because retention requires score > 0.",
                "Filtered edges are not assigned back to dataset.train_edge_index or model.edge_index and are not renormalized.",
                "Representation modulation is active from the first epoch rather than switched on only after filtering.",
            ],
            "graph_used_for_each_feature": {
                "loss_history": "Already-computed instance losses on the current model's original normalized training graph.",
                "historical_or_momentum_loss": "Exact raw model.momentum_loss immediately before current min-max normalization.",
                "filter_decision": "Exact current local post-threshold tensor and mask at epoch 15.",
                "degree_and_connectivity": "Original observed training edge multiset before filtering.",
                "structural_features": "Coalesced simple graph made only from pre-filter training edges; no validation or test edges.",
            },
            "edge_identity": {
                "definition": "edge_id equals original dataset.train_edge_index column position.",
                "duplicates_deduplicated_for_identity": False,
                "duplicate_semantics": "Duplicate user-item occurrences receive distinct edge IDs; structural overlap uses a coalesced simple graph and therefore gives duplicate coordinates the same graph context.",
                "raw_id_mapping": "Original IDs are loaded from user_list.txt/item_list.txt by explicit remap_id lookup; unavailable entries are null.",
            },
            "raw_id_mapping": {
                "user_mapping_available": self.user_raw_mapping is not None,
                "item_mapping_available": self.item_raw_mapping is not None,
                "user_mapping_note": self.user_raw_mapping_error,
                "item_mapping_note": self.item_raw_mapping_error,
            },
            "structural_feature_notes": diagnostics_schema()["structural_approximation"],
            "raw_edge_loss_at_filter_epoch_available": False,
            "is_original_observed_edge_available": label_reader is not None,
            "synthetic_labels_available": label_reader is not None,
            "synthetic_label_source": os.path.abspath(labels_path) if labels_path else None,
            "noise_validation": noise_validation,
        }
        summary = summary_builder.result()
        clean_count = summary_builder.synthetic_clean_count if label_reader else None
        noisy_count = summary_builder.synthetic_noisy_count if label_reader else None
        summary.update({
            "total_edge_count": edge_count,
            "retained_edge_count": retained_count,
            "removed_edge_count": removed_count,
            "removed_ratio": float(removed_count) / max(edge_count, 1),
            "synthetic_clean_count": clean_count,
            "synthetic_noisy_count": noisy_count,
            "clean_removal_rate": (
                float(summary_builder.synthetic_clean_removed_count) / clean_count
                if clean_count else None
            ),
            "noisy_removal_rate": (
                float(summary_builder.synthetic_noisy_removed_count) / noisy_count
                if noisy_count else None
            ),
            "clean_noisy_group_statistics": (
                {
                    "synthetic_clean": summary["removed_retained_and_degree_bucket_statistics"].get("synthetic_clean"),
                    "synthetic_noisy": summary["removed_retained_and_degree_bucket_statistics"].get("synthetic_noisy"),
                }
                if label_reader else None
            ),
        })
        _write_json(os.path.join(self.output_dir, "metadata.json"), metadata)
        _write_json(os.path.join(self.output_dir, "schema.json"), diagnostics_schema())
        _write_json(os.path.join(self.output_dir, "summary.json"), summary)
        self.logger.info("Edge diagnostics export completed")
        return metadata


def write_invariance_report(output_dir, result):
    _write_json(os.path.join(os.path.abspath(output_dir), "invariance.json"), result)


def compute_degree_connectivity_reference(edges, min_degree=2):
    """Dependency-free edge-occurrence degree reference for unit tests."""
    user_degree = Counter(user for user, _ in edges)
    item_degree = Counter(item for _, item in edges)
    result = []
    for edge_id, (user, item) in enumerate(edges):
        du = user_degree[user]
        di = item_degree[item]
        result.append({
            "edge_id": edge_id,
            "user_degree_before": du,
            "item_degree_before": di,
            "user_degree_after_if_removed": du - 1,
            "item_degree_after_if_removed": di - 1,
            "user_becomes_isolated_if_removed": du == 1,
            "item_becomes_isolated_if_removed": di == 1,
            "user_below_min_degree_if_removed": du - 1 < min_degree,
            "item_below_min_degree_if_removed": di - 1 < min_degree,
            "normalized_degree_product": 1.0 / math.sqrt(float(du * di)),
        })
    return result


def compute_structural_features_reference(edges, topk=10):
    """Exact tiny-graph definition used to validate the approximate production path.

    Each edge occurrence is distinct.  For user-side similarity the target
    occurrence is subtracted from the candidate item's user-count vector, then
    compared with every other item occurrence incident to the user.  The item
    side is symmetric.
    """
    user_edges = defaultdict(list)
    item_edges = defaultdict(list)
    item_users = defaultdict(Counter)
    user_items = defaultdict(Counter)
    for edge_id, (user, item) in enumerate(edges):
        user_edges[user].append(edge_id)
        item_edges[item].append(edge_id)
        item_users[item][user] += 1
        user_items[user][item] += 1

    def cosine(counter_a, degree_a, counter_b, degree_b):
        if degree_a <= 0 or degree_b <= 0:
            return None
        dot = sum(value * counter_b.get(key, 0) for key, value in counter_a.items())
        return float(dot) / math.sqrt(float(degree_a * degree_b))

    rows = []
    for edge_id, (user, item) in enumerate(edges):
        candidate_item = item_users[item].copy()
        candidate_item[user] -= 1
        if candidate_item[user] == 0:
            del candidate_item[user]
        item_degree_after = sum(candidate_item.values())
        user_scores = []
        if item_degree_after > 0:
            for other_edge_id in user_edges[user]:
                if other_edge_id == edge_id:
                    continue
                other_item = edges[other_edge_id][1]
                score = cosine(
                    candidate_item,
                    item_degree_after,
                    item_users[other_item],
                    sum(item_users[other_item].values()),
                )
                if score is not None:
                    user_scores.append(score)

        candidate_user = user_items[user].copy()
        candidate_user[item] -= 1
        if candidate_user[item] == 0:
            del candidate_user[item]
        user_degree_after = sum(candidate_user.values())
        item_scores = []
        if user_degree_after > 0:
            for other_edge_id in item_edges[item]:
                if other_edge_id == edge_id:
                    continue
                other_user = edges[other_edge_id][0]
                score = cosine(
                    candidate_user,
                    user_degree_after,
                    user_items[other_user],
                    sum(user_items[other_user].values()),
                )
                if score is not None:
                    item_scores.append(score)

        def aggregate(scores):
            if not scores:
                return None, None, None, 0
            ordered = sorted(scores, reverse=True)
            return (
                sum(scores) / len(scores),
                ordered[0],
                sum(ordered[:topk]) / min(topk, len(ordered)),
                len(scores),
            )

        user_mean, user_max, user_topk, user_count = aggregate(user_scores)
        item_mean, item_max, item_topk, item_count = aggregate(item_scores)
        rows.append({
            "edge_id": edge_id,
            "user_side_structure_mean": user_mean,
            "user_side_structure_max": user_max,
            "user_side_structure_topk_mean": user_topk,
            "user_side_valid_neighbor_count": user_count,
            "item_side_structure_mean": item_mean,
            "item_side_structure_max": item_max,
            "item_side_structure_topk_mean": item_topk,
            "item_side_valid_neighbor_count": item_count,
        })
    return rows
