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


SCHEMA_VERSION = "nrgcf_edge_diagnostics_v1"
COUNT_SKETCH_DIM = 64
QUANTILE_LOW = 0.20
QUANTILE_HIGH = 0.80


FIELD_SPECS = [
    ("edge_id", "int64", False, "Stable original column position in dataset.train_edge_index."),
    ("user_id_internal", "int64", False, "User ID used directly by NR-GCF."),
    ("item_id_internal", "int64", False, "Item ID used directly by NR-GCF."),
    ("user_id_raw", "int64", True, "Raw file ID; equal to internal ID because this loader does not remap IDs."),
    ("item_id_raw", "int64", True, "Raw file ID; equal to internal ID because this loader does not remap IDs."),
    ("edge_position_in_training_graph", "int64", False, "Stable original training-edge column position."),
    ("is_original_observed_edge", "bool", True, "Null: current loader cannot distinguish original observations from injected synthetic edges."),
    ("synthetic_is_noisy", "bool", True, "Null: the current loader has no synthetic-noise label."),
    ("synthetic_noise_type", "string", True, "Null: the current loader has no synthetic-noise type."),
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
    ("user_side_structure_mean", "float64", True, "CountSketch approximation of mean LOO normalized item-item co-occurrence over all other edge occurrences."),
    ("user_side_structure_max", "float64", True, "Approximate maximum over deterministic bounded representative neighbors."),
    ("user_side_structure_topk_mean", "float64", True, "Approximate top-k mean over deterministic bounded representative neighbors."),
    ("user_side_valid_neighbor_count", "int64", False, "All valid other user-neighbor edge occurrences used by the mean."),
    ("user_side_sampled_neighbor_count", "int64", False, "Representative neighbors used by approximate max/top-k."),
    ("item_side_structure_mean", "float64", True, "CountSketch approximation of mean LOO normalized user-user co-occurrence over all other edge occurrences."),
    ("item_side_structure_max", "float64", True, "Approximate maximum over deterministic bounded representative neighbors."),
    ("item_side_structure_topk_mean", "float64", True, "Approximate top-k mean over deterministic bounded representative neighbors."),
    ("item_side_valid_neighbor_count", "int64", False, "All valid other item-neighbor edge occurrences used by the mean."),
    ("item_side_sampled_neighbor_count", "int64", False, "Representative neighbors used by approximate max/top-k."),
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
            "mean": "CountSketch approximation over every valid direct neighbor edge occurrence.",
            "max_and_topk": "CountSketch approximation over a deterministic bounded edge-ID-hash representative set.",
            "target_edge_exclusion": "The target occurrence is removed analytically from its candidate fingerprint and neighbor aggregate.",
            "residual_self_influence": "Signed-hash collisions can leave approximation error; no target edge is deliberately included as direct evidence.",
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


def _signed_hash_basis(node_count, sketch_dim, device, salt):
    node_ids = torch.arange(int(node_count), device=device, dtype=torch.long)
    buckets = torch.remainder(node_ids * 1103515245 + 12345 + int(salt), int(sketch_dim))
    parity = torch.remainder(node_ids * 2654435761 + 1013904223 + int(salt), 2)
    signs = torch.where(
        parity == 0,
        torch.ones(node_count, device=device, dtype=torch.float32),
        -torch.ones(node_count, device=device, dtype=torch.float32),
    )
    basis = torch.zeros((int(node_count), int(sketch_dim)), device=device, dtype=torch.float32)
    basis[node_ids, buckets] = signs
    return basis, buckets, signs


class TwoHopCountSketch(object):
    """Scalable, deterministic two-hop structural feature engine.

    Sparse R is multiplied by fixed signed one-hot hash bases.  This estimates
    degree-normalized co-occurrence without materializing R^T R or R R^T.
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

        self.enabled = structural_mode == "two_hop_countsketch"
        if structural_mode not in ("two_hop_countsketch", "none"):
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
        self.interaction = interaction

        user_basis, self.user_bucket, self.user_sign = _signed_hash_basis(
            self.num_users, COUNT_SKETCH_DIM, self.device, salt=17
        )
        item_raw = torch.sparse.mm(self.interaction.transpose(0, 1), user_basis)
        del user_basis
        item_basis, self.item_bucket, self.item_sign = _signed_hash_basis(
            self.num_items, COUNT_SKETCH_DIM, self.device, salt=53
        )
        user_raw = torch.sparse.mm(self.interaction, item_basis)
        del item_basis

        self.item_full = item_raw / self.item_degree.clamp(min=1).sqrt().unsqueeze(1)
        self.user_full = user_raw / self.user_degree.clamp(min=1).sqrt().unsqueeze(1)
        del item_raw, user_raw

        self.item_neighbor_sum = torch.sparse.mm(self.interaction, self.item_full)
        self.user_neighbor_sum = torch.sparse.mm(
            self.interaction.transpose(0, 1), self.user_full
        )
        self.user_representative_edges = _stable_representative_edges(
            self.edge_index[0], self.num_users, self.support_limit
        )
        self.item_representative_edges = _stable_representative_edges(
            self.edge_index[1], self.num_items, self.support_limit
        )

    def _bounded_extrema(self, candidate, representatives, target_edge_ids,
                         neighbor_node_row, neighbor_full, candidate_valid):
        valid = representatives >= 0
        valid = valid & (representatives != target_edge_ids.unsqueeze(1))
        safe_edges = representatives.clamp(min=0)
        neighbor_ids = self.edge_index[neighbor_node_row][safe_edges]
        neighbor_vectors = neighbor_full[neighbor_ids]
        similarities = torch.sum(candidate.unsqueeze(1) * neighbor_vectors, dim=2)
        similarities = similarities.clamp(min=0.0, max=1.0)
        valid = valid & candidate_valid.unsqueeze(1)
        sampled_count = valid.sum(dim=1)
        masked = similarities.masked_fill(~valid, float("-inf"))
        maximum = masked.max(dim=1).values
        maximum = torch.where(
            sampled_count > 0,
            maximum,
            torch.full_like(maximum, float("nan")),
        )
        k = min(self.topk, self.support_limit)
        top_values = torch.topk(masked, k=k, dim=1).values
        finite = torch.isfinite(top_values)
        top_count = finite.sum(dim=1)
        top_sum = torch.where(finite, top_values, torch.zeros_like(top_values)).sum(dim=1)
        top_mean = top_sum / top_count.clamp(min=1)
        top_mean = torch.where(
            top_count > 0,
            top_mean,
            torch.full_like(top_mean, float("nan")),
        )
        return maximum, top_mean, sampled_count

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

        target_edge_ids = torch.arange(start, end, device=self.device, dtype=torch.long)
        users = self.edge_index[0, start:end]
        items = self.edge_index[1, start:end]
        user_degree = self.user_degree[users]
        item_degree = self.item_degree[items]
        row_ids = torch.arange(size, device=self.device)

        # Candidate item fingerprint after removing exactly this edge occurrence.
        candidate_item_raw = self.item_full[items] * item_degree.sqrt().unsqueeze(1)
        candidate_item_raw[row_ids, self.user_bucket[users]] -= self.user_sign[users]
        candidate_item = candidate_item_raw / (item_degree - 1).clamp(min=1).sqrt().unsqueeze(1)
        user_side_valid = (user_degree > 1) & (item_degree > 1)
        other_item_sum = self.item_neighbor_sum[users] - self.item_full[items]
        user_mean = torch.sum(candidate_item * other_item_sum, dim=1) / (user_degree - 1).clamp(min=1)
        user_mean = user_mean.clamp(min=0.0, max=1.0)
        user_mean = torch.where(
            user_side_valid, user_mean, torch.full_like(user_mean, float("nan"))
        )
        user_reps = self.user_representative_edges[users]
        user_max, user_topk, user_sampled = self._bounded_extrema(
            candidate_item,
            user_reps,
            target_edge_ids,
            neighbor_node_row=1,
            neighbor_full=self.item_full,
            candidate_valid=user_side_valid,
        )

        # Candidate user fingerprint after removing exactly this edge occurrence.
        candidate_user_raw = self.user_full[users] * user_degree.sqrt().unsqueeze(1)
        candidate_user_raw[row_ids, self.item_bucket[items]] -= self.item_sign[items]
        candidate_user = candidate_user_raw / (user_degree - 1).clamp(min=1).sqrt().unsqueeze(1)
        item_side_valid = (item_degree > 1) & (user_degree > 1)
        other_user_sum = self.user_neighbor_sum[items] - self.user_full[users]
        item_mean = torch.sum(candidate_user * other_user_sum, dim=1) / (item_degree - 1).clamp(min=1)
        item_mean = item_mean.clamp(min=0.0, max=1.0)
        item_mean = torch.where(
            item_side_valid, item_mean, torch.full_like(item_mean, float("nan"))
        )
        item_reps = self.item_representative_edges[items]
        item_max, item_topk, item_sampled = self._bounded_extrema(
            candidate_user,
            item_reps,
            target_edge_ids,
            neighbor_node_row=0,
            neighbor_full=self.user_full,
            candidate_valid=item_side_valid,
        )

        result = {
            "user_side_structure_mean": user_mean,
            "user_side_structure_max": user_max,
            "user_side_structure_topk_mean": user_topk,
            "user_side_valid_neighbor_count": torch.where(
                user_side_valid, (user_degree - 1).to(torch.int64), torch.zeros_like(users)
            ),
            "user_side_sampled_neighbor_count": user_sampled.to(torch.int64),
            "item_side_structure_mean": item_mean,
            "item_side_structure_max": item_max,
            "item_side_structure_topk_mean": item_topk,
            "item_side_valid_neighbor_count": torch.where(
                item_side_valid, (item_degree - 1).to(torch.int64), torch.zeros_like(items)
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
    def __init__(self, output_dir, requested_format, logger):
        self.output_dir = output_dir
        self.requested_format = requested_format
        self.actual_format = requested_format
        self.fallback_reason = None
        self.logger = logger
        self.part_index = 0
        self.pa = None
        self.pq = None
        if requested_format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                self.pa = pa
                self.pq = pq
            except ImportError as exc:
                self.actual_format = "csv"
                self.fallback_reason = "pyarrow unavailable: %s" % exc
                self.logger.warning(
                    "Parquet requested but pyarrow is unavailable; falling back to chunked CSV."
                )

    def _arrow_type(self, dtype):
        if dtype == "int64":
            return self.pa.int64()
        if dtype == "float64":
            return self.pa.float64()
        if dtype == "bool":
            return self.pa.bool_()
        return self.pa.string()

    def write(self, columns):
        extension = "parquet" if self.actual_format == "parquet" else "csv"
        filename = "edge_diagnostics_part_%05d.%s" % (self.part_index, extension)
        path = os.path.join(self.output_dir, filename)
        if self.actual_format == "parquet":
            arrays = []
            for name in FIELD_NAMES:
                values = _to_arrow_values(columns[name])
                arrays.append(self.pa.array(values, type=self._arrow_type(FIELD_TYPES[name])))
            table = self.pa.Table.from_arrays(arrays, names=FIELD_NAMES)
            self.pq.write_table(table, path)
        else:
            python_columns = dict(
                (name, _to_python_list(columns[name])) for name in FIELD_NAMES
            )
            row_count = len(python_columns[FIELD_NAMES[0]])
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(FIELD_NAMES)
                for row_index in range(row_count):
                    writer.writerow([python_columns[name][row_index] for name in FIELD_NAMES])
        self.part_index += 1
        return path


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
        dirty = bool(subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8").strip())
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
        columns = {
            "edge_id": edge_ids,
            "user_id_internal": users,
            "item_id_internal": items,
            "user_id_raw": users.clone(),
            "item_id_raw": items.clone(),
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
        engine = TwoHopCountSketch(
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
        summary_builder = SummaryBuilder(edge_count)
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
            missing = set(FIELD_NAMES) - set(columns)
            extra = set(columns) - set(FIELD_NAMES)
            if missing or extra:
                raise RuntimeError("diagnostics schema mismatch; missing=%s extra=%s" % (
                    sorted(missing), sorted(extra)
                ))
            summary_builder.update(columns)
            path = writer.write(columns)
            self.logger.info("Wrote edges [%d, %d) to %s", start, end, path)

        retained_count = int(retained_mask.to(torch.int64).sum().item())
        removed_count = edge_count - retained_count
        commit_hash, worktree_dirty = _git_info(self.repo_dir)
        metadata_thresholds = dict(
            (name, value if isinstance(value, (int, float)) and math.isfinite(value) else None)
            for name, value in thresholds.items()
        )
        metadata = {
            "dataset": self.args.dataset,
            "seed_argument": self.args.seed,
            "seed_is_applied_by_current_nrgcf_code": False,
            "requested_noise_ratio": self.args.requested_noise_ratio,
            "actual_noise_ratio": None,
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
            "structural_mode": self.args.edge_diagnostics_structural_mode,
            "structural_sketch_dim": COUNT_SKETCH_DIM,
            "topk": topk,
            "representative_neighbor_limit": engine.support_limit,
            "chunk_size": chunk_size,
            "minimum_degree_for_risk_flag": min_degree,
            "code_commit_hash": commit_hash,
            "code_worktree_dirty": worktree_dirty,
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
                "structural_features": "Original observed training edge multiset before filtering; no validation or test edges.",
            },
            "edge_identity": {
                "definition": "edge_id equals original dataset.train_edge_index column position.",
                "duplicates_deduplicated_for_identity": False,
                "duplicate_semantics": "Duplicate user-item occurrences receive distinct edge IDs. Sparse structural multiplication coalesces coordinates by summing multiplicity.",
                "raw_id_mapping": "No remapping: raw numeric IDs are used as internal IDs.",
            },
            "structural_feature_notes": diagnostics_schema()["structural_approximation"],
            "raw_edge_loss_at_filter_epoch_available": False,
            "is_original_observed_edge_available": False,
            "synthetic_labels_available": False,
        }
        summary = summary_builder.result()
        summary.update({
            "total_edge_count": edge_count,
            "retained_edge_count": retained_count,
            "removed_edge_count": removed_count,
            "removed_ratio": float(removed_count) / max(edge_count, 1),
            "synthetic_clean_count": None,
            "synthetic_noisy_count": None,
            "clean_removal_rate": None,
            "noisy_removal_rate": None,
            "clean_noisy_group_statistics": None,
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
