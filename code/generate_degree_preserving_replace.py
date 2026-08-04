"""Generate deterministic degree-preserving replacement noise for NR-GCF.

The generator swaps the item endpoints of pairs of observed training edges.
It never reads validation/test data and rejects swaps whose new coordinates
already occur in the clean training graph.  Every successful swap preserves
the degree of every user and every item, the total edge count, and the original
edge-position order used by the NR-GCF loader.
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
from collections import Counter


PROTOCOL_NAME = "degree_preserving_edge_swap"


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_train(path):
    path = pathlib.Path(path)
    rows = []
    ordered_edges = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                user = int(fields[0])
                items = [int(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(
                    "Non-integer ID in %s:%d: %s" % (path, line_number, exc)
                )
            if user < 0 or any(item < 0 for item in items):
                raise ValueError("Negative ID in %s:%d" % (path, line_number))
            rows.append((user, len(items)))
            for item in items:
                edge = (user, item)
                if edge in seen:
                    raise ValueError(
                        "Duplicate edge %r in %s:%d" % (edge, path, line_number)
                    )
                seen.add(edge)
                ordered_edges.append(edge)
    if not ordered_edges:
        raise ValueError("No training edges found in %s" % path)
    return rows, ordered_edges


def _nearest_even_replacement_count(edge_count, ratio):
    requested_count = int(round(float(ratio) * int(edge_count)))
    if requested_count % 2 == 0:
        return requested_count, requested_count
    if requested_count < edge_count:
        return requested_count, requested_count + 1
    return requested_count, requested_count - 1


def _remove_list_position(values, position):
    value = values[position]
    values[position] = values[-1]
    values.pop()
    return value


def _write_train(path, rows, variant_edges):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with path.open("w", encoding="utf-8") as stream:
        for user, item_count in rows:
            row_edges = variant_edges[cursor:cursor + item_count]
            if any(edge_user != user for edge_user, _ in row_edges):
                raise AssertionError("A user endpoint changed during item swapping")
            values = [str(user)] + [str(item) for _, item in row_edges]
            stream.write(" ".join(values) + "\n")
            cursor += item_count
    if cursor != len(variant_edges):
        raise AssertionError("Training-row reconstruction did not consume every edge")


def _write_labels(path, clean_edges, variant_edges, swap_pair_ids):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "edge_id",
            "user_id_internal",
            "item_id_internal",
            "original_user_id_internal",
            "original_item_id_internal",
            "is_original_observed_edge",
            "synthetic_is_noisy",
            "synthetic_noise_type",
            "swap_pair_id",
        ])
        for edge_id, (clean_edge, variant_edge, pair_id) in enumerate(
                zip(clean_edges, variant_edges, swap_pair_ids)):
            noisy = pair_id is not None
            writer.writerow([
                edge_id,
                variant_edge[0],
                variant_edge[1],
                clean_edge[0],
                clean_edge[1],
                not noisy,
                noisy,
                PROTOCOL_NAME if noisy else "",
                pair_id if noisy else "",
            ])


def generate_degree_preserving_replace(
        clean_train,
        requested_ratio,
        seed,
        output_train,
        labels_path,
        generation_metadata_path,
        validation_path):
    """Generate one replacement-noise split and return validation metadata."""
    ratio = float(requested_ratio)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError("Replacement noise ratio must be within [0, 1]")
    seed = int(seed)
    clean_train = pathlib.Path(clean_train)
    output_train = pathlib.Path(output_train)
    labels_path = pathlib.Path(labels_path)
    generation_metadata_path = pathlib.Path(generation_metadata_path)
    validation_path = pathlib.Path(validation_path)

    rows, clean_edges = _read_train(clean_train)
    edge_count = len(clean_edges)
    initially_requested, replacement_count = _nearest_even_replacement_count(
        edge_count, ratio
    )
    if replacement_count < 0 or replacement_count > edge_count:
        raise ValueError("Replacement count is outside the training-edge range")

    clean_edge_set = set(clean_edges)
    variant_edges = list(clean_edges)
    swap_pair_ids = [None] * edge_count
    generated_edges = set()
    available_positions = list(range(edge_count))
    rng = random.Random(seed)
    pair_target = replacement_count // 2
    attempts = 0
    max_attempts = max(100000, max(pair_target, 1) * 200)

    pair_id = 0
    while pair_id < pair_target and attempts < max_attempts:
        attempts += 1
        if len(available_positions) < 2:
            break
        first_list_position = rng.randrange(len(available_positions))
        second_list_position = rng.randrange(len(available_positions) - 1)
        if second_list_position >= first_list_position:
            second_list_position += 1
        first_edge_position = available_positions[first_list_position]
        second_edge_position = available_positions[second_list_position]
        user_a, item_a = clean_edges[first_edge_position]
        user_b, item_b = clean_edges[second_edge_position]
        if user_a == user_b or item_a == item_b:
            continue
        swapped_a = (user_a, item_b)
        swapped_b = (user_b, item_a)
        if swapped_a in clean_edge_set or swapped_b in clean_edge_set:
            continue
        if swapped_a in generated_edges or swapped_b in generated_edges:
            continue

        variant_edges[first_edge_position] = swapped_a
        variant_edges[second_edge_position] = swapped_b
        swap_pair_ids[first_edge_position] = pair_id
        swap_pair_ids[second_edge_position] = pair_id
        generated_edges.add(swapped_a)
        generated_edges.add(swapped_b)
        for list_position in sorted(
                (first_list_position, second_list_position), reverse=True):
            _remove_list_position(available_positions, list_position)
        pair_id += 1

    if pair_id != pair_target:
        raise RuntimeError(
            "Could create only %d/%d valid swap pairs after %d attempts; "
            "use a smaller ratio or a less dense graph"
            % (pair_id, pair_target, attempts)
        )

    variant_edge_set = set(variant_edges)
    if len(variant_edge_set) != edge_count:
        raise AssertionError("Replacement produced duplicate observed edges")
    observed_clean = variant_edge_set & clean_edge_set
    injected = variant_edge_set - clean_edge_set
    missing_clean = clean_edge_set - variant_edge_set
    if len(injected) != replacement_count or len(missing_clean) != replacement_count:
        raise AssertionError("Replacement clean/noisy membership counts are inconsistent")

    clean_user_degree = Counter(user for user, _ in clean_edges)
    clean_item_degree = Counter(item for _, item in clean_edges)
    variant_user_degree = Counter(user for user, _ in variant_edges)
    variant_item_degree = Counter(item for _, item in variant_edges)
    user_degree_preserved = clean_user_degree == variant_user_degree
    item_degree_preserved = clean_item_degree == variant_item_degree
    if not user_degree_preserved or not item_degree_preserved:
        raise AssertionError("Endpoint degrees were not exactly preserved")

    _write_train(output_train, rows, variant_edges)
    _write_labels(labels_path, clean_edges, variant_edges, swap_pair_ids)
    actual_ratio = replacement_count / float(edge_count)
    generation_metadata = {
        "generator": PROTOCOL_NAME,
        "seed": seed,
        "requested_noise_ratio": ratio,
        "requested_replacement_count_before_even_adjustment": initially_requested,
        "replacement_count": replacement_count,
        "swap_pair_count": pair_target,
        "attempt_count": attempts,
        "noise_ratio_definition": "replaced_edge_positions / original_edge_count",
        "pair_operation": "(u1,i1),(u2,i2) -> (u1,i2),(u2,i1)",
        "selection": "uniform random unused edge positions with rejection",
        "rejection_conditions": [
            "same user or same item",
            "new coordinate exists in the clean training graph",
            "new coordinate duplicates an earlier generated edge",
        ],
        "test_data_read": False,
        "held_out_positive_overlap_status": (
            "unknown because validation/test data is intentionally not read"
        ),
        "edge_order": "original edge positions retained; only item endpoints change",
        "user_degrees_preserved": user_degree_preserved,
        "item_degrees_preserved": item_degree_preserved,
    }
    validation = {
        "requested_noise_ratio": ratio,
        "actual_noise_ratio": actual_ratio,
        "noise_ratio_definition": "replaced_edge_positions / original_edge_count",
        "clean_edge_count": edge_count,
        "variant_edge_count": len(variant_edges),
        "observed_clean_edge_count": len(observed_clean),
        "synthetic_noisy_edge_count": len(injected),
        "removed_original_clean_edge_count": len(missing_clean),
        "swap_pair_count": pair_target,
        "duplicate_edge_count": 0,
        "total_edge_count_preserved": len(variant_edges) == edge_count,
        "all_user_degrees_preserved": user_degree_preserved,
        "all_item_degrees_preserved": item_degree_preserved,
        "clean_train_sha256": _sha256(clean_train),
        "variant_train_sha256": _sha256(output_train),
        "label_join_key": "edge_id equals original and variant loader column position",
        "synthetic_noise_type": PROTOCOL_NAME,
        "label_leakage_note": (
            "Labels come only from generated swap positions and are not read by "
            "NR-GCF training or structural feature computation."
        ),
    }
    generation_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_metadata_path.write_text(
        json.dumps(generation_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate degree-preserving edge-swap replacement noise"
    )
    parser.add_argument("--clean-train", required=True)
    parser.add_argument("--noise-ratio", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--generation-metadata", required=True)
    parser.add_argument("--validation", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    validation = generate_degree_preserving_replace(
        clean_train=args.clean_train,
        requested_ratio=args.noise_ratio,
        seed=args.seed,
        output_train=args.output_train,
        labels_path=args.labels,
        generation_metadata_path=args.generation_metadata,
        validation_path=args.validation,
    )
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
