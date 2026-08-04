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
from collections import Counter, defaultdict


PROTOCOL_NAME = "degree_preserving_edge_swap"
HARD_PROTOCOL_NAME = "degree_preserving_hard_two_hop_edge_swap"


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


def _write_labels(path, clean_edges, variant_edges, swap_pair_ids, protocol_name):
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
                protocol_name if noisy else "",
                pair_id if noisy else "",
            ])


def _bounded(values, limit, salt):
    """Select deterministic bounded supports without consuming RNG state."""
    if len(values) <= limit:
        return sorted(values)
    return sorted(
        values,
        key=lambda value: ((int(value) * 1103515245 + int(salt) * 12345) % 2147483647),
    )[:limit]


def _loo_two_hop_candidate_score(
        user, item, removed_item, removed_user,
        user_items, item_users, user_degree, item_degree, support_limit):
    """Bounded leave-one-out bilateral normalized two-hop consistency.

    The candidate edge is not inserted.  The two original edges that the swap
    would remove are analytically excluded.  This avoids candidate self-impact
    while keeping generation independent of trained embeddings and labels.
    """
    candidate_item_users = item_users[item] - {removed_user}
    candidate_user_items = user_items[user] - {removed_item}
    side_scores = []

    other_items = _bounded(
        candidate_user_items, support_limit, user * 1000003 + item
    )
    if other_items and candidate_item_users:
        candidate_degree = len(candidate_item_users)
        scores = []
        for other_item in other_items:
            overlap = len(candidate_item_users & item_users[other_item])
            denominator = math.sqrt(candidate_degree * item_degree[other_item])
            scores.append(overlap / denominator if denominator else 0.0)
        side_scores.append(sum(scores) / len(scores))

    other_users = _bounded(
        candidate_item_users, support_limit, item * 1000003 + user
    )
    if other_users and candidate_user_items:
        candidate_degree = len(candidate_user_items)
        scores = []
        for other_user in other_users:
            overlap = len(candidate_user_items & user_items[other_user])
            denominator = math.sqrt(candidate_degree * user_degree[other_user])
            scores.append(overlap / denominator if denominator else 0.0)
        side_scores.append(sum(scores) / len(scores))
    return sum(side_scores) / len(side_scores) if side_scores else 0.0


def generate_degree_preserving_replace(
        clean_train,
        requested_ratio,
        seed,
        output_train,
        labels_path,
        generation_metadata_path,
        validation_path,
        selection="uniform",
        candidate_pool_size=8,
        structural_support_limit=16):
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
    if selection not in ("uniform", "hard_two_hop"):
        raise ValueError("selection must be uniform or hard_two_hop")
    candidate_pool_size = int(candidate_pool_size)
    structural_support_limit = int(structural_support_limit)
    if candidate_pool_size <= 0 or structural_support_limit <= 0:
        raise ValueError("candidate pool size and structural support limit must be positive")

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
    attempt_multiplier = 200 * (candidate_pool_size if selection == "hard_two_hop" else 1)
    max_attempts = max(100000, max(pair_target, 1) * attempt_multiplier)

    user_items = defaultdict(set)
    item_users = defaultdict(set)
    for user, item in clean_edges:
        user_items[user].add(item)
        item_users[item].add(user)
    user_degree = Counter(user for user, _ in clean_edges)
    item_degree = Counter(item for _, item in clean_edges)
    hard_candidate_score_sum = 0.0
    hard_candidate_score_count = 0
    hard_chosen_score_sum = 0.0
    hard_chosen_score_min = None
    hard_chosen_score_max = None

    def valid_candidate(first_edge_position, second_edge_position):
        user_a, item_a = clean_edges[first_edge_position]
        user_b, item_b = clean_edges[second_edge_position]
        if user_a == user_b or item_a == item_b:
            return None
        swapped_a = (user_a, item_b)
        swapped_b = (user_b, item_a)
        if swapped_a in clean_edge_set or swapped_b in clean_edge_set:
            return None
        if swapped_a in generated_edges or swapped_b in generated_edges:
            return None
        return swapped_a, swapped_b

    def draw_candidate():
        first_list_position = rng.randrange(len(available_positions))
        second_list_position = rng.randrange(len(available_positions) - 1)
        if second_list_position >= first_list_position:
            second_list_position += 1
        first_edge_position = available_positions[first_list_position]
        second_edge_position = available_positions[second_list_position]
        swapped = valid_candidate(first_edge_position, second_edge_position)
        if swapped is None:
            return None
        return (
            first_list_position,
            second_list_position,
            first_edge_position,
            second_edge_position,
            swapped[0],
            swapped[1],
        )

    pair_id = 0
    while pair_id < pair_target and attempts < max_attempts:
        if len(available_positions) < 2:
            break
        candidates = []
        target_candidates = candidate_pool_size if selection == "hard_two_hop" else 1
        candidate_attempt_limit = max(50, target_candidates * 50)
        local_attempts = 0
        while (len(candidates) < target_candidates
               and local_attempts < candidate_attempt_limit
               and attempts < max_attempts):
            local_attempts += 1
            attempts += 1
            candidate = draw_candidate()
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            continue

        if selection == "hard_two_hop":
            scored = []
            for candidate in candidates:
                _, _, first_edge_position, second_edge_position, swapped_a, swapped_b = candidate
                user_a, item_a = clean_edges[first_edge_position]
                user_b, item_b = clean_edges[second_edge_position]
                score_a = _loo_two_hop_candidate_score(
                    user_a, item_b, item_a, user_b,
                    user_items, item_users, user_degree, item_degree,
                    structural_support_limit,
                )
                score_b = _loo_two_hop_candidate_score(
                    user_b, item_a, item_b, user_a,
                    user_items, item_users, user_degree, item_degree,
                    structural_support_limit,
                )
                scored.append((0.5 * (score_a + score_b), candidate))
            candidate_score_sum = sum(value[0] for value in scored)
            hard_candidate_score_sum += candidate_score_sum
            hard_candidate_score_count += len(scored)
            chosen_score, chosen = max(scored, key=lambda value: value[0])
            hard_chosen_score_sum += chosen_score
            hard_chosen_score_min = (
                chosen_score if hard_chosen_score_min is None
                else min(hard_chosen_score_min, chosen_score)
            )
            hard_chosen_score_max = (
                chosen_score if hard_chosen_score_max is None
                else max(hard_chosen_score_max, chosen_score)
            )
        else:
            chosen = candidates[0]

        (first_list_position, second_list_position,
         first_edge_position, second_edge_position,
         swapped_a, swapped_b) = chosen

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
    protocol_name = HARD_PROTOCOL_NAME if selection == "hard_two_hop" else PROTOCOL_NAME
    _write_labels(
        labels_path, clean_edges, variant_edges, swap_pair_ids, protocol_name
    )
    actual_ratio = replacement_count / float(edge_count)
    generation_metadata = {
        "generator": protocol_name,
        "seed": seed,
        "requested_noise_ratio": ratio,
        "requested_replacement_count_before_even_adjustment": initially_requested,
        "replacement_count": replacement_count,
        "swap_pair_count": pair_target,
        "attempt_count": attempts,
        "noise_ratio_definition": "replaced_edge_positions / original_edge_count",
        "pair_operation": "(u1,i1),(u2,i2) -> (u1,i2),(u2,i1)",
        "selection": selection,
        "hard_selection_definition": (
            "best of a random candidate pool by bounded bilateral leave-one-out "
            "degree-normalized two-hop overlap"
            if selection == "hard_two_hop" else None
        ),
        "candidate_pool_size": candidate_pool_size if selection == "hard_two_hop" else 1,
        "structural_support_limit": structural_support_limit if selection == "hard_two_hop" else None,
        "hard_selection_approximation": (
            "candidate supports are deterministically bounded; no dense node-node "
            "matrix or all-pair neighbor Cartesian product is constructed"
            if selection == "hard_two_hop" else None
        ),
        "hard_candidate_score_mean": (
            hard_candidate_score_sum / hard_candidate_score_count
            if hard_candidate_score_count else None
        ),
        "hard_chosen_score_mean": (
            hard_chosen_score_sum / pair_target if pair_target else None
        ),
        "hard_chosen_score_min": hard_chosen_score_min,
        "hard_chosen_score_max": hard_chosen_score_max,
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
        "synthetic_noise_type": protocol_name,
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
    parser.add_argument(
        "--selection", choices=["uniform", "hard_two_hop"], default="uniform"
    )
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--structural-support-limit", type=int, default=16)
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
        selection=args.selection,
        candidate_pool_size=args.candidate_pool_size,
        structural_support_limit=args.structural_support_limit,
    )
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
