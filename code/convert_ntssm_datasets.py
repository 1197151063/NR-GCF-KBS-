#!/usr/bin/env python3
"""Convert NT-SSM triple splits to the grouped NR-GCF data format.

The conversion deliberately merges only ``train.txt`` and ``valid.txt``.
Every source test interaction is retained in ``test.txt``.  Numeric IDs are
preserved, and the converter rejects malformed input, duplicate interactions,
split overlap, or non-contiguous global IDs instead of silently changing the
evaluation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Pair = Tuple[int, int]
SUPPORTED_DATASETS = ("lastfm", "ml-1m")
SCHEMA_VERSION = "nrgcf-ntssm-conversion-v1"


@dataclass(frozen=True)
class TripleSplit:
    name: str
    path: Path
    pairs: Tuple[Pair, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_triple_split(path: Path, split_name: str) -> TripleSplit:
    pairs: List[Pair] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected three columns, got {len(tokens)}"
                )
            try:
                user_id, item_id = int(tokens[0]), int(tokens[1])
                value = float(tokens[2])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: IDs and interaction value must be numeric"
                ) from error
            if user_id < 0 or item_id < 0:
                raise ValueError(f"{path}:{line_number}: negative IDs are unsupported")
            if value != 1.0:
                raise ValueError(
                    f"{path}:{line_number}: expected implicit value 1, got {tokens[2]}"
                )
            pairs.append((user_id, item_id))

    if not pairs:
        raise ValueError(f"{path}: split is empty")
    duplicate_count = len(pairs) - len(set(pairs))
    if duplicate_count:
        raise ValueError(
            f"{path}: contains {duplicate_count} duplicate user-item rows; "
            "conversion refuses to guess a deduplication policy"
        )
    return TripleSplit(split_name, path, tuple(pairs))


def group_pairs(pairs: Iterable[Pair]) -> "OrderedDict[int, List[int]]":
    grouped: "OrderedDict[int, List[int]]" = OrderedDict()
    for user_id, item_id in pairs:
        grouped.setdefault(user_id, []).append(item_id)
    return grouped


def write_grouped_pairs(path: Path, grouped: Mapping[int, Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for user_id, item_ids in grouped.items():
            if not item_ids:
                continue
            stream.write(" ".join((str(user_id), *(str(item) for item in item_ids))))
            stream.write("\n")


def read_grouped_pairs(path: Path) -> Tuple[Pair, ...]:
    pairs: List[Pair] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            tokens = line.split()
            if len(tokens) < 2:
                raise ValueError(
                    f"{path}:{line_number}: expected one user and at least one item"
                )
            try:
                user_id = int(tokens[0])
                item_ids = [int(token) for token in tokens[1:]]
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: grouped IDs must be integers"
                ) from error
            pairs.extend((user_id, item_id) for item_id in item_ids)
    return tuple(pairs)


def _contiguity(values: Iterable[int]) -> Dict[str, object]:
    unique = set(values)
    minimum = min(unique)
    maximum = max(unique)
    missing = maximum - minimum + 1 - len(unique)
    return {
        "count": len(unique),
        "min": minimum,
        "max": maximum,
        "missing_within_range": missing,
        "zero_based_contiguous": minimum == 0 and missing == 0,
    }


def _split_stats(pairs: Sequence[Pair]) -> Dict[str, object]:
    users = {user for user, _ in pairs}
    items = {item for _, item in pairs}
    return {
        "interaction_count": len(pairs),
        "unique_interaction_count": len(set(pairs)),
        "duplicate_interaction_count": len(pairs) - len(set(pairs)),
        "user_count": len(users),
        "item_count": len(items),
        "user_id_min": min(users),
        "user_id_max": max(users),
        "item_id_min": min(items),
        "item_id_max": max(items),
    }


def convert_dataset(source_dir: Path, output_dir: Path, dataset: str) -> Path:
    source_paths = {
        split: source_dir / dataset / f"{split}.txt"
        for split in ("train", "valid", "test")
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing NT-SSM split(s): " + ", ".join(missing))

    splits = {
        split: read_triple_split(path, split)
        for split, path in source_paths.items()
    }
    pair_sets = {split: set(value.pairs) for split, value in splits.items()}
    overlaps = {
        "train_valid": len(pair_sets["train"] & pair_sets["valid"]),
        "train_test": len(pair_sets["train"] & pair_sets["test"]),
        "valid_test": len(pair_sets["valid"] & pair_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(
            f"{dataset}: source splits overlap ({overlaps}); refusing to alter test semantics"
        )

    merged_pairs = splits["train"].pairs + splits["valid"].pairs
    global_pairs = merged_pairs + splits["test"].pairs
    user_ids = [user for user, _ in global_pairs]
    item_ids = [item for _, item in global_pairs]
    user_contiguity = _contiguity(user_ids)
    item_contiguity = _contiguity(item_ids)
    if not user_contiguity["zero_based_contiguous"]:
        raise ValueError(f"{dataset}: global user IDs are not zero-based contiguous")
    if not item_contiguity["zero_based_contiguous"]:
        raise ValueError(f"{dataset}: global item IDs are not zero-based contiguous")

    # All source training users appear before validation is appended in the
    # released data.  Ordered grouping therefore gives stable NR-GCF edge IDs:
    # user order follows train, and validation items follow that user's train
    # items.  Test is grouped independently and never filtered.
    train_grouped = group_pairs(splits["train"].pairs)
    for user_id, item_id in splits["valid"].pairs:
        train_grouped.setdefault(user_id, []).append(item_id)
    test_grouped = group_pairs(splits["test"].pairs)

    destination = output_dir / dataset
    destination_train = destination / "train.txt"
    destination_test = destination / "test.txt"
    write_grouped_pairs(destination_train, train_grouped)
    write_grouped_pairs(destination_test, test_grouped)

    converted_train = read_grouped_pairs(destination_train)
    converted_test = read_grouped_pairs(destination_test)
    if set(converted_train) != set(merged_pairs) or len(converted_train) != len(merged_pairs):
        raise AssertionError(f"{dataset}: converted train is not pair-equivalent")
    if converted_test != splits["test"].pairs:
        raise AssertionError(
            f"{dataset}: converted test does not preserve source interaction order"
        )

    merged_users = {user for user, _ in converted_train}
    merged_items = {item for _, item in converted_train}
    test_users = {user for user, _ in converted_test}
    test_items = {item for _, item in converted_test}
    cold_test_pairs = [
        (user, item)
        for user, item in converted_test
        if user not in merged_users or item not in merged_items
    ]

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_format": "one user_id item_id implicit_value triple per line",
        "destination_format": "one user_id followed by all item_ids per line",
        "id_policy": "preserve released NT-SSM integer IDs without remapping",
        "train_policy": (
            "pair union of source train and valid; within each user, train order "
            "then valid order; no deduplication occurred"
        ),
        "test_policy": "all source test interactions preserved; no filtering or relocation",
        "nrgcf_edge_order": (
            "users follow source train first-occurrence order; within each user, "
            "source train items precede source valid items"
        ),
        "source": {
            split: {
                "path": f"{dataset}/{split}.txt",
                "sha256": sha256_file(value.path),
                **_split_stats(value.pairs),
            }
            for split, value in splits.items()
        },
        "split_overlap_counts": overlaps,
        "converted": {
            "train": {
                "path": destination_train.name,
                "sha256": sha256_file(destination_train),
                **_split_stats(converted_train),
            },
            "test": {
                "path": destination_test.name,
                "sha256": sha256_file(destination_test),
                **_split_stats(converted_test),
            },
        },
        "global_user_ids": user_contiguity,
        "global_item_ids": item_contiguity,
        "test_cold_start": {
            "user_count": len(test_users - merged_users),
            "item_count": len(test_items - merged_items),
            "interaction_count": len(cold_test_pairs),
            "policy": (
                "retained exactly as requested; NR-GCF allocates these item IDs but "
                "they receive no training-edge propagation"
            ),
        },
        "validation": {
            "merged_train_pair_equivalent": True,
            "test_sequence_equivalent": True,
            "numeric_ids_preserved": True,
            "source_duplicate_count": 0,
            "converted_duplicate_count": 0,
        },
    }
    metadata_path = destination / "conversion_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="NT-SSM dataset directory containing lastfm/ and ml-1m/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="NR-GCF data directory to receive grouped datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        metadata_path = convert_dataset(args.source_root, args.output_root, dataset)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_count = metadata["converted"]["train"]["interaction_count"]
        test_count = metadata["converted"]["test"]["interaction_count"]
        cold_items = metadata["test_cold_start"]["item_count"]
        print(
            f"{dataset}: train={train_count}, test={test_count}, "
            f"cold_test_items={cold_items}, metadata={metadata_path}"
        )


if __name__ == "__main__":
    main()
