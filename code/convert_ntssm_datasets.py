#!/usr/bin/env python3
"""Prepare LastFM and MovieLens splits in the grouped NR-GCF data format.

LastFM is read from NT-SSM triples, with train and validation merged.  MovieLens
is imported from a separately supplied grouped train/test directory.  Both test
splits are made training-closed.  Numeric IDs are preserved, and malformed or
ambiguous source data is rejected.
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
SCHEMA_VERSION = "nrgcf-dataset-conversion-v6"


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


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset: str,
    filter_cold_start: bool = True,
    merge_validation: bool = False,
) -> Path:
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

    training_pairs = (
        splits["train"].pairs + splits["valid"].pairs
        if merge_validation
        else splits["train"].pairs
    )
    global_pairs = (
        splits["train"].pairs
        + splits["valid"].pairs
        + splits["test"].pairs
    )
    user_ids = [user for user, _ in global_pairs]
    item_ids = [item for _, item in global_pairs]
    user_contiguity = _contiguity(user_ids)
    item_contiguity = _contiguity(item_ids)
    if not user_contiguity["zero_based_contiguous"]:
        raise ValueError(f"{dataset}: global user IDs are not zero-based contiguous")
    if not item_contiguity["zero_based_contiguous"]:
        raise ValueError(f"{dataset}: global item IDs are not zero-based contiguous")

    train_grouped = group_pairs(splits["train"].pairs)
    if merge_validation:
        for user_id, item_id in splits["valid"].pairs:
            train_grouped.setdefault(user_id, []).append(item_id)

    training_users = {user for user, _ in training_pairs}
    training_items = {item for _, item in training_pairs}
    cold_test_pairs = tuple(
        (user, item)
        for user, item in splits["test"].pairs
        if user not in training_users or item not in training_items
    )
    if filter_cold_start:
        retained_test_pairs = tuple(
            (user, item)
            for user, item in splits["test"].pairs
            if user in training_users and item in training_items
        )
        filtered_test_pairs = cold_test_pairs
    else:
        retained_test_pairs = splits["test"].pairs
        filtered_test_pairs = ()
    if not retained_test_pairs:
        raise ValueError(f"{dataset}: cold-start filtering removed the entire test split")
    test_grouped = group_pairs(retained_test_pairs)

    destination = output_dir / dataset
    destination_train = destination / "train.txt"
    destination_test = destination / "test.txt"
    write_grouped_pairs(destination_train, train_grouped)
    write_grouped_pairs(destination_test, test_grouped)

    converted_train = read_grouped_pairs(destination_train)
    converted_test = read_grouped_pairs(destination_test)
    if set(converted_train) != set(training_pairs) or len(converted_train) != len(training_pairs):
        raise AssertionError(f"{dataset}: converted train is not pair-equivalent")
    if converted_test != retained_test_pairs:
        raise AssertionError(
            f"{dataset}: converted test does not match the training-closed source test"
        )

    source_test_users = {user for user, _ in splits["test"].pairs}
    source_test_items = {item for _, item in splits["test"].pairs}
    retained_test_users = {user for user, _ in converted_test}
    cold_users = {
        user for user, _ in cold_test_pairs if user not in training_users
    }
    cold_items = {
        item for _, item in cold_test_pairs if item not in training_items
    }
    filtered_due_to_user_only = sum(
        user not in training_users and item in training_items
        for user, item in filtered_test_pairs
    )
    filtered_due_to_item_only = sum(
        user in training_users and item not in training_items
        for user, item in filtered_test_pairs
    )
    filtered_due_to_both = sum(
        user not in training_users and item not in training_items
        for user, item in filtered_test_pairs
    )
    converted_global_pairs = converted_train + converted_test

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_kind": "ntssm_train_valid_test_triples",
        "source_format": "one user_id item_id implicit_value triple per line",
        "destination_format": "one user_id followed by all item_ids per line",
        "id_policy": "preserve released NT-SSM integer IDs without remapping",
        "train_policy": (
            "merge source train and valid; within each user, train order then "
            "valid order"
            if merge_validation
            else "use source train only; source valid is excluded"
        ),
        "validation_policy": "merged_into_train" if merge_validation else "excluded",
        "test_policy": (
            "filter training-unseen user/item endpoints and preserve retained order"
            if filter_cold_start
            else "preserve every source test interaction and its order"
        ),
        "nrgcf_edge_order": (
            "users and items follow source train order"
            if not merge_validation
            else (
                "users follow source train first-occurrence order; within each "
                "user, source train items precede source valid items"
            )
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
        "source_global_user_ids": user_contiguity,
        "source_global_item_ids": item_contiguity,
        "converted_global_user_ids": _contiguity(
            user for user, _ in converted_global_pairs
        ),
        "converted_global_item_ids": _contiguity(
            item for _, item in converted_global_pairs
        ),
        "test_cold_start": {
            "mode": "filter" if filter_cold_start else "retain",
            "definition": (
                "source test pair (u,i) is cold when u or i is absent from the "
                "converted training split"
            ),
            "source_interaction_count": len(splits["test"].pairs),
            "converted_interaction_count": len(converted_test),
            "cold_interaction_count": len(cold_test_pairs),
            "retained_cold_interaction_count": (
                0 if filter_cold_start else len(cold_test_pairs)
            ),
            "filtered_interaction_count": len(filtered_test_pairs),
            "cold_user_count": len(cold_users),
            "cold_item_count": len(cold_items),
            "filtered_due_to_user_only_count": filtered_due_to_user_only,
            "filtered_due_to_item_only_count": filtered_due_to_item_only,
            "filtered_due_to_both_count": filtered_due_to_both,
            "source_test_user_count": len(source_test_users),
            "retained_test_user_count": len(retained_test_users),
            "users_losing_all_test_interactions": len(
                source_test_users - retained_test_users
            ),
            "source_test_item_count": len(source_test_items),
        },
        "validation": {
            "converted_train_pair_equivalent": True,
            "source_validation_merged_into_train": merge_validation,
            "source_validation_excluded": not merge_validation,
            "test_training_closed": all(
                user in training_users and item in training_items
                for user, item in converted_test
            ),
            "source_test_sequence_equivalent": (
                converted_test == splits["test"].pairs
            ),
            "retained_test_sequence_preserved": True,
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


def import_grouped_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset: str = "ml-1m",
    filter_cold_start: bool = True,
) -> Path:
    """Import an already-grouped train/test dataset without ID remapping."""
    source_train_path = source_dir / "train.txt"
    source_test_path = source_dir / "test.txt"
    missing = [
        str(path)
        for path in (source_train_path, source_test_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing grouped MovieLens split(s): " + ", ".join(missing)
        )

    source_train = read_grouped_pairs(source_train_path)
    source_test = read_grouped_pairs(source_test_path)
    if not source_train or not source_test:
        raise ValueError(f"{source_dir}: grouped train/test splits must be non-empty")
    for split_name, pairs in (("train", source_train), ("test", source_test)):
        duplicate_count = len(pairs) - len(set(pairs))
        if duplicate_count:
            raise ValueError(
                f"{source_dir}/{split_name}.txt contains {duplicate_count} "
                "duplicate user-item interactions"
            )
    overlap_count = len(set(source_train) & set(source_test))
    if overlap_count:
        raise ValueError(
            f"{dataset}: grouped train/test overlap by {overlap_count} interactions"
        )

    global_pairs = source_train + source_test
    source_user_contiguity = _contiguity(user for user, _ in global_pairs)
    source_item_contiguity = _contiguity(item for _, item in global_pairs)
    if not source_user_contiguity["zero_based_contiguous"]:
        raise ValueError(f"{dataset}: global user IDs are not zero-based contiguous")
    if not source_item_contiguity["zero_based_contiguous"]:
        raise ValueError(f"{dataset}: global item IDs are not zero-based contiguous")

    training_users = {user for user, _ in source_train}
    training_items = {item for _, item in source_train}
    cold_test_pairs = tuple(
        (user, item)
        for user, item in source_test
        if user not in training_users or item not in training_items
    )
    retained_test = (
        tuple(
            (user, item)
            for user, item in source_test
            if user in training_users and item in training_items
        )
        if filter_cold_start
        else source_test
    )
    if not retained_test:
        raise ValueError(f"{dataset}: cold-start filtering removed the entire test split")

    destination = output_dir / dataset
    destination_train = destination / "train.txt"
    destination_test = destination / "test.txt"
    write_grouped_pairs(destination_train, group_pairs(source_train))
    write_grouped_pairs(destination_test, group_pairs(retained_test))
    converted_train = read_grouped_pairs(destination_train)
    converted_test = read_grouped_pairs(destination_test)
    if converted_train != source_train:
        raise AssertionError(f"{dataset}: grouped train sequence changed during import")
    if converted_test != retained_test:
        raise AssertionError(f"{dataset}: grouped test sequence changed during import")

    source_test_users = {user for user, _ in source_test}
    retained_test_users = {user for user, _ in converted_test}
    cold_users = {
        user for user, _ in cold_test_pairs if user not in training_users
    }
    cold_items = {
        item for _, item in cold_test_pairs if item not in training_items
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_kind": "latest_user_supplied_grouped_train_test",
        "source_format": "one user_id followed by all item_ids per line",
        "destination_format": "one user_id followed by all item_ids per line",
        "id_policy": "preserve supplied integer IDs without remapping",
        "train_policy": "use supplied grouped train exactly; no validation split",
        "validation_policy": "not_present",
        "test_policy": "filter train-unseen endpoints and preserve retained order",
        "source": {
            "train": {
                "path": "ml_data/train.txt",
                "sha256": sha256_file(source_train_path),
                **_split_stats(source_train),
            },
            "test": {
                "path": "ml_data/test.txt",
                "sha256": sha256_file(source_test_path),
                **_split_stats(source_test),
            },
        },
        "split_overlap_counts": {"train_test": overlap_count},
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
        "source_global_user_ids": source_user_contiguity,
        "source_global_item_ids": source_item_contiguity,
        "converted_global_user_ids": _contiguity(
            user for user, _ in converted_train + converted_test
        ),
        "converted_global_item_ids": _contiguity(
            item for _, item in converted_train + converted_test
        ),
        "test_cold_start": {
            "mode": "filter" if filter_cold_start else "retain",
            "definition": (
                "source test pair (u,i) is cold when u or i is absent from "
                "supplied train"
            ),
            "source_interaction_count": len(source_test),
            "converted_interaction_count": len(converted_test),
            "cold_interaction_count": len(cold_test_pairs),
            "retained_cold_interaction_count": (
                0 if filter_cold_start else len(cold_test_pairs)
            ),
            "filtered_interaction_count": (
                len(cold_test_pairs) if filter_cold_start else 0
            ),
            "cold_user_count": len(cold_users),
            "cold_item_count": len(cold_items),
            "source_test_user_count": len(source_test_users),
            "retained_test_user_count": len(retained_test_users),
            "users_losing_all_test_interactions": len(
                source_test_users - retained_test_users
            ),
        },
        "validation": {
            "converted_train_sequence_equivalent": True,
            "retained_test_sequence_preserved": True,
            "test_training_closed": all(
                user in training_users and item in training_items
                for user, item in converted_test
            ),
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
        help="NT-SSM dataset directory containing lastfm/ triple splits",
    )
    parser.add_argument(
        "--movielens-source-root",
        type=Path,
        help="user-supplied grouped MovieLens directory containing train.txt/test.txt",
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
    parser.add_argument(
        "--retain-cold-start",
        action="store_true",
        help=(
            "diagnostic override: retain test interactions whose user or item is "
            "absent from merged train; default filters them"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        if dataset == "ml-1m":
            if args.movielens_source_root is None:
                raise SystemExit(
                    "--movielens-source-root is required when converting ml-1m"
                )
            metadata_path = import_grouped_dataset(
                args.movielens_source_root,
                args.output_root,
                dataset,
                filter_cold_start=not args.retain_cold_start,
            )
        else:
            metadata_path = convert_dataset(
                args.source_root,
                args.output_root,
                dataset,
                filter_cold_start=not args.retain_cold_start,
                merge_validation=True,
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_count = metadata["converted"]["train"]["interaction_count"]
        test_count = metadata["converted"]["test"]["interaction_count"]
        cold = metadata["test_cold_start"]
        print(
            f"{dataset}: train={train_count}, test={test_count}, "
            f"cold_test_interactions={cold['cold_interaction_count']}, "
            f"cold_mode={cold['mode']}, metadata={metadata_path}"
        )


if __name__ == "__main__":
    main()
