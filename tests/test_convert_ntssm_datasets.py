import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
import sys

sys.path.insert(0, str(CODE_ROOT))

from convert_ntssm_datasets import convert_dataset, read_grouped_pairs


class ConvertNtssmDatasetsTest(unittest.TestCase):
    def test_merge_train_valid_and_preserve_cold_test_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "lastfm"
            destination = root / "destination"
            source.mkdir(parents=True)
            (source / "train.txt").write_text(
                "0 0 1\n0 1 1\n1 2 1\n", encoding="utf-8"
            )
            (source / "valid.txt").write_text(
                "0 2 1\n1 3 1\n", encoding="utf-8"
            )
            # Item 4 and user 2 are deliberately training-unseen.
            (source / "test.txt").write_text(
                "0 3 1\n1 4 1\n2 0 1\n", encoding="utf-8"
            )

            metadata_path = convert_dataset(
                root / "source", destination, "lastfm"
            )

            self.assertEqual(
                read_grouped_pairs(destination / "lastfm" / "train.txt"),
                ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3)),
            )
            self.assertEqual(
                read_grouped_pairs(destination / "lastfm" / "test.txt"),
                ((0, 3), (1, 4), (2, 0)),
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["converted"]["train"]["interaction_count"], 5)
            self.assertEqual(metadata["converted"]["test"]["interaction_count"], 3)
            cold = metadata["test_cold_start"]
            self.assertEqual(cold["mode"], "retain")
            self.assertEqual(cold["cold_interaction_count"], 2)
            self.assertEqual(cold["retained_cold_interaction_count"], 2)
            self.assertEqual(cold["filtered_interaction_count"], 0)
            self.assertEqual(cold["cold_user_count"], 1)
            self.assertEqual(cold["cold_item_count"], 1)
            self.assertFalse(metadata["validation"]["test_training_closed"])
            self.assertTrue(
                metadata["validation"]["source_test_sequence_equivalent"]
            )

            filtered_destination = root / "filtered_destination"
            filtered_metadata_path = convert_dataset(
                root / "source",
                filtered_destination,
                "lastfm",
                filter_cold_start=True,
            )
            self.assertEqual(
                read_grouped_pairs(filtered_destination / "lastfm" / "test.txt"),
                ((0, 3),),
            )
            filtered_metadata = json.loads(
                filtered_metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(filtered_metadata["test_cold_start"]["mode"], "filter")
            self.assertEqual(
                filtered_metadata["test_cold_start"]["filtered_interaction_count"],
                2,
            )
            self.assertTrue(filtered_metadata["validation"]["test_training_closed"])

    def test_duplicate_source_interaction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "lastfm"
            source.mkdir(parents=True)
            (source / "train.txt").write_text("0 0 1\n0 0 1\n", encoding="utf-8")
            (source / "valid.txt").write_text("0 1 1\n", encoding="utf-8")
            (source / "test.txt").write_text("0 2 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                convert_dataset(root / "source", root / "destination", "lastfm")

    def test_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "lastfm"
            source.mkdir(parents=True)
            (source / "train.txt").write_text("0 0 1\n", encoding="utf-8")
            (source / "valid.txt").write_text("0 1 1\n", encoding="utf-8")
            (source / "test.txt").write_text("0 1 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overlap"):
                convert_dataset(root / "source", root / "destination", "lastfm")


if __name__ == "__main__":
    unittest.main()
