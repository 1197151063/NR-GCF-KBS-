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
    def test_merge_train_valid_and_preserve_test(self):
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
            # Item 4 is deliberately test-only.  It must not be dropped.
            (source / "test.txt").write_text(
                "0 3 1\n1 4 1\n", encoding="utf-8"
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
                ((0, 3), (1, 4)),
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["converted"]["train"]["interaction_count"], 5)
            self.assertEqual(metadata["converted"]["test"]["interaction_count"], 2)
            self.assertEqual(metadata["test_cold_start"]["item_count"], 1)
            self.assertTrue(metadata["validation"]["test_sequence_equivalent"])

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
