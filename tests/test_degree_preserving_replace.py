import csv
import json
import pathlib
import sys
import tempfile
import unittest
from collections import Counter


CODE_DIR = pathlib.Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from generate_degree_preserving_replace import (  # noqa: E402
    PROTOCOL_NAME,
    _read_train,
    generate_degree_preserving_replace,
)


SMALL_TRAIN = """\
0 0 1
1 1 2
2 2 3
3 3 4
4 4 5
5 5 0
"""


class DegreePreservingReplaceTest(unittest.TestCase):
    def _generate(self, root, ratio=0.5, seed=7, suffix=""):
        root = pathlib.Path(root)
        clean = root / "train.txt"
        if not clean.exists():
            clean.write_text(SMALL_TRAIN, encoding="utf-8")
        output = root / ("variant%s.txt" % suffix)
        labels = root / ("labels%s.csv" % suffix)
        generation = root / ("generation%s.json" % suffix)
        validation = root / ("validation%s.json" % suffix)
        result = generate_degree_preserving_replace(
            clean_train=clean,
            requested_ratio=ratio,
            seed=seed,
            output_train=output,
            labels_path=labels,
            generation_metadata_path=generation,
            validation_path=validation,
        )
        return clean, output, labels, generation, validation, result

    def test_swap_preserves_every_endpoint_degree_and_edge_position(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, output, labels, generation, validation, result = self._generate(temp)
            _, clean_edges = _read_train(clean)
            _, variant_edges = _read_train(output)
            self.assertEqual(len(clean_edges), len(variant_edges))
            self.assertEqual(
                Counter(user for user, _ in clean_edges),
                Counter(user for user, _ in variant_edges),
            )
            self.assertEqual(
                Counter(item for _, item in clean_edges),
                Counter(item for _, item in variant_edges),
            )
            self.assertEqual(result["synthetic_noisy_edge_count"], 6)
            self.assertTrue(result["all_user_degrees_preserved"])
            self.assertTrue(result["all_item_degrees_preserved"])
            self.assertEqual(len(set(variant_edges)), len(variant_edges))

            with labels.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["edge_id"]) for row in rows], list(range(12)))
            noisy_rows = [row for row in rows if row["synthetic_is_noisy"] == "True"]
            self.assertEqual(len(noisy_rows), 6)
            clean_edge_set = set(clean_edges)
            for row in noisy_rows:
                observed = (int(row["user_id_internal"]), int(row["item_id_internal"]))
                self.assertNotIn(observed, clean_edge_set)
                self.assertEqual(row["synthetic_noise_type"], PROTOCOL_NAME)
                edge_id = int(row["edge_id"])
                self.assertEqual(
                    int(row["original_user_id_internal"]), clean_edges[edge_id][0]
                )
                self.assertEqual(
                    int(row["original_item_id_internal"]), clean_edges[edge_id][1]
                )

            generation_data = json.loads(generation.read_text(encoding="utf-8"))
            validation_data = json.loads(validation.read_text(encoding="utf-8"))
            self.assertFalse(generation_data["test_data_read"])
            self.assertEqual(validation_data, result)

    def test_same_seed_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            first = self._generate(temp, suffix="_a")
            second = self._generate(temp, suffix="_b")
            self.assertEqual(first[1].read_bytes(), second[1].read_bytes())
            self.assertEqual(first[2].read_bytes(), second[2].read_bytes())
            self.assertEqual(first[3].read_bytes(), second[3].read_bytes())
            self.assertEqual(first[4].read_bytes(), second[4].read_bytes())

    def test_odd_requested_count_is_adjusted_to_complete_swap_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, generation, _, result = self._generate(temp, ratio=0.1)
            details = json.loads(generation.read_text(encoding="utf-8"))
            self.assertEqual(details["requested_replacement_count_before_even_adjustment"], 1)
            self.assertEqual(details["replacement_count"], 2)
            self.assertEqual(details["swap_pair_count"], 1)
            self.assertAlmostEqual(result["actual_noise_ratio"], 2.0 / 12.0)

    def test_invalid_ratio_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                self._generate(temp, ratio=1.1)


if __name__ == "__main__":
    unittest.main()
