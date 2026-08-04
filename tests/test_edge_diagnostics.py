import csv
import logging
import math
import os
import sys
import tempfile
import unittest
from unittest import mock


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_DIR, "code")
sys.path.insert(0, CODE_DIR)

import edge_diagnostics as diagnostics
import parse


class EdgeDiagnosticsReferenceTest(unittest.TestCase):
    def setUp(self):
        # A small bipartite graph with one 2x2 biclique and one tail edge.
        self.edges = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]

    def test_stable_edge_identity_and_degree_risk(self):
        rows = diagnostics.compute_degree_connectivity_reference(
            self.edges, min_degree=2
        )
        self.assertEqual([row["edge_id"] for row in rows], list(range(len(self.edges))))
        self.assertEqual(rows[0]["user_degree_before"], 2)
        self.assertEqual(rows[0]["item_degree_before"], 2)
        self.assertFalse(rows[0]["user_becomes_isolated_if_removed"])
        self.assertFalse(rows[0]["item_becomes_isolated_if_removed"])
        self.assertTrue(rows[0]["user_below_min_degree_if_removed"])
        self.assertTrue(rows[0]["item_below_min_degree_if_removed"])
        self.assertAlmostEqual(rows[0]["normalized_degree_product"], 0.5)

        tail = rows[4]
        self.assertTrue(tail["user_becomes_isolated_if_removed"])
        self.assertFalse(tail["item_becomes_isolated_if_removed"])

    def test_exact_leave_one_edge_out_structure(self):
        before = list(self.edges)
        rows = diagnostics.compute_structural_features_reference(self.edges, topk=2)
        self.assertEqual(self.edges, before, "reference diagnostics must not mutate edges")

        edge_00 = rows[0]
        # Item 0 without user 0 contains only user 1.  User 0's other item 1
        # contains users {0,1,2}: cosine = 1/sqrt(3).
        self.assertAlmostEqual(
            edge_00["user_side_structure_mean"], 1.0 / math.sqrt(3.0)
        )
        # User 0 without item 0 contains only item 1.  Item 0's other user 1
        # contains items {0,1}: cosine = 1/sqrt(2).
        self.assertAlmostEqual(
            edge_00["item_side_structure_mean"], 1.0 / math.sqrt(2.0)
        )
        self.assertEqual(edge_00["user_side_valid_neighbor_count"], 1)
        self.assertEqual(edge_00["item_side_valid_neighbor_count"], 1)

        tail = rows[4]
        self.assertIsNone(tail["item_side_structure_mean"])
        self.assertEqual(tail["item_side_valid_neighbor_count"], 0)

    def test_duplicate_edges_keep_distinct_edge_ids(self):
        duplicated = [(0, 0), (0, 0), (1, 0)]
        degree_rows = diagnostics.compute_degree_connectivity_reference(duplicated)
        structure_rows = diagnostics.compute_structural_features_reference(duplicated)
        self.assertEqual([row["edge_id"] for row in degree_rows], [0, 1, 2])
        self.assertEqual([row["edge_id"] for row in structure_rows], [0, 1, 2])
        self.assertEqual(degree_rows[0]["user_degree_before"], 2)
        self.assertEqual(degree_rows[1]["user_degree_before"], 2)

    def test_schema_and_chunked_csv_writer(self):
        schema = diagnostics.diagnostics_schema()
        self.assertEqual(schema["diagnostics_schema_version"], diagnostics.SCHEMA_VERSION)
        schema_names = [field["name"] for field in schema["fields"]]
        self.assertEqual(schema_names, diagnostics.FIELD_NAMES)
        self.assertEqual(len(schema_names), len(set(schema_names)))

        logger = logging.getLogger("edge-diagnostics-test")
        with tempfile.TemporaryDirectory() as directory:
            writer = diagnostics.PartWriter(directory, "csv", logger)
            columns = {}
            for name, dtype, nullable, _ in diagnostics.FIELD_SPECS:
                if nullable:
                    value = None
                elif dtype == "int64":
                    value = 1
                elif dtype == "float64":
                    value = 0.5
                elif dtype == "bool":
                    value = True
                else:
                    value = "test"
                columns[name] = [value, value]
            path = writer.write(columns)
            self.assertTrue(os.path.exists(path))
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], diagnostics.FIELD_NAMES)
            self.assertEqual(len(rows), 3)

    def test_diagnostics_arguments_default_off_and_parse(self):
        with mock.patch.object(sys, "argv", ["NR-GCF.py"]):
            args = parse.parse_args()
        self.assertFalse(args.export_edge_diagnostics)
        self.assertEqual(args.edge_diagnostics_dir, "edge_diagnostics")
        self.assertEqual(args.edge_diagnostics_format, "parquet")
        self.assertEqual(args.edge_diagnostics_structural_mode, "two_hop_countsketch")

        with mock.patch.object(sys, "argv", [
            "NR-GCF.py",
            "--export-edge-diagnostics",
            "--edge-diagnostics-format", "csv",
            "--edge-diagnostics-topk", "7",
            "--edge-diagnostics-chunk-size", "128",
        ]):
            configured = parse.parse_args()
        self.assertTrue(configured.export_edge_diagnostics)
        self.assertEqual(configured.edge_diagnostics_format, "csv")
        self.assertEqual(configured.edge_diagnostics_topk, 7)
        self.assertEqual(configured.edge_diagnostics_chunk_size, 128)


if __name__ == "__main__":
    unittest.main()
