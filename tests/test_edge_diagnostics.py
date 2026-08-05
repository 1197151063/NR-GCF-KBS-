import csv
import json
import logging
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
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

    @unittest.skipIf(diagnostics.torch is None, "PyTorch is unavailable")
    def test_minhash_structure_tracks_exact_small_graph(self):
        edge_index = diagnostics.torch.tensor(self.edges, dtype=diagnostics.torch.long).t()
        engine = diagnostics.TwoHopMinHash(
            edge_index=edge_index,
            num_users=3,
            num_items=2,
            topk=2,
            structural_mode="two_hop_minhash",
        )
        approximate = engine.compute_chunk(0, len(self.edges))
        exact = diagnostics.compute_structural_features_reference(self.edges, topk=2)
        self.assertAlmostEqual(
            float(approximate["user_side_structure_mean"][0]),
            exact[0]["user_side_structure_mean"],
            delta=0.15,
        )
        self.assertAlmostEqual(
            float(approximate["item_side_structure_mean"][0]),
            exact[0]["item_side_structure_mean"],
            delta=0.15,
        )
        self.assertTrue(
            math.isnan(float(approximate["item_side_structure_mean"][4]))
        )

    @unittest.skipIf(diagnostics.torch is None, "PyTorch is unavailable")
    def test_minhash_does_not_turn_degree_into_overlap(self):
        # Both endpoints have degree two, but after removing (0, 0) the
        # candidate and comparison neighborhoods are disjoint on both sides.
        edges = [(0, 0), (0, 1), (1, 0), (2, 1)]
        edge_index = diagnostics.torch.tensor(edges, dtype=diagnostics.torch.long).t()
        engine = diagnostics.TwoHopMinHash(
            edge_index=edge_index,
            num_users=3,
            num_items=2,
            topk=2,
            structural_mode="two_hop_minhash",
        )
        result = engine.compute_chunk(0, len(edges))
        self.assertEqual(float(result["user_side_structure_mean"][0]), 0.0)
        self.assertEqual(float(result["item_side_structure_mean"][0]), 0.0)

    @unittest.skipIf(diagnostics.torch is None, "PyTorch is unavailable")
    def test_minhash_has_low_error_on_deterministic_small_graph(self):
        edges = [
            (user, item)
            for user in range(20)
            for item in range(15)
            if ((user * 17 + item * 11 + user * item * 3) % 13) < 4
        ]
        edge_index = diagnostics.torch.tensor(
            edges, dtype=diagnostics.torch.long
        ).t()
        engine = diagnostics.TwoHopMinHash(
            edge_index=edge_index,
            num_users=20,
            num_items=15,
            topk=10,
            structural_mode="two_hop_minhash",
        )
        approximate = engine.compute_chunk(0, len(edges))
        exact = diagnostics.compute_structural_features_reference(edges, topk=10)
        for side in ("user", "item"):
            errors = []
            field = "%s_side_structure_mean" % side
            for edge_id, row in enumerate(exact):
                expected = row[field]
                observed = float(approximate[field][edge_id])
                if expected is not None and math.isfinite(observed):
                    errors.append(abs(expected - observed))
            self.assertLess(sum(errors) / len(errors), 0.05)

    def test_raw_id_mapping_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "user_list.txt")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("org_id remap_id\nraw-user-b 1\nraw-user-a 0\n")
            mapping, error = diagnostics.load_raw_id_mapping(path, expected_count=2)
        self.assertIsNone(error)
        self.assertEqual(mapping, ["raw-user-a", "raw-user-b"])

    def test_synthetic_label_reader_validates_stable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "labels.csv")
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "edge_id", "user_id_internal", "item_id_internal",
                    "is_original_observed_edge", "synthetic_is_noisy",
                    "synthetic_noise_type",
                ])
                writer.writerow([0, 2, 3, True, False, ""])
                writer.writerow([1, 4, 5, False, True, "toy_noise"])
            reader = diagnostics.SyntheticLabelReader(path)
            chunk = reader.read_chunk(0, 2, [2, 4], [3, 5], [False, True])
            reader.verify_complete()
            reader.close()
        self.assertEqual(chunk["synthetic_is_noisy"], [False, True])
        self.assertEqual(reader.noisy_removed_count, 1)

    @unittest.skipIf(diagnostics.torch is None, "PyTorch is unavailable")
    def test_small_export_uses_raw_mappings_labels_and_v3_metadata(self):
        torch = diagnostics.torch
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = os.path.join(directory, "data", "toy")
            output_dir = os.path.join(directory, "run", "edge_diagnostics")
            os.makedirs(dataset_dir)
            with open(os.path.join(dataset_dir, "user_list.txt"), "w") as stream:
                stream.write("org_id remap_id\nu0-raw 0\nu1-raw 1\nu2-raw 2\n")
            with open(os.path.join(dataset_dir, "item_list.txt"), "w") as stream:
                stream.write("org_id remap_id\ni0-raw 0\ni1-raw 1\n")

            edge_index = torch.tensor(self.edges, dtype=torch.long).t()
            labels_path = os.path.join(directory, "labels.csv")
            with open(labels_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "edge_id", "user_id_internal", "item_id_internal",
                    "is_original_observed_edge", "synthetic_is_noisy",
                    "synthetic_noise_type",
                ])
                for edge_id, (user, item) in enumerate(self.edges):
                    noisy = edge_id == len(self.edges) - 1
                    writer.writerow([
                        edge_id, user, item, not noisy, noisy,
                        "toy_noise" if noisy else "",
                    ])
            validation_path = os.path.join(directory, "noise_validation.json")
            with open(validation_path, "w", encoding="utf-8") as stream:
                json.dump({"actual_noise_ratio": 1.0 / len(self.edges)}, stream)
            history = diagnostics.EdgeLossHistory(len(self.edges))
            edge_ids = torch.arange(len(self.edges))
            history.observe(edge_ids, torch.linspace(0.1, 0.5, len(self.edges)))
            normalized = torch.linspace(0.1, 0.9, len(self.edges))
            post_threshold = normalized.clone()
            post_threshold[post_threshold > 0.8] = 0
            retained = post_threshold > 0
            args = SimpleNamespace(
                dataset="toy",
                seed=7,
                requested_noise_ratio=0.0,
                edge_diagnostics_chunk_size=3,
                edge_diagnostics_topk=2,
                edge_diagnostics_min_degree=2,
                edge_diagnostics_structural_mode="two_hop_minhash",
                edge_diagnostics_format="csv",
                edge_diagnostics_labels_file=labels_path,
                edge_diagnostics_noise_validation_file=validation_path,
            )
            guard_model = torch.nn.Linear(1, 1)
            guard = diagnostics.DiagnosticsInvarianceGuard(
                guard_model,
                {
                    "edge_index": edge_index,
                    "normalized": normalized,
                    "post_threshold": post_threshold,
                    "retained": retained,
                },
            )
            exporter = diagnostics.EdgeDiagnosticsExporter(
                args=args,
                model_config={"beta": 0.8},
                output_dir=output_dir,
                repo_dir=directory,
            )
            exporter.export(
                edge_index=edge_index,
                num_users=3,
                num_items=2,
                history=history,
                raw_momentum=torch.linspace(0.0, 1.0, len(self.edges)),
                normalized_score=normalized,
                post_threshold_score=post_threshold,
                retained_mask=retained,
                filtering_epoch=15,
                warmup_epoch_count=1,
                threshold=0.8,
            )
            invariance = guard.verify()
            with open(os.path.join(output_dir, "edge_diagnostics.csv"), newline="") as stream:
                first = next(csv.DictReader(stream))
            with open(os.path.join(output_dir, "metadata.json")) as stream:
                metadata = json.load(stream)
        self.assertEqual(first["user_id_raw"], "u0-raw")
        self.assertEqual(first["item_id_raw"], "i0-raw")
        self.assertEqual(metadata["diagnostics_schema_version"], diagnostics.SCHEMA_VERSION)
        self.assertTrue(metadata["seed_is_applied_by_current_nrgcf_code"])
        self.assertEqual(metadata["structural_signature_dim"], diagnostics.MINHASH_DIM)
        self.assertTrue(metadata["synthetic_labels_available"])
        self.assertAlmostEqual(metadata["actual_noise_ratio"], 1.0 / len(self.edges))
        self.assertTrue(invariance["passed"])

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
        fields = dict((field["name"], field) for field in schema["fields"])
        self.assertEqual(fields["user_id_raw"]["dtype"], "string")
        self.assertEqual(fields["item_id_raw"]["dtype"], "string")

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
            writer.write(columns)
            writer.close()
            self.assertTrue(os.path.exists(path))
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], diagnostics.FIELD_NAMES)
            self.assertEqual(len(rows), 5)

        with tempfile.TemporaryDirectory() as directory:
            writer = diagnostics.PartWriter(directory, "csv_gzip", logger)
            writer.write(columns)
            writer.close()
            self.assertTrue(os.path.exists(os.path.join(directory, "edge_diagnostics.csv.gz")))

    def test_diagnostics_arguments_default_off_and_parse(self):
        with mock.patch.object(sys, "argv", ["NR-GCF.py"]):
            args = parse.parse_args()
        self.assertFalse(args.export_edge_diagnostics)
        self.assertEqual(args.edge_diagnostics_dir, "edge_diagnostics")
        self.assertEqual(args.edge_diagnostics_format, "parquet")
        self.assertEqual(args.edge_diagnostics_structural_mode, "two_hop_minhash")
        self.assertEqual(args.edge_diagnostics_chunk_size, 8192)
        self.assertIsNone(args.edge_diagnostics_labels_file)
        self.assertEqual(args.edge_filter_mode, "current")

        with mock.patch.object(sys, "argv", [
            "NR-GCF.py",
            "--export-edge-diagnostics",
            "--edge-diagnostics-format", "csv",
            "--edge-diagnostics-topk", "7",
            "--edge-diagnostics-chunk-size", "128",
            "--edge-filter-mode", "hard_structure_momentum",
            "--export-edge-reliability-summary",
            "--edge-reliability-filtering-epoch", "20",
            "--edge-reliability-momentum-decay", "0.85",
            "--edge-reliability-min-weight", "0.2",
        ]):
            configured = parse.parse_args()
        self.assertTrue(configured.export_edge_diagnostics)
        self.assertEqual(configured.edge_diagnostics_format, "csv")
        self.assertEqual(configured.edge_diagnostics_topk, 7)
        self.assertEqual(configured.edge_diagnostics_chunk_size, 128)
        self.assertEqual(configured.edge_filter_mode, "hard_structure_momentum")
        self.assertTrue(configured.export_edge_reliability_summary)
        self.assertEqual(configured.edge_reliability_filtering_epoch, 20)
        self.assertEqual(configured.edge_reliability_momentum_decay, 0.85)
        self.assertEqual(configured.edge_reliability_min_weight, 0.2)


if __name__ == "__main__":
    unittest.main()
