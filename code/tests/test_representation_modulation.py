import math
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np


CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from edge_reliability import (
    node_confidence_from_edge_reliability,
    write_training_summary,
)


class _NumpyRow(object):
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)

    def numpy(self):
        return self.values


class _NumpyEdgeIndex(object):
    def __init__(self, users, items):
        self.rows = (_NumpyRow(users), _NumpyRow(items))

    def __getitem__(self, index):
        return self.rows[index]


class RepresentationModulationTest(unittest.TestCase):
    def test_blend_always_records_active_lambda(self):
        with tempfile.TemporaryDirectory() as output_dir:
            write_training_summary(
                output_dir=output_dir,
                mode="hard_structure_momentum",
                requested_epochs=10,
                epochs_completed=10,
                best_epoch=4,
                best_recall=0.1,
                best_ndcg=0.05,
                final_loss=0.2,
                propagation_edge_count=4,
                positive_training_edge_count=4,
                representation_modulation_mode="blend_always",
                representation_modulation_ramp_epochs=0,
                representation_modulation_lambda=0.6,
                representation_modulation_trace=[],
                best_post_filter_epoch=4,
                best_post_filter_recall=0.1,
                best_post_filter_ndcg=0.05,
                early_stopping_patience=20,
                early_stopped=False,
                early_stopping_wait=0,
                filtering_schedule="adaptive",
                configured_filtering_epoch=4,
                actual_filtering_epoch=3,
                adaptive_filtering_trace=[],
            )
            with open(
                    pathlib.Path(output_dir) / "training_summary.json",
                    encoding="utf-8") as stream:
                summary = json.load(stream)
        modulation = summary["representation_modulation"]
        self.assertEqual(modulation["mode"], "blend_always")
        self.assertAlmostEqual(modulation["lambda"], 0.6)
        self.assertIn("Active weight", modulation["lambda_note"])

    def test_retained_edge_reliability_aggregates_to_node_confidence(self):
        edge_index = _NumpyEdgeIndex(
            users=[0, 0, 1, 1, 2],
            items=[0, 1, 1, 2, 2],
        )
        user_confidence, item_confidence = (
            node_confidence_from_edge_reliability(
                edge_index_cpu=edge_index,
                reliability=np.asarray([0.8, 0.2, 0.6, 0.4, math.nan]),
                retained_mask=np.asarray([True, False, True, True, True]),
                num_users=4,
                num_items=4,
            )
        )

        # Removed edge (u0, i1) contributes nothing.  Missing reliability on a
        # retained edge is neutral confidence one.  Isolated nodes have zero
        # weight and cannot influence the global scale estimator.
        np.testing.assert_allclose(
            user_confidence, np.asarray([0.8, 0.5, 1.0, 0.0]), rtol=1e-6
        )
        np.testing.assert_allclose(
            item_confidence, np.asarray([0.8, 0.6, 0.7, 0.0]), rtol=1e-6
        )

    def test_identity_mismatch_is_rejected(self):
        edge_index = _NumpyEdgeIndex(users=[0, 1], items=[0, 1])
        with self.assertRaisesRegex(ValueError, 'identity'):
            node_confidence_from_edge_reliability(
                edge_index_cpu=edge_index,
                reliability=np.asarray([0.5]),
                retained_mask=np.asarray([True, True]),
                num_users=2,
                num_items=2,
            )


if __name__ == '__main__':
    unittest.main()
