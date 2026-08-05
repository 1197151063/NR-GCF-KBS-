import math
import pathlib
import sys
import unittest

import numpy as np


CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from edge_reliability import node_confidence_from_edge_reliability


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
