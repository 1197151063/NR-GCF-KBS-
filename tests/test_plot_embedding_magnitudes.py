import json
import os
import sys
import tempfile
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from plot_embedding_magnitudes import compare_rows, load_run


class PlotEmbeddingMagnitudesTest(unittest.TestCase):
    def _write_run(self, user_values, item_values):
        report = {
            "representation_modulation": {
                "trace": [{
                    "epoch": 1,
                    "layer_magnitudes": [
                        {
                            "after_user_mean_l2": user_values[0],
                            "after_item_mean_l2": item_values[0],
                        },
                        {
                            "after_user_mean_l2": user_values[1],
                            "after_item_mean_l2": item_values[1],
                        },
                    ],
                }],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(report, stream)
            stream.flush()
            return load_run(stream.name)

    def test_compares_two_runs_and_averages_layers(self):
        lightgcn = self._write_run([2.0, 4.0], [6.0, 8.0])
        norm = self._write_run([2.8, 5.6], [5.0, 7.0])
        row = compare_rows(lightgcn, norm)[0]
        self.assertEqual(row["user"], 3.0)
        self.assertAlmostEqual(row["user_norm"], 4.2)
        self.assertEqual(row["item"], 7.0)
        self.assertEqual(row["item_norm"], 6.0)


if __name__ == "__main__":
    unittest.main()
