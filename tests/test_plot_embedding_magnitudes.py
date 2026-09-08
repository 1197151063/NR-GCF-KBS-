import json
import os
import sys
import tempfile
import unittest


CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from plot_embedding_magnitudes import load_rows


class PlotEmbeddingMagnitudesTest(unittest.TestCase):
    def test_averages_layers_per_epoch(self):
        report = {
            "representation_modulation": {
                "lambda": 0.4,
                "trace": [{
                    "epoch": 1,
                    "layer_magnitudes": [
                        {
                            "before_all_mean_l2": 2.0,
                            "pure_crossnorm_all_mean_l2": 4.0,
                            "after_all_mean_l2": 2.8,
                        },
                        {
                            "before_all_mean_l2": 4.0,
                            "pure_crossnorm_all_mean_l2": 8.0,
                            "after_all_mean_l2": 5.6,
                        },
                    ],
                }],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(report, stream)
            stream.flush()
            rows, blend_weight = load_rows(stream.name)
        self.assertEqual(blend_weight, 0.4)
        self.assertEqual(rows[0]["before_crossnorm"], 3.0)
        self.assertEqual(rows[0]["pure_crossnorm"], 6.0)
        self.assertAlmostEqual(rows[0]["after_crossnorm_blend"], 4.2)


if __name__ == "__main__":
    unittest.main()
