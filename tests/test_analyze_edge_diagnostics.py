import csv
import gzip
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


CODE_DIR = pathlib.Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

if importlib.util.find_spec("numpy") is not None:
    from analyze_edge_diagnostics import REQUIRED_COLUMNS, analyze  # noqa: E402
else:
    REQUIRED_COLUMNS = []
    analyze = None


class AnalyzeEdgeDiagnosticsTest(unittest.TestCase):
    @unittest.skipIf(analyze is None, "NumPy is unavailable")
    def test_compact_analysis_from_one_gzip_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            diagnostics = root / "edge_diagnostics"
            diagnostics.mkdir()
            table = diagnostics / "edge_diagnostics.csv.gz"
            with gzip.open(table, "wt", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                for edge_id in range(10):
                    noisy = edge_id >= 5
                    structure = 0.9 - edge_id * 0.08
                    writer.writerow({
                        "edge_id": edge_id,
                        "synthetic_is_noisy": noisy,
                        "nr_gcf_removed": edge_id in (8, 9),
                        "normalized_edge_score": edge_id / 10.0,
                        "historical_or_momentum_loss": float(edge_id),
                        "current_edge_loss": edge_id / 20.0,
                        "user_degree_before": 10,
                        "item_degree_before": 10,
                        "min_endpoint_degree": 10,
                        "user_side_structure_mean": structure,
                        "item_side_structure_mean": structure,
                        "bilateral_structure_mean": structure,
                    })
            output = root / "pilot_analysis.json"
            report = analyze(diagnostics, output)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["edge_count"], 10)
            self.assertEqual(saved["synthetic_noisy_count"], 5)
            self.assertAlmostEqual(
                saved["classification_metrics"]["normalized_momentum_loss"]["auroc"],
                1.0,
            )
            self.assertAlmostEqual(
                saved["classification_metrics"]["bilateral_structure"]["auroc"],
                1.0,
            )
            self.assertEqual(saved["filtering"]["removed_noisy_count"], 2)


if __name__ == "__main__":
    unittest.main()
