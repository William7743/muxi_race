"""CPU-only diagnostics tests: extract pure functions without importing GPU modules."""
import ast
import contextlib
import io
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "xpuoj_data" / "remote_stage_ab.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
FUNCTIONS = [
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in ("compute_paired_stats", "print_summary")
]
assert len(FUNCTIONS) == 2
NAMESPACE = {"statistics": statistics, "CompiledCandidate": SimpleNamespace}
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
compute_paired_stats = NAMESPACE["compute_paired_stats"]
print_summary = NAMESPACE["print_summary"]


class PairedStatsTests(unittest.TestCase):
    def test_v724_actual_samples(self):
        baseline = [4.681216, 4.698368, 4.656640, 4.689408]
        candidate = [4.691712, 4.708352, 4.655616, 4.661248]
        result = compute_paired_stats(baseline, candidate)
        for observed, expected in zip(result["delta_us"], [10.496, 9.984, -1.024, -28.160]):
            self.assertAlmostEqual(observed, expected, places=9)
        self.assertAlmostEqual(result["median_delta_us"], 4.480, places=9)
        self.assertAlmostEqual(result["mean_delta_us"], -2.176, places=9)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (2, 0, 2))

    def test_identical_constant_samples(self):
        result = compute_paired_stats([4.5] * 4, [4.5] * 4)
        self.assertEqual(result["delta_us"], [0.0] * 4)
        self.assertEqual(result["median_delta_us"], 0.0)
        self.assertEqual(result["mean_delta_us"], 0.0)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (0, 4, 0))

    def test_single_sample(self):
        result = compute_paired_stats([4.0], [3.75])
        self.assertEqual(result["delta_us"], [-250.0])
        self.assertEqual(result["median_delta_us"], -250.0)
        self.assertEqual(result["mean_delta_us"], -250.0)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (1, 0, 0))

    def test_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            compute_paired_stats([1.0, 2.0], [1.0])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            compute_paired_stats([], [1.0])

    def test_empty_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compute_paired_stats([], [])

    def test_win_tie_loss_use_strict_numeric_comparison(self):
        result = compute_paired_stats([1.0, 1.0, 1.0], [1.0 - 1e-12, 1.0, 1.0 + 1e-12])
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (1, 1, 1))

    def test_original_summary_is_preserved_before_appended_pairs(self):
        candidates = [SimpleNamespace(index=0, label="c0:base"), SimpleNamespace(index=1, label="c1:probe")]
        samples = {0: [4.0], 1: [3.75]}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(candidates, samples)
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[:4], [
            "summary_begin",
            "summary candidate=c0:base median_ms=4.000000 mean_ms=4.000000 stdev_ms=0.000000 "
            "min_ms=4.000000 max_ms=4.000000 speedup_vs_c0=+0.000% samples_ms=[4.000000]",
            "summary candidate=c1:probe median_ms=3.750000 mean_ms=3.750000 stdev_ms=0.000000 "
            "min_ms=3.750000 max_ms=3.750000 speedup_vs_c0=+6.667% samples_ms=[3.750000]",
            "summary_end",
        ])
        self.assertEqual(len(lines), 6)
        self.assertIn("wins=0 ties=1 losses=0 delta_us=[+0.000]", lines[4])
        self.assertIn("median_delta_us=-250.000 mean_delta_us=-250.000", lines[5])
        self.assertIn("wins=1 ties=0 losses=0 delta_us=[-250.000]", lines[5])


if __name__ == "__main__":
    unittest.main()
