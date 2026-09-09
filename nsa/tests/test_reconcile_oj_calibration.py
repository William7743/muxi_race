"""CPU-only regression checks for scoring, guard rejection and OJ reconciliation."""
import copy
import unittest
from reconcile_oj_calibration import analyze, DEST, qualified, score


class CalibrationTests(unittest.TestCase):
    def test_exact_boundary_floor(self):
        self.assertEqual(score(400, 100), 80)
        self.assertEqual(score(400, 100.0001), 79)
        self.assertEqual(score(305, 140), 68)
        self.assertEqual(score(305, 65), 82)
        self.assertEqual(score(305, 63.722446862011864), 82)
        self.assertEqual(score(305, 62.46), 83)

    def test_reject_stable_but_busy_guard(self):
        row = {'samples_us': {'cupti': [10, 10, 10]}, 'guard_samples': {'cupti': [
            {'before_us': 10, 'after_us': 10, 'reference_ratio': 1.3},
            {'before_us': 10, 'after_us': 10, 'reference_ratio': 1.0},
            {'before_us': 10, 'after_us': 11, 'reference_ratio': 1.0}]}}
        q = qualified(row)
        self.assertEqual(q['accepted'], 1)
        self.assertIsNone(q['local_us'])
        self.assertIsNone(q['guard_ratio'])
        good = copy.deepcopy(row)
        good['guard_samples']['cupti'][0]['reference_ratio'] = 1
        self.assertEqual(qualified(good)['local_us'], 10)

    def test_actual_submissions_are_not_used_to_fit(self):
        report = analyze(DEST/'calibration_097_103_236.jsonl')
        self.assertEqual(report['reference_checks'], 42)
        vals = {v['submission_id']: v for v in report['validations']}
        self.assertFalse(vals[141647]['anchor_self_check'])
        self.assertGreater(vals[141647]['projected_minus_actual'], 1.5)
        pair = next(p for p in report['pair_diagnostics'] if p['a'] == 141647 and p['b'] == 141648)
        self.assertEqual(len(pair['unchanged_paths']), 13)
        self.assertEqual([p['case_id'] for p in pair['changed_paths']], [13])
        self.assertAlmostEqual(pair['unchanged_path_score_delta'], 22/14)
        self.assertAlmostEqual(pair['total_score_delta'], 25/14)


if __name__ == '__main__':
    unittest.main()
