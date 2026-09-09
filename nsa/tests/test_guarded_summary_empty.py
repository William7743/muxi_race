"""Do not turn a pending/empty download into a successful measurement report."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EmptyInput(unittest.TestCase):
    def test_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'pending.jsonl'
            output = Path(directory)/'report.json'
            source.touch()
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_guarded_pairs.py')),
                str(source), '--parent', 'p', '--candidate', 'c', '--output', str(output)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No completed measurement rows', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
