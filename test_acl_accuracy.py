from pathlib import Path
import subprocess
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class ACLAccuracyTests(unittest.TestCase):
    def test_gemma_factorial_accuracy_and_response_bias(self):
        frame = pd.read_json(ROOT / "data/gemma/gemma_harmonized_robustness_v1.jsonl", lines=True)
        accuracy = frame.groupby("factorial_code").correct.mean().to_dict()
        expected = {"CC": 61/64, "CE": 14/64, "FC": 57/64, "FE": 56/64}
        for condition, value in expected.items():
            self.assertAlmostEqual(accuracy[condition], value)
        ce = frame[frame.factorial_code == "CE"]
        self.assertEqual(len(ce), 64)
        self.assertEqual(int((ce.predicted_digit == 1).sum()), 62)
        self.assertEqual(int(frame.trial_index.nunique()), 16)

    def test_existing_acl_accuracy_script_runs(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "analysis/acl_accuracy_reanalysis.py"), str(ROOT)],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("Gemma 3 4B", completed.stdout)
        self.assertIn("CE-CC: -73.44 pp [-93.75, -51.56]", completed.stdout)
        self.assertIn("FE-CE: +65.62 pp [+35.94, +89.06]", completed.stdout)
        self.assertIn("FE-CC: -7.81 pp [-18.75, +0.00]", completed.stdout)

if __name__ == "__main__":
    unittest.main()
