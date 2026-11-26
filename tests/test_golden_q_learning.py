"""Q-learning must give identical results for the same seed across refactors.

tests/golden/q_learning.json was generated on main BEFORE the pipeline refactor
(see tests/golden_support.py). A mismatch means a change altered the RNG consumption,
the order of updates or the dynamics of the environment.

The fingerprints depend on the Python version (random's algorithms and float formatting),
so the golden check runs on the project's Python, 3.12 (environment.yml, CI), and is
skipped on any other.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # golden_support, however the test is launched

from configs import PROJECT_ROOT  # noqa: E402
from golden_support import GOLDEN, SEEDS, cases, fingerprint, fit_case  # noqa: E402

PROJECT_PYTHON = (3, 12)


class GoldenQLearningTests(unittest.TestCase):
    @unittest.skipUnless(sys.version_info[:2] == PROJECT_PYTHON, "el golden se generó con Python 3.12")
    def test_every_case_matches_the_golden_fingerprint(self):
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(set(golden), set(cases()))
        for name in golden:
            for seed in SEEDS:
                with self.subTest(case=name, seed=seed):
                    self.assertEqual(fingerprint(fit_case(name, seed)), golden[name][str(seed)])

    def test_results_do_not_depend_on_the_string_hash_seed(self):
        """str hashes change per process (PYTHONHASHSEED); iterating a set of actions would
        make a seed irreproducible across runs. A single process never notices it. Checked
        for every model whose fit could iterate hashed containers."""
        for model in ("q_learning", "dyna_q", "prioritized_sweeping", "beam_search"):
            outputs = []
            for hash_seed in ("1", "2"):
                env = {**os.environ, "PYTHONHASHSEED": hash_seed,
                       "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), str(Path(__file__).resolve().parent)])}
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("golden_support.py")),
                                         "--case", "grid-physical-external-multi", "1", model],
                                        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
                with self.subTest(model=model, hash_seed=hash_seed):
                    self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(result.stdout)
            with self.subTest(model=model):
                self.assertTrue(outputs[0])
                self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
