"""The shared loop of the modes that train (src/runner.py), independent of the model."""

from pathlib import Path
import tempfile
import unittest

from sklearn.exceptions import NotFittedError

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.instance import MineInstance
from src.evaluation.tiny_oracle import tiny_mine
from src.models.q_learning import QLearningEstimator
from src.runner import compare_reference, fit_seeds, hyperparameters, seed_row, seeds_summary


def tiny_instance() -> MineInstance:
    mine, rules = tiny_mine()
    return MineInstance(mine, rules, ProblemParameters(), EnvironmentParameters(12, "finite"))


def cheap() -> QLearningEstimator:
    return QLearningEstimator(transitions=300, policy_rollouts=2)


class RunnerTests(unittest.TestCase):
    def test_one_clone_per_seed_and_the_original_stays_unfitted(self):
        instance, estimator = tiny_instance(), cheap()
        runs = fit_seeds(instance, estimator, [4, 7])
        with self.assertRaises(NotFittedError):
            estimator.predict(instance)
        self.assertEqual([run.seed for run in runs], [4, 7])
        self.assertEqual([run.estimator.random_state for run in runs], [4, 7])
        for run in runs:
            self.assertIsNone(run.artifacts)
            self.assertEqual(run.score, run.estimator.score(instance))
            self.assertEqual(run.blocks, len(run.estimator.predict(instance).extractions))
        self.assertEqual(hyperparameters(estimator), {k: v for k, v in estimator.get_params().items() if k != "random_state"})

    def test_artifacts_go_to_one_folder_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = fit_seeds(tiny_instance(), cheap(), [1, 2], Path(directory))
            self.assertEqual([run.artifacts for run in runs], ["seed-1", "seed-2"])
            self.assertTrue(all(any((Path(directory) / run.artifacts).iterdir()) for run in runs))

    def test_rows_summary_and_reference_totals(self):
        runs = fit_seeds(tiny_instance(), cheap(), [0, 1, 2])
        rows = [seed_row(run) for run in runs]
        self.assertEqual([row["score"] for row in rows], [run.score for run in runs])
        self.assertEqual(rows[0]["diagnostics"], runs[0].estimator.diagnostics())
        summary = seeds_summary(runs)
        self.assertEqual((summary["seeds"], summary["score_max"], summary["score_min"]),
                         (3, max(r.score for r in runs), min(r.score for r in runs)))
        reference = runs[0].score
        totals = compare_reference(runs, rows, reference)
        self.assertEqual(totals["reference_npv"], reference)
        self.assertEqual(totals["matches_reference"], sum(run.score == reference for run in runs))
        for key in rows[0]["reference"]:
            self.assertEqual(totals["reference"][key], sum(row["reference"][key] for row in rows))

    def test_compare_reference_must_return_counts(self):
        for counts in ([1], {"x": True}, {"x": -1}, {"x": 0.5}, {1: 1}):
            class Broken(QLearningEstimator):
                def compare_reference(self, reference_npv):
                    return counts

            runs = fit_seeds(tiny_instance(), Broken(transitions=50, policy_rollouts=1), [0])
            with self.subTest(counts=counts), self.assertRaises(RuntimeError):
                compare_reference(runs, [seed_row(runs[0])], 0.0)


if __name__ == "__main__":
    unittest.main()
