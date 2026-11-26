"""The shared loop of the modes that train (src/runner.py), independent of the model."""

from pathlib import Path
from statistics import mean
import tempfile
import unittest

from sklearn.exceptions import NotFittedError

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.instance import MineInstance
from src.evaluation.tiny_oracle import tiny_mine
from src.models.q_learning import QLearningEstimator
from src.runner import compare_reference, cost, cost_to_reach, fit_seeds, gap_percent, hyperparameters, mean_interval, seed_row, seeds_summary


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
        self.assertEqual(summary["cost_unit"], "transitions")
        self.assertIsNotNone(summary["score_std"])
        low, high = summary["score_ci95"]
        self.assertLessEqual(low, summary["score_mean"])
        self.assertGreaterEqual(high, summary["score_mean"])
        for run, row in zip(runs, rows):
            self.assertEqual(row["cost"]["unit"], "transitions")
            self.assertEqual(row["cost"]["spent"], {"transitions": 300, "simulator_calls": 300, "updates": 300})
            self.assertEqual(row["cost"]["to_best"], run.estimator.trace_[-1][0])
        self.assertEqual(summary["cost_spent_mean"], 300)
        self.assertEqual(summary["cost_to_best_mean"], mean(r["cost"]["to_best"]["transitions"] for r in rows))
        reference = runs[0].score
        totals = compare_reference(runs, rows, reference)
        self.assertEqual(totals["reference_npv"], reference)
        self.assertEqual(totals["matches_reference"], sum(run.score == reference for run in runs))
        self.assertEqual(rows[0]["gap_percent"], 0.0)
        self.assertEqual(rows[0]["cost_to_reference"], rows[0]["cost"]["to_best"])
        self.assertEqual(totals["reached_reference"], sum(r["cost_to_reference"] is not None for r in rows))
        self.assertEqual(totals["cost_to_reference_by_seed"],
                         [None if r["cost_to_reference"] is None else r["cost_to_reference"]["transitions"] for r in rows])
        for key in rows[0]["reference"]:
            self.assertEqual(totals["reference"][key], sum(row["reference"][key] for row in rows))

    def test_statistics_gap_and_cost_to_reach(self):
        self.assertEqual(mean_interval([5.0]), {"mean": 5.0, "std": None, "ci95": None})
        result = mean_interval([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["std"], 1.0)
        half = 4.302652729911275 / 3 ** 0.5  # t(0.975, 2) * s / sqrt(n)
        self.assertAlmostEqual(result["ci95"][0], 2 - half)
        self.assertAlmostEqual(result["ci95"][1], 2 + half)
        self.assertEqual(gap_percent(90, 100), 10)
        self.assertEqual(gap_percent(110, 100), -10)
        self.assertEqual(gap_percent(-12, -10), 20)  # below a negative reference is still a positive gap
        self.assertIsNone(gap_percent(5, 0))
        trace = (({"transitions": 1, "updates": 1}, 3.0), ({"transitions": 7, "updates": 9}, 10.0),
                 ({"transitions": 20, "updates": 30}, 12.0))
        self.assertEqual([cost_to_reach(trace, target) for target in (2, 10, 11)], [c for c, _ in trace])
        self.assertIsNone(cost_to_reach(trace, 13))
        self.assertEqual(mean_interval([]), {"mean": None, "std": None, "ci95": None})

    def test_cost_checks_the_trace_of_the_model(self):
        run = fit_seeds(tiny_instance(), cheap(), [0])[0]
        model, last = run.estimator, run.estimator.trace_[-1]
        broken = {"empty": (), "no unit": (({"updates": 1}, last[1]),), "wrong end": (({"transitions": 1}, last[1] + 1),),
                  "cost goes down": (({"transitions": 5}, last[1] - 1), ({"transitions": 2}, last[1])),
                  "no improvement": (({"transitions": 1}, last[1]), ({"transitions": 2}, last[1]))}
        for case, trace in broken.items():
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                model.trace_ = trace
                cost(model, run.score)
        model.trace_ = (last,)
        model.counters_ = {"updates": 3}
        with self.assertRaises(RuntimeError):
            cost(model, run.score)  # cost_unit missing from counters_

    def test_one_seed_summary_has_no_spread(self):
        summary = seeds_summary(fit_seeds(tiny_instance(), cheap(), [0]))
        self.assertIsNone(summary["score_std"])
        self.assertIsNone(summary["score_ci95"])

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
