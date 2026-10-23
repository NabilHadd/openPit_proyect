from copy import deepcopy
from statistics import mean
import subprocess
import sys
from types import SimpleNamespace
import unittest

from src.config import PROJECT_ROOT
from src.evaluation.tiny_oracle import TinyOracleResult
from src.evaluation.validation_report import attach_benchmark_metrics, summarize_run, summarize_suite


def sample_run(seed=0, transitions=10):
    def path(value, steps):
        return SimpleNamespace(evaluation=SimpleNamespace(npv=value), steps=steps)
    return SimpleNamespace(seed=seed, best_observed=path(15, 3), best_completed=None,
                           training={"transitions": transitions, "history": ["excluded"],
                                     "unique_q_states": 6, "unique_state_action_pairs": 12},
                           policy=(SimpleNamespace(path=path(15, 4), terminated=True, truncated=False,
                                                   reason="stop", decisions={"all_zero_q_decisions": 1}),
                                   SimpleNamespace(path=path(15, 3), terminated=False, truncated=True,
                                                   reason="external_episode_limit", decisions={}),
                                   SimpleNamespace(path=path(10, 2), terminated=True, truncated=False,
                                                   reason="stop", decisions={})), generation_seconds=2.5)


class ValidationReportTests(unittest.TestCase):
    def test_import_has_no_runtime_training_storage_or_benchmark_dependency(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import src.evaluation.validation_report; "
                                 "assert not any(name.startswith('src.training') for name in sys.modules); "
                                 "assert 'src.evaluation.validation_artifacts' not in sys.modules; "
                                 "assert 'src.evaluation.tiny_oracle' not in sys.modules; "
                                 "assert 'src.evaluation.benchmark' not in sys.modules"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def row(self):
        return summarize_run(sample_run(), directory="seed-0", exports={"best-observed": "verified-plan.json"})

    def test_summary_projects_results_without_training_disk_or_a_model(self):
        run = sample_run()
        row = summarize_run(run, directory="seed-0", exports={})
        self.assertEqual(row, {
            "seed": 0, "directory": "seed-0", "exports": {}, "best_observed_npv": 15,
            "best_completed_npv": None, "policy_values": [15, 15, 10], "policy_mean": 40 / 3,
            "policy_median": 15, "policy_completed_flags": [True, False, True],
            "policy_min": 10, "policy_max": 15, "policy_transitions": 9, "policy_truncations": 1,
            "policy_termination_counts": {"external_episode_limit": 1, "stop": 2},
            "policy_decisions": [{"all_zero_q_decisions": 1}, {}, {}],
            "training": {"transitions": 10, "unique_q_states": 6, "unique_state_action_pairs": 12},
            "generation_seconds": 2.5})
        self.assertEqual(run.training["history"], ["excluded"])
        self.assertIsNot(row["training"], run.training)

    def test_benchmark_distinguishes_value_hits_completion_and_finite_coverage(self):
        row = self.row()
        oracle = TinyOracleResult(15, 15, (), (), 20, 40)
        attach_benchmark_metrics(row, target=15, oracle=oracle, finite_horizon=True)
        self.assertEqual((row["policy_hits"], row["policy_completed_hits"], row["policy_truncated_value_hits"]), (2, 1, 1))
        self.assertTrue(row["best_observed_hit"])
        self.assertEqual(row["best_observed_gap"], 0)
        self.assertAlmostEqual(row["policy_mean_gap"], 5 / 3)
        self.assertEqual(row["training"]["reachable_q_state_coverage"], .3)
        self.assertEqual(row["training"]["reachable_state_action_coverage"], .3)
        for oracle, finite in ((oracle, False), (None, True)):
            other = self.row()
            attach_benchmark_metrics(other, target=15, oracle=oracle, finite_horizon=finite)
            self.assertNotIn("reachable_q_state_coverage", other["training"])

    def test_hit_tolerance_and_strict_completion_pairing_remain_unchanged(self):
        row = self.row()
        row.update(best_observed_npv=15 + .5e-6, policy_values=[15 + .5e-6, 15 + 2e-6],
                   policy_completed_flags=[True, True])
        attach_benchmark_metrics(row, target=15, oracle=None, finite_horizon=True)
        self.assertTrue(row["best_observed_hit"])
        self.assertEqual(row["policy_hits"], 1)
        self.assertEqual(row["policy_completed_hits"], 1)
        row["policy_completed_flags"] = [True]
        with self.assertRaises(ValueError):
            attach_benchmark_metrics(row, target=15, oracle=None, finite_horizon=True)

    def test_suite_aggregates_without_mutating_rows(self):
        left, right = self.row(), self.row()
        right.update(best_observed_npv=12, policy_values=[12, 10, 8], policy_mean=10)
        for row in (left, right):
            attach_benchmark_metrics(row, target=15, oracle=None, finite_horizon=True)
        rows = [left, right]
        before = deepcopy(rows)
        result = summarize_suite(rows, evaluation_rollouts=3, training_transitions=201, tiny=False)
        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["best_observed_max"], 15)
        self.assertEqual(result["best_observed_hits"], 1)
        self.assertEqual(result["policy_mean"], mean([40 / 3, 10]))
        self.assertEqual(result["policy_hits"], 2)
        self.assertEqual(result["policy_completed_hits"], 1)
        self.assertEqual(result["policy_truncated_value_hits"], 1)
        self.assertEqual(result["policy_rollouts"], 6)
        self.assertEqual(result["policy_transitions"], 18)
        self.assertEqual(result["training_transitions_per_seed"], 201)
        self.assertNotIn("tiny_policy_gate", result)
        self.assertEqual(rows, before)

    def test_tiny_gate_requires_ten_seeds_and_every_rollout_completed(self):
        row = self.row()
        row.update(policy_values=[15] * 3, policy_completed_flags=[True] * 3)
        attach_benchmark_metrics(row, target=15, oracle=None, finite_horizon=True)
        for size, expected in ((9, "not_assessed_fewer_than_10_seeds"), (10, "passed")):
            result = summarize_suite([deepcopy(row) for _ in range(size)], evaluation_rollouts=3,
                                     training_transitions=10, tiny=True)
            self.assertEqual(result["tiny_policy_gate"], expected)
        rows = [deepcopy(row) for _ in range(10)]
        rows[-1]["policy_completed_hits"] = 2
        rows[-1]["policy_truncated_value_hits"] = 1
        result = summarize_suite(rows, evaluation_rollouts=3, training_transitions=10, tiny=True)
        self.assertEqual(result["policy_hits"], 30)
        self.assertEqual(result["policy_completed_hits"], 29)
        self.assertEqual(result["tiny_policy_gate"], "failed")


if __name__ == "__main__":
    unittest.main()
