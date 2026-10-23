from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import Mock, patch

from src import validation
from src.config import LearningParameters, SpatialParameters
from src.evaluation.tiny_oracle import tiny_mine
from src.models.validated_q_learning import ValidationQLearning
from src.training.validation_runner import ValidationBudget, train_transition_budget
from src.training.validation_suite import evaluate_validation_benchmark, run_validation_suite


class ValidationSuiteTests(unittest.TestCase):
    def run_suite(self, output, *, transitions=10, evaluation_rollouts=20, seeds=(0, 1), **dependencies):
        mine, rules = tiny_mine()
        return run_validation_suite(mine, rules, SpatialParameters("max-z", 12, True), LearningParameters(alpha=1),
                                    seeds, ValidationBudget(transitions, evaluation_rollouts), Path(output), **dependencies)

    def test_posthoc_oracle_transition_budgets_coverage_and_experimental_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = []
            def tracked(*args, **kwargs):
                run = train_transition_budget(*args, **kwargs)
                completed.append(run.seed)
                return run
            def oracle(*args, **kwargs):
                self.assertEqual(completed, [0, 1])
                return evaluate_validation_benchmark(*args, **kwargs)
            directory = self.run_suite(temporary, transitions=501, evaluation_rollouts=3, trainer=tracked, benchmark=oracle)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["benchmark"]["optimum"], 15)
            self.assertFalse(report["is_final_model"])
            self.assertFalse(report["benchmark_used_by_agent"])
            self.assertEqual(report["summary"]["comparison_status"], "smoke_fewer_than_10_seeds")
            self.assertEqual(report["summary"]["tiny_policy_gate"], "not_assessed_fewer_than_10_seeds")
            for row in report["runs"]:
                self.assertEqual(row["training"]["transitions"], 501)
                self.assertLessEqual(row["training"]["reachable_q_state_coverage"], 1)
                self.assertLessEqual(row["training"]["reachable_state_action_coverage"], 1)
                folder = directory / row["directory"]
                for name in ("training.json", "q-table.json", "policy.json"):
                    self.assertFalse(json.loads((folder / name).read_text())["is_final_model"])
                for role, name in row["exports"].items():
                    exported = json.loads((folder / role / name).read_text())
                    self.assertFalse(exported["is_final_model"])
                    self.assertTrue(exported["evaluation"]["feasible"])

    def test_failed_seed_does_not_compute_oracle_or_partial_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            def failing(*args, **kwargs):
                if args[4] == 1:
                    raise RuntimeError("controlled failure")
                return train_transition_budget(*args, **kwargs)
            oracle = Mock()
            with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                self.run_suite(temporary, trainer=failing, benchmark=oracle)
            oracle.assert_not_called()
            report = json.loads(next(Path(temporary).glob("*/report.json")).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_runs"], 1)
            self.assertNotIn("summary", report)

    def test_previous_model_is_released_before_training_the_next_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = []
            def tracked(*args, **kwargs):
                if models:
                    self.assertIsNone(models[-1]())
                run = train_transition_budget(*args, **kwargs)
                models.append(weakref.ref(run.model))
                return run
            self.run_suite(temporary, trainer=tracked)
            self.assertTrue(all(model() is None for model in models))

    def test_rejected_export_does_not_complete_seed_or_compute_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            oracle = Mock()
            with patch("src.evaluation.validation_artifacts.export_checked_plan", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "El verificador rechazó un plan de validación"):
                    self.run_suite(temporary, seeds=(0,), benchmark=oracle)
            oracle.assert_not_called()
            report = json.loads(next(Path(temporary).glob("*/report.json")).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_runs"], 0)
            self.assertNotIn("summary", report)

    def test_complete_computation_is_not_a_passed_tiny_policy_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = validation.parser().parse_args(["--transitions", "1", "--evaluation-rollouts", "1", "--output", temporary])
            with redirect_stdout(io.StringIO()):
                directory = validation.execute(args)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["completed_runs"], 10)
            self.assertEqual(report["summary"]["tiny_policy_gate"], "failed")

    def test_external_cap_one_cannot_certify_the_start_prefix_as_task_optimal(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = validation.parser().parse_args(["--transitions", "1", "--evaluation-rollouts", "1",
                                                    "--horizon-mode", "external", "--max-steps", "1", "--output", temporary])
            with redirect_stdout(io.StringIO()):
                directory = validation.execute(args)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["benchmark"]["optimum"], 15)
            self.assertEqual(report["summary"]["policy_hits"], 0)
            self.assertEqual(report["summary"]["tiny_policy_gate"], "failed")

    def test_optimal_value_at_external_cutoff_is_not_a_completed_policy_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = validation.parser().parse_args(["--transitions", "3", "--evaluation-rollouts", "1",
                                                    "--horizon-mode", "external", "--max-steps", "3", "--output", temporary])
            def scripted(model, state, actions, *, explore=True):
                return "start" if state.position is None else "down"
            with patch.object(ValidationQLearning, "select_action", scripted), redirect_stdout(io.StringIO()):
                directory = validation.execute(args)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["summary"]["policy_hits"], 10)
            self.assertEqual(report["summary"]["policy_completed_hits"], 0)
            self.assertEqual(report["summary"]["policy_truncated_value_hits"], 10)
            self.assertEqual(report["summary"]["tiny_policy_gate"], "failed")

    def test_invalid_args_do_not_create_successful_reports(self):
        for extra in (["--transitions", "0"], ["--transitions", "1", "--seeds", "0", "0"],
                      ["--transitions", "1", "--precedence", "grid9-z-up"]):
            with tempfile.TemporaryDirectory() as temporary:
                args = validation.parser().parse_args([*extra, "--output", temporary])
                with self.assertRaises(ValueError):
                    validation.execute(args)
                self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
