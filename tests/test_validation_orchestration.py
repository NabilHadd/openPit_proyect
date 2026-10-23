from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock

from src.config import LearningParameters, SpatialParameters
from src.evaluation.tiny_oracle import tiny_mine
from src.training.validation_runner import ValidationBudget
from src.training.validation_suite import run_validation_suite
from test_validation_report import sample_run


class MemoryArtifacts:
    def __init__(self, events, *, fail_start=False, fail_write=None):
        self.events = events
        self.fail_start = fail_start
        self.fail_write = fail_write
        self.snapshots = []

    def start_suite(self, mine, rules, spatial, learning, seeds, budget, output):
        self.events.append("start")
        if self.fail_start:
            raise OSError("initial write")
        report = {"status": "running", "expected_runs": len(seeds), "completed_runs": 0, "runs": []}
        self.write_report(output, report)
        return output, report

    def save_run(self, directory, mine, rules, run):
        self.events.append(("export", run.seed))
        return f"seed-{run.seed}", {}

    def write_report(self, directory, report):
        self.events.append(("write", report["status"], report["completed_runs"]))
        if self.fail_write and self.fail_write(report):
            self.fail_write = None
            raise OSError("report write")
        self.snapshots.append(deepcopy(report))


class ValidationOrchestrationTests(unittest.TestCase):
    def execute(self, *, trainer=None, benchmark=None, store=None):
        mine, rules = tiny_mine()
        return run_validation_suite(mine, rules, SpatialParameters("max-z", 12, True), LearningParameters(),
                                    (0, 1), ValidationBudget(10, 3), Path("unused-validation-output"),
                                    trainer=trainer, benchmark=benchmark, store=store)

    def test_injected_collaborators_preserve_order_without_training_or_file_io(self):
        events = []
        store = MemoryArtifacts(events)
        def trainer(mine, rules, spatial, learning, seed, budget):
            events.append(("train", seed))
            return sample_run(seed, budget.transitions)
        def benchmark(mine, rules, spatial):
            events.append("benchmark")
            return 15, {"kind": "test", "used_by_agent": False}, None
        result = self.execute(trainer=trainer, benchmark=benchmark, store=store)
        self.assertEqual(result, Path("unused-validation-output"))
        self.assertEqual(events, ["start", ("write", "running", 0),
                                  ("train", 0), ("export", 0), ("write", "running", 1),
                                  ("train", 1), ("export", 1), ("write", "running", 2),
                                  "benchmark", ("write", "completed", 2)])
        self.assertNotIn("benchmark", store.snapshots[-2])
        self.assertEqual(store.snapshots[-1]["summary"]["policy_completed_hits"], 2)

    def test_wrong_seed_or_budget_fails_before_export_and_benchmark(self):
        for field in ("seed", "transitions"):
            with self.subTest(field=field):
                events = []
                store = MemoryArtifacts(events)
                benchmark = Mock()
                def trainer(*args):
                    return sample_run(9 if field == "seed" else args[4], 11 if field == "transitions" else 10)
                with self.assertRaisesRegex(RuntimeError, "El ejecutor cambió la semilla o el presupuesto"):
                    self.execute(trainer=trainer, benchmark=benchmark, store=store)
                benchmark.assert_not_called()
                self.assertFalse(any(isinstance(e, tuple) and e[0] == "export" for e in events))
                self.assertEqual(store.snapshots[-1]["status"], "failed")
                self.assertEqual(store.snapshots[-1]["completed_runs"], 0)

    def test_benchmark_failure_preserves_completed_runs_without_success_summary(self):
        store = MemoryArtifacts([])
        with self.assertRaisesRegex(ValueError, "benchmark failure"):
            self.execute(trainer=lambda *args: sample_run(args[4]),
                         benchmark=Mock(side_effect=ValueError("benchmark failure")), store=store)
        last = store.snapshots[-1]
        self.assertEqual((last["status"], last["completed_runs"]), ("failed", 2))
        self.assertEqual(last["error"], "ValueError: benchmark failure")
        self.assertNotIn("summary", last)

    def test_checkpoint_and_final_write_failures_keep_existing_failure_snapshots(self):
        for stage in ("checkpoint", "final"):
            with self.subTest(stage=stage):
                store = MemoryArtifacts([], fail_write=lambda r: (r["status"] == "completed" if stage == "final" else
                                                                  r["status"] == "running" and r["completed_runs"] == 1))
                benchmark = Mock(return_value=(15, {}, None))
                with self.assertRaisesRegex(OSError, "report write"):
                    self.execute(trainer=lambda *args: sample_run(args[4]), benchmark=benchmark, store=store)
                last = store.snapshots[-1]
                self.assertEqual(last["status"], "failed")
                self.assertEqual(last["error"], "OSError: report write")
                self.assertEqual(last["completed_runs"], 2 if stage == "final" else 1)
                self.assertEqual("summary" in last, stage == "final")
                self.assertEqual(benchmark.call_count, 1 if stage == "final" else 0)

    def test_initial_write_failure_stays_outside_failure_checkpoint_handler(self):
        store = MemoryArtifacts([], fail_start=True)
        trainer, benchmark = Mock(), Mock()
        with self.assertRaisesRegex(OSError, "initial write"):
            self.execute(trainer=trainer, benchmark=benchmark, store=store)
        trainer.assert_not_called()
        benchmark.assert_not_called()
        self.assertEqual(store.events, ["start"])


if __name__ == "__main__":
    unittest.main()
