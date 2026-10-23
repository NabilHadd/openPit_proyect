from dataclasses import asdict
from contextlib import redirect_stdout
import inspect
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from openpyxl import Workbook

import main
from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import grid9_z_up
from src.evaluation.benchmark import maximum_closure
from src.evaluation.control_report import attach_benchmark_outcome, first_benchmark_hit, summarize_controls
from src.training.controlled_experiments import CONTROL_SPECS, iter_control_runs


def toy_mine():
    return Mine((Block(1, 1, 1, 3, 5), Block(2, 1, 1, 2, -4),
                 Block(3, 1, 1, 1, 15), Block(4, 2, 1, 3, -1)))


def toy_workbook(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "105 Bloques"
    sheet.append(["ID", "X", "Y", "Z", "Valor"])
    for block in toy_mine().blocks:
        sheet.append([block.id, block.x, block.y, block.z, block.value])
    workbook.save(path)


class ControlledRunnerTests(unittest.TestCase):
    def test_stream_does_not_retain_previous_models_without_cyclic_gc(self):
        mine = toy_mine()
        runs = iter_control_runs(mine, grid9_z_up(mine), ProblemParameters(), SpatialParameters("max-z", 12, True),
                                LearningParameters(episodes=2), (0, 1))
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            first = next(runs)
            previous_model = weakref.ref(first.model)
            del first
            second = next(runs)
            self.assertIsNone(previous_model())
            self.assertEqual(second.seed, 1)
            runs.close()
        finally:
            if was_enabled:
                gc.enable()

    def test_four_combinations_same_environment_and_streaming_seed_isolation(self):
        mine = toy_mine()
        runs = iter_control_runs(mine, grid9_z_up(mine), ProblemParameters(),
                                SpatialParameters("max-z", 12, True), LearningParameters(episodes=3, alpha=1, epsilon=0.3), (2, 5))
        self.assertTrue(inspect.isgenerator(runs))
        records = list(runs)
        self.assertEqual([(r.spec.id, r.seed) for r in records], [(s.id, seed) for s in CONTROL_SPECS for seed in (2, 5)])
        self.assertEqual(len({id(r.model.q) for r in records}), 8)
        self.assertTrue(all(r.learning.alpha == 1 and r.learning.epsilon == 0.3 and r.learning.episodes == 3 for r in records))
        self.assertTrue(all(r.learning.gamma == r.spec.gamma for r in records))
        self.assertTrue(all(r.result.selected_path.evaluation.feasible for r in records))
        self.assertTrue(all(r.generation_seconds >= r.diagnostics["timings"]["learning_seconds"] + r.diagnostics["timings"]["greedy_seconds"] for r in records))
        repeated = list(iter_control_runs(mine, grid9_z_up(mine), ProblemParameters(), SpatialParameters("max-z", 12, True),
                                         LearningParameters(episodes=3, alpha=1, epsilon=0.3), (2, 5)))
        for a, b in zip(records, repeated, strict=True):
            self.assertEqual(a.result, b.result)
            self.assertEqual(a.model.q, b.model.q)
            self.assertEqual({k: v for k, v in a.diagnostics.items() if k != "timings"},
                             {k: v for k, v in b.diagnostics.items() if k != "timings"})

    def test_invalid_controls_are_rejected(self):
        mine = toy_mine()
        for problem, spatial, seeds in (
            (ProblemParameters(), SpatialParameters("max-z", 10, False), (0,)),
            (ProblemParameters(), SpatialParameters("max-z", 10, True, "external"), (0,)),
            (ProblemParameters(capacity=1), SpatialParameters("max-z", 10, True), (0,)),
            (ProblemParameters(), SpatialParameters("max-z", 10, True), (0, 0)),
        ):
            with self.assertRaises(ValueError):
                list(iter_control_runs(mine, grid9_z_up(mine), problem, spatial, LearningParameters(episodes=1), seeds))

    def test_posthoc_target_comes_from_data_and_first_hit_keeps_actual_step(self):
        mine = toy_mine()
        target = maximum_closure(mine, grid9_z_up(mine)).evaluation.npv
        self.assertEqual(target, 15)
        run = next(iter_control_runs(mine, grid9_z_up(mine), ProblemParameters(), SpatialParameters("max-z", 12, True),
                                     LearningParameters(episodes=30, alpha=1, epsilon=0.5), (8,)))
        row = {"selected_evaluation": asdict(run.result.selected_path.evaluation),
               "best_episode_evaluation": asdict(run.result.best_evaluation),
               "greedy_evaluation": asdict(run.result.greedy_evaluation)}
        payload = asdict(run.result)
        before = run.model.q.copy(), run.model.rng.getstate()
        attach_benchmark_outcome(row, payload, target)
        self.assertEqual(before, (run.model.q, run.model.rng.getstate()))
        self.assertFalse(row["benchmark"]["used_by_agent"])
        self.assertIsNotNone(first_benchmark_hit(payload, target))
        hit = row["benchmark"]["first_observed_hit"]
        if hit["phase"] == "learning":
            episode = payload["history"][hit["episode"]]
            self.assertEqual(hit["step"], episode["best_path"]["steps"])
            self.assertEqual(hit["training_transitions_before_hit"], sum(e["steps"] for e in payload["history"][:hit["episode"]]) + hit["step"])

    def test_milestone_requires_ten_seeds_and_compatible_reference(self):
        mine = Mine((Block(1, 0, 0, 3, 7),))
        seeds = tuple(range(10))
        rows = []
        for run in iter_control_runs(mine, grid9_z_up(mine), ProblemParameters(), SpatialParameters("max-z", 3, True),
                                     LearningParameters(episodes=1), seeds):
            row = {"control": asdict(run.spec), "seed": run.seed,
                   "selected_evaluation": asdict(run.result.selected_path.evaluation),
                   "best_episode_evaluation": asdict(run.result.best_evaluation),
                   "greedy_evaluation": asdict(run.result.greedy_evaluation),
                   "diagnostics": run.diagnostics, "generation_seconds": run.generation_seconds}
            attach_benchmark_outcome(row, asdict(run.result), 7)
            rows.append(row)
        compatible = summarize_controls(rows, seeds, reference_compatible=True)
        self.assertTrue(all(r["milestone_1"] == "achieved" and r["selected_hits"] == 10 for r in compatible["controls"].values()))
        incompatible = summarize_controls(rows, seeds, reference_compatible=False)
        self.assertTrue(all(r["milestone_1"] == "not_assessed_reference_unverified_or_incompatible" for r in incompatible["controls"].values()))


class ControlledCLITests(unittest.TestCase):
    def arguments(self, path, directory, *extra):
        return main.parser().parse_args(["experiments", "--data", str(path), "--precedence", "grid9-z-up",
                                         "--output", str(directory), "--episodes", "3", "--max-steps", "12", *extra])

    def test_complete_cli_posthoc_order_and_experimental_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "toy.xlsx"
            toy_workbook(path)
            emitted = []
            def tracked(*args, **kwargs):
                for run in iter_control_runs(*args, **kwargs):
                    emitted.append((run.spec.id, run.seed))
                    yield run
            def posthoc(*args, **kwargs):
                self.assertEqual(len(emitted), 8)
                return maximum_closure(*args, **kwargs)
            def absent_reference(*args, **kwargs):
                self.assertEqual(len(emitted), 8)
                raise ValueError("Fixture sin hojas visuales privadas")
            with patch.object(main, "iter_control_runs", tracked), patch.object(main, "maximum_closure", posthoc), \
                    patch.object(main, "load_selections", absent_reference), redirect_stdout(io.StringIO()):
                directory = main.execute(self.arguments(path, root, "--seeds", "0", "1"))
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["completed_runs"], 8)
            self.assertFalse(report["is_final_model"])
            self.assertEqual(report["reference_validation"]["status"], "unavailable_or_invalid")
            self.assertEqual(report["exact_benchmark"]["exact_value"], "15")
            self.assertTrue(report["effective_config"]["spatial"]["construction_up"])
            self.assertTrue((directory / "summary.md").exists())
            for row in report["runs"]:
                exported = directory / row["artifact_directory"] / "selected" / row["exports"]["selected"]
                plan = json.loads(exported.read_text())
                self.assertFalse(plan["is_final_model"])
                self.assertEqual(plan["experimental_control"]["control"], row["control"])
                self.assertEqual(plan["evaluation"], row["selected_evaluation"])
            self.assertTrue(all(s["milestone_1"] == "not_assessed_fewer_than_10_seeds" for s in report["summary"]["controls"].values()))
            with self.assertRaises(ValueError):
                summarize_controls(report["runs"][:-1], (0, 1))

    def test_incomplete_stream_is_reported_failed_without_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "toy.xlsx"
            toy_workbook(path)
            def incomplete(*args, **kwargs):
                yield next(iter_control_runs(*args, **kwargs))
            with patch.object(main, "iter_control_runs", incomplete), patch.object(main, "maximum_closure") as benchmark:
                with self.assertRaisesRegex(RuntimeError, "incompleta"):
                    main.execute(self.arguments(path, root, "--seeds", "0"))
                benchmark.assert_not_called()
            reports = list(root.glob("*/report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_runs"], 1)
            self.assertEqual(report["expected_runs"], 4)
            self.assertNotIn("summary", report)

    def test_default_is_ten_distinct_seeds_and_no_legacy_model_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "toy.xlsx"
            toy_workbook(path)
            args = self.arguments(path, temporary)
            args.episodes = 1
            with redirect_stdout(io.StringIO()):
                directory = main.execute(args)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["effective_config"]["seeds"], list(range(10)))
            self.assertEqual(report["completed_runs"], 40)
            self.assertTrue(all(r["milestone_1"] == "not_assessed_reference_unverified_or_incompatible" for r in report["summary"]["controls"].values()))
            for extra in (("--gamma", "1"), ("--model", "spatial-cone"), ("--model", "full-state")):
                with self.assertRaises(ValueError):
                    main.execute(self.arguments(path, temporary, *extra))


if __name__ == "__main__":
    unittest.main()
