"""Mode experiments (main.py): every model of `experiments`, the same seeds, a summary.

Test models are registered here and removed afterwards, so the contract never sees them.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.evaluation.reporter import create_run_directory
from src.models.q_learning import QLearningEstimator, QLearningParams
from src.models.register import REGISTRY, register
from test_cli import SURFACE, run, runs, synthetic_inputs, write_config, write_plan

FAST = {"params": {"transitions": 300, "policy_rollouts": 2}}


class Exploding(QLearningEstimator):
    def fit(self, instance):
        raise RuntimeError("falla a propósito")


class Fixed(QLearningEstimator):
    deterministic = True


def experiments_config(directory, models, **values) -> Path:
    synthetic_inputs(directory)
    write_plan(Path(directory) / "solution.csv", SURFACE)
    for key in models:
        Path(directory, f"{key}.yml").write_text(yaml.safe_dump(FAST))
    return write_config(directory, experiments=list(models), seeds=[9, 10], **values)


class ExperimentsTests(unittest.TestCase):
    def setUp(self):
        for key, cls in (("exploding_test", Exploding), ("fixed_test", Fixed)):
            register(key, params=QLearningParams)(cls)
            self.addCleanup(REGISTRY.pop, key, None)

    def test_every_model_runs_a_failure_is_recorded_and_the_next_one_goes_on(self):
        with tempfile.TemporaryDirectory() as directory:
            config = experiments_config(directory, ["q_learning", "exploding_test", "fixed_test"], name="demo")
            result = run("experiments", "--config", str(config))
            self.assertEqual(result.parent, Path(directory) / "experiments")
            self.assertRegex(result.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_experiments_demo$")
            self.assertEqual((result / "config.yml").read_bytes(), config.read_bytes())
            summary = json.loads((result / "summary.json").read_text())
            models = {entry["model"]: entry for entry in summary["models"]}
            self.assertEqual([e["model"] for e in summary["models"]], ["q_learning", "exploding_test", "fixed_test"])

            ok = models["q_learning"]
            self.assertEqual((ok["status"], ok["seeds"], ok["cost_unit"]), ("ok", [9, 10], "transitions"))
            self.assertEqual(len(ok["summary"]["score_ci95"]), 2)
            self.assertEqual(ok["summary"]["reference_npv"], 70)
            report = json.loads((result / "q_learning" / "report.json").read_text())
            self.assertEqual([row["seed"] for row in report["seeds"]], [9, 10])
            self.assertTrue((result / "q_learning" / "seed-10" / "best_observed.csv").is_file())
            self.assertTrue((result / "q_learning" / "q_learning.yml").is_file())

            failed = models["exploding_test"]
            self.assertEqual((failed["status"], failed["error"]["type"]), ("error", "RuntimeError"))
            self.assertIn("falla a propósito", failed["error"]["message"])
            self.assertIn("Traceback", failed["error"]["traceback"])
            self.assertEqual(json.loads((result / "exploding_test" / "report.json").read_text())["status"], "error")

            fixed = models["fixed_test"]  # deterministic: first seed only, n=1, no interval
            self.assertEqual((fixed["status"], fixed["seeds"], fixed["deterministic"]), ("ok", [9], True))
            self.assertIsNone(fixed["summary"]["score_ci95"])
            self.assertIsNone(fixed["summary"]["score_std"])

            with (result / "summary.csv").open(newline="") as handle:
                rows = {row["model"]: row for row in csv.DictReader(handle)}
            self.assertEqual(set(rows), set(models))
            self.assertEqual((rows["q_learning"]["status"], rows["q_learning"]["seeds"]), ("ok", "2"))
            self.assertEqual(rows["q_learning"]["reached_reference"], f"{ok['summary']['reached_reference']}/2")
            self.assertEqual(rows["fixed_test"]["seeds"], "1")
            self.assertEqual(rows["fixed_test"]["ci95_low"], "")
            self.assertEqual(rows["exploding_test"]["status"], "error")
            self.assertTrue(rows["exploding_test"]["error"].startswith("RuntimeError: falla a propósito"))
            self.assertIn("transitions=300", rows["q_learning"]["params"])

    def test_the_planned_models_run_side_by_side(self):
        """The models of configs/config.yml (experiments) with cheap YAMLs, on synthetic data."""
        cheap = {"dyna_q": {"params": {"transitions": 300, "policy_rollouts": 2, "planning_steps": 3}},
                 "prioritized_sweeping": {"params": {"transitions": 300, "policy_rollouts": 2, "planning_steps": 3}},
                 "dqn": {"params": {"transitions": 300, "policy_rollouts": 2, "warmup": 50, "replay_size": 300,
                                    "target_update": 100}}}
        with tempfile.TemporaryDirectory() as directory:
            experiments_config(directory, [])  # data and reference
            for key, values in cheap.items():
                Path(directory, f"{key}.yml").write_text(yaml.safe_dump(values))
            config = write_config(directory, experiments=list(cheap), seeds=[9, 10])
            result = run("experiments", "--config", str(config))
            models = {e["model"]: e for e in json.loads((result / "summary.json").read_text())["models"]}
            self.assertEqual({k: e["status"] for k, e in models.items()}, dict.fromkeys(cheap, "ok"))
            self.assertEqual((models["dqn"]["seeds"], models["dqn"]["cost_unit"]), ([9, 10], "transitions"))
            self.assertEqual(models["dyna_q"]["seeds"], [9, 10])
            row = json.loads((result / "dyna_q" / "report.json").read_text())["seeds"][0]
            self.assertEqual(row["cost"]["spent"]["planning_updates"], 900)
            self.assertTrue((result / "dqn" / "seed-9" / "best_observed.csv").is_file())
            self.assertTrue((result / "dqn" / "seed-9" / "training.json").is_file())

    def test_every_model_yaml_is_validated_before_anything_is_trained(self):
        with tempfile.TemporaryDirectory() as directory:
            config = experiments_config(directory, ["q_learning", "fixed_test"])
            Path(directory, "fixed_test.yml").write_text(yaml.safe_dump({"params": {"alpha": 0}}))
            with self.assertRaisesRegex(ValueError, "alpha"):
                run("experiments", "--config", str(config))
            Path(directory, "fixed_test.yml").unlink()
            with self.assertRaisesRegex(ValueError, "No existe el YAML del modelo fixed_test"):
                run("experiments", "--config", str(config))
            unknown = write_config(directory, "unknown.yml", experiments=["q_learning", "ppo"])
            with self.assertRaisesRegex(ValueError, "Modelo sin implementación: ppo"):
                run("experiments", "--config", str(unknown))
            empty = write_config(directory, "empty.yml")
            with self.assertRaisesRegex(ValueError, "experiments requiere"):
                run("experiments", "--config", str(empty))
            self.assertEqual(runs(directory, "experiments"), [])  # nothing was trained

    def test_folder_names_are_readable_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root, now = Path(directory) / "train", datetime(2026, 11, 26, 15, 30)
            names = [create_run_directory(root, "train_q_learning", None, now).name for _ in range(3)]
            self.assertEqual(names, ["2026-11-26_15-30_train_q_learning", "2026-11-26_15-30_train_q_learning_2",
                                     "2026-11-26_15-30_train_q_learning_3"])
            self.assertEqual(create_run_directory(root, "experiments", "v1", now).name, "2026-11-26_15-30_experiments_v1")


if __name__ == "__main__":
    unittest.main()
