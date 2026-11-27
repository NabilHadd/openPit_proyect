from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from configs import PROJECT_ROOT, Distribution
from configs.config_settings import EnvironmentParameters, ProblemParameters, TuningParameters, load_config, parse_config
from configs.model_settings import parse_model_settings
from main import execute, parser
from src.data.dataset import load_mine, load_solution
from src.data.precedences import pit_precedences
from src.models.q_learning import QLearningParams
from src.models.register import load_models, load_settings

SURFACE = range(71, 106)  # every block of Z=3 (the surface): closed and feasible


def block_id(x, y, z):
    return x + (y - 1) * 7 + (z - 1) * 35


def synthetic_inputs(directory):
    """Fabricated 7x5x3 data (value 2 everywhere), NOT the client's data or reference.
    The default fabricated solution (IDs 1..42) is infeasible."""
    data, solution = Path(directory) / "data.csv", Path(directory) / "solution.csv"
    rows = ["id,x,y,z,value"] + [f"{block_id(x, y, z)},{x},{y},{z},2"
                                 for z in range(1, 4) for y in range(1, 6) for x in range(1, 8)]
    data.write_text("\n".join(rows) + "\n")
    write_plan(solution, range(1, 43))
    return data, solution


def write_plan(path, ids, ordered=False):
    rows = "".join(f"{n if ordered else ''},{i},0\n" for n, i in enumerate(ids, start=1))
    Path(path).write_text("order,id,period\n" + rows)


def write_config(directory, filename="config.yml", **values) -> Path:
    """Common YAML plus, next to it (the convention configs/<key>.yml), the Q-learning YAML."""
    data, solution = Path(directory) / "data.csv", Path(directory) / "solution.csv"
    q_learning = Path(directory) / "q_learning.yml"
    q_learning.write_text(yaml.safe_dump({"params": {"transitions": 300, "policy_rollouts": 2}}))
    raw = {"data": str(data), "solution": str(solution), "output": str(directory), "seeds": [9],
           "environment": {"max_steps": 30}, "train": "q_learning"}
    raw.update(values)
    path = Path(directory) / filename
    path.write_text(yaml.safe_dump(raw))
    return path


MINIMAL = {"data": "x", "train": "q_learning"}


def runs(directory, mode) -> list[Path]:
    return sorted((Path(directory) / mode).glob("*")) if (Path(directory) / mode).is_dir() else []


def run(*arguments):
    with redirect_stdout(io.StringIO()):
        return execute(parser().parse_args(list(arguments)))


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_valid(self):
        config = load_config("configs/config.yml")
        self.assertEqual(config.environment.horizon, "external")
        self.assertEqual(config.train, "q_learning")
        self.assertEqual(set(load_models(config, config.experiments)), set(config.experiments))  # all valid
        self.assertEqual(config.model_file("q_learning"), PROJECT_ROOT / "configs" / "q_learning.yml")
        settings = load_settings(config.train, config.model_file(config.train))
        self.assertEqual(settings.params.state, "physical")
        self.assertTrue(settings.distributions)
        self.assertGreaterEqual(config.tuning.n_iter, 1)

    def test_unknown_keys_and_invalid_values_are_errors(self):
        for raw in ({**MINIMAL, "episodes": 5}, {**MINIMAL, "agent": {"state": "physical"}},
                    {**MINIMAL, "training": {"transitions": 10}}, {**MINIMAL, "environment": {"transitions": 10}},
                    {"data": "x"}, {"train": "q_learning"}, {**MINIMAL, "model": "q_learning"},
                    {**MINIMAL, "models": {"q_learning": "q_learning.yml"}}, {**MINIMAL, "train": "q-learning"},
                    {**MINIMAL, "train": 3}, {**MINIMAL, "experiments": "q_learning"},
                    {**MINIMAL, "experiments": ["q_learning", "q_learning"]}, {**MINIMAL, "experiments": [None]},
                    {**MINIMAL, "name": "a/b"}, {**MINIMAL, "name": ""}, {**MINIMAL, "name": 5},
                    {**MINIMAL, "seeds": 3}, {**MINIMAL, "seeds": [1, 1]},
                    {**MINIMAL, "environment": {"horizon": "implicit"}}, {**MINIMAL, "tuning": {"seeds": [0]}},
                    {**MINIMAL, "tuning": {"seeds": [5, 5]}}, {**MINIMAL, "tuning": {"n_iter": 0}},
                    {**MINIMAL, "tuning": {"search_seed": -1}}, {**MINIMAL, "tuning": {"distributions": {}}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_config(raw)
        config = parse_config(MINIMAL)
        self.assertEqual((config.train, config.experiments, config.name, config.output), ("q_learning", (), None, "artifacts"))
        self.assertEqual(parse_config({"data": "x", "experiments": ["q_learning"], "name": "v-2_b"}).experiments,
                         ("q_learning",))
        self.assertIsNone(config.solution)
        self.assertEqual(config.problem, ProblemParameters())
        self.assertEqual(config.environment, EnvironmentParameters())
        self.assertEqual(config.tuning, TuningParameters())

    def test_model_file_is_strict_and_checks_every_distribution(self):
        def tuned(name, spec):
            return {"tuning": {"distributions": {name: spec}}}
        uniform = {"distribution": "uniform", "low": 0.1, "high": 0.5}
        for raw in ({"alpha": 1}, {"params": {"horizon": "finite"}}, {"params": {"alpha": 0}},
                    {"tuning": {"seeds": [1]}}, {"tuning": {"grid": {"alpha": [0.5]}}},
                    tuned("lambda", uniform), tuned("alpha", [0.1, 0.5]), tuned("alpha", {"low": 0.1, "high": 0.5}),
                    tuned("alpha", {**uniform, "distribution": "normal"}), tuned("alpha", {"distribution": "uniform", "low": 0.1}),
                    tuned("alpha", {**uniform, "values": [0.2]}), tuned("alpha", {**uniform, "low": 0.5}),
                    tuned("alpha", {**uniform, "low": "1e-2"}), tuned("alpha", {**uniform, "low": 0.0}),
                    tuned("alpha", {**uniform, "high": 1.5}), tuned("epsilon", {**uniform, "distribution": "loguniform", "low": 0.0}),
                    tuned("epsilon_schedule", {"distribution": "choice", "values": []}),
                    tuned("epsilon_schedule", {"distribution": "choice", "values": "linear"}),
                    tuned("epsilon_schedule", {"distribution": "choice", "values": ["linear", "linar"]}),
                    tuned("alpha", {"distribution": "choice", "values": ["0.5"]}), tuned("state", uniform),
                    tuned("transitions", {"distribution": "choice", "values": [100, 200]}),
                    tuned("policy_rollouts", {"distribution": "uniform", "low": 1, "high": 5})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_model_settings(raw, QLearningParams, "q_learning")
        self.assertEqual(parse_model_settings({}, QLearningParams, "q_learning").params, QLearningParams())
        settings = parse_model_settings({"params": {"alpha": 0.5}, "tuning": {"distributions": {
            "alpha": {"distribution": "loguniform", "low": 1.0e-3, "high": 1},
            "epsilon_schedule": {"distribution": "choice", "values": ["constant", "linear"]}}}}, QLearningParams, "q_learning")
        self.assertEqual(settings.params.alpha, 0.5)
        self.assertEqual(settings.distributions, {"alpha": Distribution("loguniform", low=1.0e-3, high=1),
                                                  "epsilon_schedule": Distribution("choice", values=("constant", "linear"))})

    def test_integer_hyperparameters_take_randint_or_choice_only(self):
        @dataclass(frozen=True)
        class Params:
            steps: int = 5
            rate: float = 0.5

            def __post_init__(self):
                if not 1 <= self.steps <= 50:
                    raise ValueError("steps fuera de rango")

        def tuned(name, spec):
            return {"tuning": {"distributions": {name: spec}}}
        for raw in (tuned("steps", {"distribution": "uniform", "low": 1, "high": 9}),
                    tuned("steps", {"distribution": "loguniform", "low": 1, "high": 9}),
                    tuned("steps", {"distribution": "randint", "low": 1.0, "high": 9}),
                    tuned("steps", {"distribution": "randint", "low": 9, "high": 1}),
                    tuned("steps", {"distribution": "randint", "low": 1, "high": 99}),  # 99 is not valid
                    tuned("steps", {"distribution": "randint", "low": True, "high": 9}),
                    tuned("rate", {"distribution": "randint", "low": 0, "high": 1})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_model_settings(raw, Params, "m")
        settings = parse_model_settings({"tuning": {"distributions": {
            "steps": {"distribution": "randint", "low": 1, "high": 50},
            "rate": {"distribution": "uniform", "low": 0.1, "high": 0.9}}}}, Params, "m")
        self.assertEqual(settings.distributions["steps"], Distribution("randint", low=1, high=50))

    def test_only_mode_config_and_boolean_flags_on_the_terminal(self):
        with redirect_stderr(io.StringIO()):
            for arguments in (["train", "--data", "x"], ["train", "--plan", "x"], ["train", "--plot"],
                              ["evaluate-plan", "--plot"], ["benchmark"], ["experiments", "--plan", "x"]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    parser().parse_args(arguments)


class CLITests(unittest.TestCase):
    def test_reference_cannot_influence_training(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_inputs(directory)
            config = write_config(directory)
            first = run("train", "--config", str(config))
            write_plan(Path(directory) / "solution.csv", SURFACE)
            second = run("train", "--config", str(config))
            third = run("train", "--config", str(write_config(directory, "none.yml", solution=None)))
            names = sorted(p.name for p in (first / "seed-9").iterdir())
            self.assertEqual(names, ["best_observed.csv", "policy_best.csv", "q-table.json", "training.json", "training.svg"])
            for name in names:
                self.assertEqual((first / "seed-9" / name).read_bytes(), (second / "seed-9" / name).read_bytes())
                self.assertEqual((first / "seed-9" / name).read_bytes(), (third / "seed-9" / name).read_bytes())
            report = json.loads((first / "report.json").read_text())
            self.assertEqual(report["data"]["block_count"], 105)
            self.assertEqual(report["precedence_rule"], "pit-9-above")
            self.assertEqual(report["schema_version"], 5)
            self.assertEqual((first / "config.yml").read_bytes(), config.read_bytes())  # exact copy
            self.assertEqual((first / "q_learning.yml").read_bytes(), (Path(directory) / "q_learning.yml").read_bytes())
            self.assertRegex(first.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_train_q_learning(_\d+)?$")
            self.assertEqual(first.parent, Path(directory) / "train")
            self.assertEqual(report["model"]["name"], "q_learning")
            self.assertEqual(report["model"]["params"]["transitions"], 300)
            self.assertNotIn("random_state", report["model"]["params"])
            self.assertEqual(report["config"]["environment"]["max_steps"], 30)
            self.assertFalse(report["reference"]["feasible"])
            self.assertNotIn("reference_npv", report["summary"])
            self.assertNotIn("matches_reference", report["seeds"][0])
            report = json.loads((second / "report.json").read_text())
            summary, row = report["summary"], report["seeds"][0]
            self.assertEqual(summary["reference_npv"], 70)
            self.assertEqual(summary["matches_reference"], int(row["matches_reference"]))
            self.assertEqual(summary["reference"], row["reference"])  # one seed: totals = its counts
            self.assertEqual(row["reference"]["policy_rollouts"], 2)
            report = json.loads((third / "report.json").read_text())
            self.assertIsNone(report["reference"])

    def test_train_outputs_are_verified_plan_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = synthetic_inputs(directory)
            write_plan(Path(directory) / "solution.csv", SURFACE)
            result = run("train", "--config", str(write_config(directory)))
            mine = load_mine(data)
            row = json.loads((result / "report.json").read_text())["seeds"][0]
            self.assertEqual(row["artifacts"], "seed-9")
            for name in ("best_observed.csv", "policy_best.csv"):
                with self.subTest(plan=name):
                    plan = load_solution(result / row["artifacts"] / name, mine)
                    self.assertTrue(plan.ordered)
            delivered = load_solution(result / row["artifacts"] / "best_observed.csv", mine)
            self.assertEqual(row["blocks"], len(delivered.entries))
            self.assertEqual(row["score"], row["diagnostics"]["best_observed"]["npv"])
            self.assertEqual(row["diagnostics"]["training"]["transitions"], 300)

    def test_evaluate_plan_with_and_without_order(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_inputs(directory)
            write_plan(Path(directory) / "solution.csv", SURFACE)
            config = str(write_config(directory))
            ordered = Path(directory) / "ordered.csv"
            write_plan(ordered, (98, 105), ordered=True)
            default = run("evaluate-plan", "--config", config)  # no --plan: evaluates the YAML solution
            explicit = run("evaluate-plan", "--config", config, "--plan", str(ordered))
            report = json.loads((default / "report.json").read_text())["plan"]
            self.assertEqual((report["npv"], report["feasible"], report["file"]["ordered"]), (70, True, False))
            self.assertEqual((default / "plan.csv").read_text().splitlines()[0], "order,id,period")
            self.assertTrue(json.loads((explicit / "report.json").read_text())["plan"]["file"]["ordered"])
            write_plan(ordered, (31, 105), ordered=True)  # 31 is an unreachable edge block of Z=1
            rejected = run("evaluate-plan", "--config", config, "--plan", str(ordered))
            report = json.loads((rejected / "report.json").read_text())["plan"]
            self.assertFalse(report["feasible"])
            self.assertTrue(any("inalcanzable" in e for e in report["errors"]))
            self.assertFalse((rejected / "plan.csv").exists())

    def test_missing_solution_is_an_error_and_null_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_inputs(directory)
            missing = write_config(directory, "missing.yml", solution=str(Path(directory) / "missing.csv"))
            with self.assertRaisesRegex(ValueError, "No existe la solución"):
                run("train", "--config", str(missing))
            with self.assertRaisesRegex(ValueError, "solution es null"):
                run("evaluate-plan", "--config", str(write_config(directory, "none.yml", solution=None)))

    def test_model_files_are_read_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_inputs(directory)
            alone = Path(directory) / "alone"
            alone.mkdir()
            missing = write_config(directory, "missing.yml")
            (alone / "missing.yml").write_text(missing.read_text())  # same config, no q_learning.yml next to it
            other = write_config(directory, "other.yml", train="ppo")
            broken = write_config(directory, "broken.yml")
            Path(directory, "q_learning.yml").write_text(yaml.safe_dump({"params": {"epsilon_schedule": "cubic"}}))
            for config, message in ((alone / "missing.yml", "No existe el YAML del modelo q_learning"),
                                    (other, "Modelo sin implementación: ppo"), (broken, "epsilon_schedule")):
                with self.subTest(config=config.name), self.assertRaisesRegex(ValueError, message):
                    run("train", "--config", str(config))
            self.assertEqual(runs(directory, "train"), [])  # nothing was trained

    def test_cli_outside_repository_and_controlled_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_inputs(directory)
            write_plan(Path(directory) / "solution.csv", SURFACE)
            command = [sys.executable, str(PROJECT_ROOT / "main.py"), "evaluate-plan", "--config",
                       str(write_config(directory))]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("VAN recalculado: 70", result.stdout)
            bad = Path(directory) / "bad.csv"
            for content in ("[]\n", "order,id,period\n1,999,0\n", "id,x\n1,2\n"):
                with self.subTest(content=content):
                    bad.write_text(content)
                    result = subprocess.run([*command, "--plan", str(bad)], cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("Error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
            (Path(directory) / "broken.yml").write_text("data: [unclosed\n")
            result = subprocess.run([*command[:3], "--config", str(Path(directory) / "broken.yml")],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("YAML inválido", result.stderr)


if __name__ == "__main__":
    unittest.main()
