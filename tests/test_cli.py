from contextlib import redirect_stderr, redirect_stdout
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
from configs.q_learning_settings import QLearningParams, load_q_learning_settings, parse_q_learning_settings
from main import execute, parser
from src.data.dataset import load_mine, load_solution
from src.data.precedences import pit_precedences

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


def write_config(directory, name="config.yml", **values) -> Path:
    """Common YAML plus the Q-learning YAML it points to, both inside `directory`."""
    data, solution = Path(directory) / "data.csv", Path(directory) / "solution.csv"
    q_learning = Path(directory) / "q_learning.yml"
    q_learning.write_text(yaml.safe_dump({"params": {"transitions": 300, "policy_rollouts": 2}}))
    raw = {"data": str(data), "solution": str(solution), "output": str(directory), "seeds": [9],
           "environment": {"max_steps": 30}, "model": "q_learning", "models": {"q_learning": str(q_learning)}}
    raw.update(values)
    path = Path(directory) / name
    path.write_text(yaml.safe_dump(raw))
    return path


MINIMAL = {"data": "x", "model": "q_learning", "models": {"q_learning": "q_learning.yml"}}


def run(*arguments):
    with redirect_stdout(io.StringIO()):
        return execute(parser().parse_args(list(arguments)))


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_valid(self):
        config = load_config("configs/config.yml")
        self.assertEqual(config.environment.horizon, "external")
        self.assertEqual(config.model, "q_learning")
        settings = load_q_learning_settings(config.models[config.model])
        self.assertEqual(settings.params.state, "physical")
        self.assertTrue(settings.distributions)
        self.assertGreaterEqual(config.tuning.n_iter, 1)

    def test_unknown_keys_and_invalid_values_are_errors(self):
        for raw in ({**MINIMAL, "episodes": 5}, {**MINIMAL, "agent": {"state": "physical"}},
                    {**MINIMAL, "training": {"transitions": 10}}, {**MINIMAL, "environment": {"transitions": 10}},
                    {key: MINIMAL[key] for key in ("model", "models")}, {key: MINIMAL[key] for key in ("data", "models")},
                    {key: MINIMAL[key] for key in ("data", "model")}, {**MINIMAL, "model": "dqn"},
                    {**MINIMAL, "models": {"q_learning": None}}, {**MINIMAL, "seeds": 3}, {**MINIMAL, "seeds": [1, 1]},
                    {**MINIMAL, "environment": {"horizon": "implicit"}}, {**MINIMAL, "tuning": {"seeds": [0]}},
                    {**MINIMAL, "tuning": {"seeds": [5, 5]}}, {**MINIMAL, "tuning": {"n_iter": 0}},
                    {**MINIMAL, "tuning": {"search_seed": -1}}, {**MINIMAL, "tuning": {"distributions": {}}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_config(raw)
        config = parse_config(MINIMAL)
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
                parse_q_learning_settings(raw)
        self.assertEqual(parse_q_learning_settings({}).params, QLearningParams())
        settings = parse_q_learning_settings({"params": {"alpha": 0.5}, "tuning": {"distributions": {
            "alpha": {"distribution": "loguniform", "low": 1.0e-3, "high": 1},
            "epsilon_schedule": {"distribution": "choice", "values": ["constant", "linear"]}}}})
        self.assertEqual(settings.params.alpha, 0.5)
        self.assertEqual(settings.distributions, {"alpha": Distribution("loguniform", low=1.0e-3, high=1),
                                                  "epsilon_schedule": Distribution("choice", values=("constant", "linear"))})

    def test_only_mode_config_and_boolean_flags_on_the_terminal(self):
        with redirect_stderr(io.StringIO()):
            for arguments in (["train", "--data", "x"], ["train", "--plan", "x"], ["evaluate-plan", "--plot"],
                              ["benchmark"], ["experiments"]):
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
            for name in ("training.json", "q-table.json", "best_observed.csv", "policy_best.csv"):
                self.assertEqual((first / "seed-9" / name).read_bytes(), (second / "seed-9" / name).read_bytes())
                self.assertEqual((first / "seed-9" / name).read_bytes(), (third / "seed-9" / name).read_bytes())
            report = json.loads((first / "report.json").read_text())
            self.assertEqual(report["data"]["block_count"], 105)
            self.assertEqual(report["precedence_rule"], "pit-9-above")
            self.assertEqual(report["schema_version"], 3)
            self.assertEqual(report["model"]["name"], "q_learning")
            self.assertEqual(report["model"]["params"]["transitions"], 300)
            self.assertEqual(report["config"]["environment"]["max_steps"], 30)
            self.assertFalse(report["reference"]["feasible"])
            self.assertNotIn("reference_npv", report["summary"])
            report = json.loads((second / "report.json").read_text())
            self.assertEqual(report["summary"]["reference_npv"], 70)
            self.assertEqual(report["summary"]["policy_rollouts"], 2)
            self.assertIn("matches_reference", report["seeds"][0]["best_observed"])
            report = json.loads((third / "report.json").read_text())
            self.assertIsNone(report["reference"])

    def test_train_outputs_are_verified_plan_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = synthetic_inputs(directory)
            write_plan(Path(directory) / "solution.csv", SURFACE)
            result = run("train", "--config", str(write_config(directory)), "--plot")
            mine = load_mine(data)
            row = json.loads((result / "report.json").read_text())["seeds"][0]
            for key, file in (("best_observed", row["best_observed"]["file"]), ("policy", row["policy"]["best_file"])):
                with self.subTest(plan=key):
                    plan = load_solution(result / file, mine)
                    self.assertTrue(plan.ordered)
            self.assertTrue((result / "seed-9" / "training.svg").is_file())
            self.assertEqual(row["training"]["transitions"], 300)

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
            q_learning = str(Path(directory) / "q_learning.yml")
            missing = write_config(directory, "missing.yml", models={"q_learning": str(Path(directory) / "none.yml")})
            other = write_config(directory, "other.yml", models={"q_learning": q_learning, "dqn": q_learning})
            broken = write_config(directory, "broken.yml")
            Path(q_learning).write_text(yaml.safe_dump({"params": {"epsilon_schedule": "cubic"}}))
            for config, message in ((missing, "No existe la configuración"), (other, "Modelos sin implementación"),
                                    (broken, "epsilon_schedule")):
                with self.subTest(config=config.name), self.assertRaisesRegex(ValueError, message):
                    run("train", "--config", str(config))
            self.assertEqual(list(Path(directory).glob("*-train-*")), [])  # nothing was trained

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
