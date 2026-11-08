from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from main import execute, parser
from src.config import PROJECT_ROOT, ProblemParameters, load_config, parse_config
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
    data, solution = Path(directory) / "data.csv", Path(directory) / "solution.csv"
    raw = {"data": str(data), "solution": str(solution), "output": str(directory), "seeds": [9],
           "training": {"transitions": 300, "max_steps": 30}, "evaluation": {"policy_rollouts": 2}}
    raw.update(values)
    path = Path(directory) / name
    path.write_text(yaml.safe_dump(raw))
    return path


def run(*arguments):
    with redirect_stdout(io.StringIO()):
        return execute(parser().parse_args(list(arguments)))


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_valid(self):
        config = load_config("configs/config.yml")
        self.assertEqual(config.agent.state, "physical")
        self.assertEqual(config.training.horizon, "external")

    def test_unknown_keys_and_invalid_values_are_errors(self):
        for raw in ({"data": "x", "episodes": 5}, {"data": "x", "agent": {"construction_up": True}},
                    {"seeds": [0]}, {"data": "x", "seeds": 3}, {"data": "x", "seeds": [1, 1]},
                    {"data": "x", "training": {"horizon": "implicit"}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_config(raw)
        config = parse_config({"data": "x"})
        self.assertIsNone(config.solution)
        self.assertEqual(config.problem, ProblemParameters())

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
            self.assertEqual(report["config"]["training"]["transitions"], 300)
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
