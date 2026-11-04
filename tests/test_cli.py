from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from main import execute, parser
from src.config import PROJECT_ROOT, ProblemParameters
from src.data.dataset import load_mine, load_solution
from src.data.precedences import pit_precedences
from src.evaluation.metrics import Extraction, Plan
from src.evaluation.reporter import export_checked_plan

SURFACE = range(71, 106)  # every block of Z=3 (the surface): closed and feasible


def block_id(x, y, z):
    return x + (y - 1) * 7 + (z - 1) * 35


def synthetic_inputs(directory, surface_value=2):
    """Fabricated 7x5x3 data (value 2 everywhere), NOT the client's data or reference.

    The default fabricated solution (IDs 1..42) contains unreachable edge blocks and
    lower blocks without their upper blocks, so it is infeasible.
    """
    data, solution = Path(directory) / "data.csv", Path(directory) / "solution.csv"
    rows = ["id,x,y,z,value"]
    for z in range(1, 4):
        for y in range(1, 6):
            for x in range(1, 8):
                value = surface_value if block_id(x, y, z) == 71 else 2
                rows.append(f"{block_id(x, y, z)},{x},{y},{z},{value}")
    data.write_text("\n".join(rows) + "\n")
    write_plan(solution, range(1, 43))
    return data, solution


def write_plan(path, ids, ordered=False):
    rows = "".join(f"{n if ordered else ''},{i},0\n" for n, i in enumerate(ids, start=1))
    Path(path).write_text("order,id,period\n" + rows)


class CLITests(unittest.TestCase):
    def test_reference_cannot_influence_training(self):
        with tempfile.TemporaryDirectory() as directory:
            data, solution = synthetic_inputs(directory)
            arguments = ["train", "--data", str(data), "--solution", str(solution), "--output", directory,
                         "--episodes", "6", "--seeds", "9"]
            with redirect_stdout(io.StringIO()):
                first = execute(parser().parse_args(arguments))
                write_plan(solution, SURFACE)
                second = execute(parser().parse_args(arguments))
                third = execute(parser().parse_args([*arguments[:3], "--no-solution", *arguments[5:]]))
            for name in ("training.json", "q-table.json"):
                self.assertEqual((first / "seed-9" / name).read_bytes(), (second / "seed-9" / name).read_bytes())
                self.assertEqual((first / "seed-9" / name).read_bytes(), (third / "seed-9" / name).read_bytes())
            report = json.loads((first / "report.json").read_text())
            self.assertEqual(report["data"]["block_count"], 105)
            self.assertEqual(report["precedences"]["rule"], "pit-9-above")
            self.assertFalse(report["validation_metadata"]["used_by_agent"])
            self.assertIn("source_sha256", report["code"])
            self.assertEqual(report["effective_config"]["learning"]["episodes"], 6)
            self.assertFalse(report["comparison"]["reference_feasible"])
            self.assertFalse(report["comparison"]["comparable"])
            self.assertIsNone(report["comparison"]["best_episode_target_hits"])
            report = json.loads((second / "report.json").read_text())
            self.assertTrue(report["comparison"]["reference_feasible"])
            self.assertEqual(report["comparison"]["reference_value"], 70)
            report = json.loads((third / "report.json").read_text())
            self.assertEqual(report["comparison"]["reference"], "not_provided")
            self.assertNotIn("reference", report)

    def test_exported_plan_is_csv_and_only_if_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, _ = synthetic_inputs(directory)
            mine = load_mine(data)
            precedences = pit_precedences(mine)
            plan = Plan((Extraction(105), Extraction(98)))
            self.assertIsNone(export_checked_plan(root, mine, plan, ProblemParameters(), None))
            self.assertEqual(export_checked_plan(root, mine, plan, ProblemParameters(), precedences), "plan.csv")
            self.assertEqual((root / "plan.csv").read_text(), "order,id,period\n1,105,0\n2,98,0\n")
            exported = load_solution(root / "plan.csv", mine)
            self.assertTrue(exported.ordered)
            self.assertEqual(exported.ids, (105, 98))
            self.assertIsNone(export_checked_plan(root, mine, Plan((Extraction(1),)), ProblemParameters(), precedences))

    def test_evaluate_plan_with_and_without_order(self):
        with tempfile.TemporaryDirectory() as directory:
            data, solution = synthetic_inputs(directory)
            write_plan(solution, SURFACE)
            ordered = Path(directory) / "ordered.csv"
            write_plan(ordered, (98, 105), ordered=True)
            base = ["evaluate-plan", "--data", str(data), "--solution", str(solution), "--output", directory]
            with redirect_stdout(io.StringIO()):
                default = execute(parser().parse_args(base))  # no --plan: evaluates --solution
                explicit = execute(parser().parse_args([*base, "--plan", str(ordered)]))
            report = json.loads((default / "report.json").read_text())
            self.assertEqual(report["evaluation"]["npv"], 70)
            self.assertTrue(report["evaluation"]["feasible"])
            self.assertFalse(report["evaluated_plan"]["plan_file"]["ordered"])
            self.assertEqual((default / "plan.csv").read_text().splitlines()[0], "order,id,period")
            report = json.loads((explicit / "report.json").read_text())
            self.assertTrue(report["evaluated_plan"]["plan_file"]["ordered"])
            self.assertEqual([e["block_id"] for e in report["plan"]["extractions"]], [98, 105])
            write_plan(ordered, (31, 105), ordered=True)  # 31 is an unreachable edge block of Z=1
            with redirect_stdout(io.StringIO()):
                rejected = execute(parser().parse_args([*base, "--plan", str(ordered)]))
            report = json.loads((rejected / "report.json").read_text())
            self.assertFalse(report["evaluation"]["feasible"])
            self.assertTrue(any("inalcanzable" in e for e in report["evaluation"]["errors"]))
            self.assertIsNone(report["plan_export"])

    def test_spatial_reference_and_benchmark_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            data, solution = synthetic_inputs(directory)
            args = parser().parse_args(["train", "--data", str(data), "--solution", str(solution), "--output", directory,
                                        "--model", "spatial-cone", "--episodes", "6", "--max-steps", "30", "--seeds", "9"])
            with redirect_stdout(io.StringIO()):
                first = execute(args)
                write_plan(solution, SURFACE)
                args.exact_benchmark = True
                second = execute(args)
            for name in ("training.json", "q-table.json"):
                self.assertEqual((first / "seed-9" / name).read_bytes(), (second / "seed-9" / name).read_bytes())
            report = json.loads((second / "report.json").read_text())
            self.assertEqual(report["effective_config"]["learning"], {"episodes": 6, "alpha": 1.0, "gamma": 0.7, "epsilon": 0.3})
            self.assertFalse(report["benchmark_comparison"]["used_by_agent"])
            self.assertTrue(report["comparison"]["reference_feasible"])
            entry = report["experiments"][0]
            exported = load_solution(second / "seed-9" / "selected" / entry["selected_export"], load_mine(data))
            self.assertEqual(list(exported.ids), [e["block_id"] for e in entry["selected_path"]["plan"]["extractions"]])
            self.assertTrue(entry["selected_path"]["evaluation"]["feasible"])
            self.assertEqual(len(entry["selected_path"]["actions"]), entry["selected_path"]["steps"])
            self.assertIn("B:", report["domain_confirmations"]["reward"])

    def test_benchmark_cli_without_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = synthetic_inputs(directory, surface_value=1.25)
            args = parser().parse_args(["benchmark", "--data", str(data), "--no-solution", "--output", directory])
            with redirect_stdout(io.StringIO()):
                directory = execute(args)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["evaluation"]["npv"], 105.25)  # only the 53 reachable blocks
            self.assertNotIn("reference", report)
            self.assertEqual(report["exact_benchmark"]["exact_value"], "421/4")

    def test_solution_is_required_unless_explicitly_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = synthetic_inputs(directory)
            base = ["--data", str(data), "--output", directory, "--episodes", "1"]
            with self.assertRaisesRegex(ValueError, "No existe la solución"):
                execute(parser().parse_args(["train", *base, "--solution", str(Path(directory) / "missing.csv")]))
            with self.assertRaisesRegex(ValueError, "no admite --no-solution"):
                execute(parser().parse_args(["evaluate-plan", *base, "--no-solution"]))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(["train", *base, "--no-solution", "--solution", "x.csv"])

    def test_cli_outside_repository_and_controlled_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            data, solution = synthetic_inputs(directory)
            write_plan(solution, SURFACE)
            command = [sys.executable, str(PROJECT_ROOT / "main.py"), "evaluate-plan", "--data", str(data),
                       "--solution", str(solution), "--output", directory]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("VAN recalculado: 70", result.stdout)
            bad = Path(directory) / "bad.csv"
            for content in ("[]\n", "order,id,period\n1,999,0\n", "id,x\n1,2\n"):
                with self.subTest(content=content):
                    bad.write_text(content)
                    result = subprocess.run([*command[:5], "--plan", str(bad), "--output", directory],
                                            cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("Error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
