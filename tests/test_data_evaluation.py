import tempfile
import unittest
from pathlib import Path

from src.config import PROJECT_ROOT, LearningParameters, ProblemParameters
from src.data.dataset import Block, Mine, file_hash, load_mine, load_solution
from src.data.precedences import Precedences, order_selection, pit_precedences
from src.evaluation.benchmark import maximum_closure
from src.evaluation.metrics import Extraction, Plan, evaluate_plan


def small_mine():
    return Mine((Block(10, 1, 1, 2, -4, tonnes=2), Block(20, 1, 1, 1, 15, tonnes=3)))


class DataTests(unittest.TestCase):
    def test_parameter_validation(self):
        invalid_problems = ({"periods": 0}, {"discount_rate": float("nan")},
                            {"discount_rate": -0.1}, {"capacity": 0}, {"capacity_unit": "unknown"})
        invalid_learning = ({"episodes": 0}, {"alpha": 0}, {"gamma": 1.1}, {"epsilon": -0.1})
        for parameters in invalid_problems:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ProblemParameters(**parameters)
        for parameters in invalid_learning:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                LearningParameters(**parameters)

    def test_table_and_traceability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("id,x,y,z,value\n8,1,2,3,-4\n")
            mine = load_mine(path)
            self.assertEqual(mine.blocks[0].source_row, 2)
            self.assertEqual(mine.blocks[0].id, 8)
            self.assertEqual(mine.blocks[0].value, -4.0)
            self.assertIsNone(mine.blocks[0].tonnes)
            self.assertEqual(mine.data_sha256, file_hash(path))
            for bad in ("", "abc", "inf", "nan", "True"):
                with self.subTest(bad=bad):
                    path.write_text(f"id,x,y,z,value\n8,1,2,3,{bad}\n")
                    with self.assertRaises(ValueError):
                        load_mine(path)
            path.write_text("id,x,y,z,value\n8,1,2,3.5,1\n")
            with self.assertRaisesRegex(ValueError, "entero"):
                load_mine(path)
            path.write_text("id,x,y,z,value\n8,1,2,3,-4\n8,2,2,3,5\n")
            with self.assertRaisesRegex(ValueError, "duplicados"):
                load_mine(path)
            path.write_text("id,x,y,z,value,tonnes\n8,1,2,3,-4,2.5\n9,2,2,3,1,\n")
            self.assertEqual([b.tonnes for b in load_mine(path).blocks], [2.5, None])

    def test_required_headers_and_coordinates(self):
        with self.assertRaisesRegex(ValueError, "ID interno"):
            Mine((Block(10.0, 1, 1, 1, 3),))
        with self.assertRaisesRegex(ValueError, "duplicados"):
            Mine((Block(1, 1, 1, 1, 3), Block(2, 1, 1, 1, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            for header in ("id,x,y,z", "id,x,y,z,value,grade"):
                with self.subTest(header=header):
                    path.write_text(header + "\n")
                    with self.assertRaisesRegex(ValueError, "columnas"):
                        load_mine(path)

    def test_load_mine_marks_blocks_without_full_cone_unreachable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            rows = ["id,x,y,z,value"] + [f"{x + 3 * y + 1},{x},{y},2,-1" for y in range(3) for x in range(3)]
            rows += ["10,1,1,1,20", "11,0,0,1,5"]  # center (full cone) and corner (incomplete cone)
            path.write_text("\n".join(rows) + "\n")
            mine = load_mine(path)
            self.assertEqual({b.id for b in mine.blocks if not b.reachable}, {11})
            precedences = pit_precedences(mine)
            evaluation = evaluate_plan(mine, Plan((Extraction(1), Extraction(11))), ProblemParameters(), precedences)
            self.assertFalse(evaluation.feasible)
            self.assertTrue(any("inalcanzable" in e for e in evaluation.errors))
            self.assertEqual(maximum_closure(mine, precedences).exact_value, "11")

    def test_solution_csv(self):
        mine = small_mine()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.csv"
            path.write_text("order,id,period\n,20,0\n,10,0\n")
            solution = load_solution(path, mine)
            self.assertFalse(solution.ordered)
            self.assertEqual(solution.ids, (20, 10))
            path.write_text("order,id,period\n2,20,0\n1,10,0\n")
            solution = load_solution(path, mine)
            self.assertTrue(solution.ordered)
            self.assertEqual(solution.ids, (10, 20))
            self.assertEqual(solution.data_sha256, file_hash(path))
            for body, message in ((",99,0", "no existen"), (",10,0\n,10,0", "repetidos"), ("1,10,0\n,20,0", "order"),
                                  ("1,10,0\n3,20,0", "order"), (",10,-1", "period"), ("", "no contiene")):
                with self.subTest(body=body):
                    path.write_text("order,id,period\n" + body + "\n")
                    with self.assertRaisesRegex(ValueError, message):
                        load_solution(path, mine)

    def test_examples_are_valid_and_solution_is_optimal(self):
        examples = PROJECT_ROOT / "data" / "processed"
        mine = load_mine(examples / "data.example.csv")
        solution = load_solution(examples / "solution.example.csv", mine)
        precedences = pit_precedences(mine)
        plan = Plan(tuple(Extraction(e.block_id, e.period) for e in solution.entries))
        evaluation = evaluate_plan(mine, plan, ProblemParameters(), precedences)
        self.assertTrue(evaluation.feasible)
        self.assertEqual(evaluation.npv, maximum_closure(mine, precedences).evaluation.npv)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.mine = small_mine()
        self.precedences = Precedences({10: (), 20: (10,)})

    def test_same_period_order_and_discount(self):
        plan = Plan((Extraction(10), Extraction(20)))
        result = evaluate_plan(self.mine, plan, ProblemParameters(), self.precedences)
        self.assertTrue(result.feasible)
        self.assertEqual(result.npv, 11)
        delayed = Plan((Extraction(10, 0), Extraction(20, 1)))
        result = evaluate_plan(self.mine, delayed, ProblemParameters(periods=2, discount_rate=0.5), self.precedences)
        self.assertEqual(result.npv, 6)

    def test_errors_no_repair(self):
        for plan in (Plan((Extraction(20),)), Plan((Extraction(20), Extraction(10))),
                     Plan((Extraction(10), Extraction(10))), Plan((Extraction(999),)),
                     Plan((Extraction(10, -1),)), Plan((Extraction(True),))):
            self.assertFalse(evaluate_plan(self.mine, plan, ProblemParameters(), self.precedences).feasible)

    def test_period_order_horizon_and_types(self):
        problem = ProblemParameters(periods=2)
        for plan in (Plan((Extraction(10, 1), Extraction(20, 0))),
                     Plan((Extraction(10, 2),)), Plan((Extraction(10, True),)),
                     Plan((Extraction(10, 0.0),))):
            self.assertFalse(evaluate_plan(self.mine, plan, problem, self.precedences).feasible)

    def test_capacity_units_and_unknown_precedence(self):
        plan = Plan((Extraction(10), Extraction(20)))
        for unit, capacity, feasible in (("blocks", 2, True), ("tonnes", 4, False), ("tonnes", 5, True)):
            result = evaluate_plan(self.mine, plan, ProblemParameters(capacity=capacity, capacity_unit=unit), self.precedences)
            self.assertEqual(result.feasible, feasible)
        self.assertIsNone(evaluate_plan(self.mine, plan, ProblemParameters(), None).feasible)
        with self.assertRaisesRegex(ValueError, "tonelaje"):
            evaluate_plan(Mine((Block(1, 1, 1, 1, 1),)), Plan(),
                          ProblemParameters(capacity=5, capacity_unit="tonnes"), None)

    def test_selection_order_is_explicit_and_checks_closure(self):
        self.assertEqual(order_selection((20, 10), self.precedences), (10, 20))
        with self.assertRaises(ValueError):
            order_selection((20,), self.precedences)

    def test_pit_rule_nine_above_and_cycle(self):
        mine = Mine(tuple(Block(i + 1, i % 3, i // 3, 2, 1) for i in range(9)) + (Block(10, 1, 1, 1, 9),))
        self.assertEqual(pit_precedences(mine).predecessors[10], frozenset(range(1, 10)))
        cycle = Precedences({10: frozenset({20}), 20: frozenset({10})}, "test", "test")
        with self.assertRaisesRegex(ValueError, "Ciclo"):
            cycle.validate(self.mine)


if __name__ == "__main__":
    unittest.main()
