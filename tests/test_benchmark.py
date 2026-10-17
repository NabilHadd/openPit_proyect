from itertools import combinations
import random
import unittest

from src.config import ProblemParameters
from src.data.dataset import Block, Mine
from src.data.transforms import Precedences, complete_cone_candidates
from src.evaluation.benchmark import maximum_closure


class ClosureTests(unittest.TestCase):
    def test_complete_support_candidates_are_geometry_not_economics(self):
        mine = Mine(tuple(Block(x + 7 * (y - 1) + 35 * (z - 1), x, y, z, -1)
                          for z in range(1, 4) for y in range(1, 6) for x in range(1, 8)))
        ids = complete_cone_candidates(mine, "max-z")
        self.assertEqual(len(ids), 53)
        self.assertEqual([sum(mine.by_id[i].z == z for i in ids) for z in (1, 2, 3)], [3, 15, 35])

    def test_negative_predecessor_and_decimal_values(self):
        mine = Mine((Block(10, 0, 0, 0, -4.5), Block(20, 0, 0, 1, 15.25), Block(30, 1, 0, 0, -20)))
        rules = Precedences({10: frozenset(), 20: frozenset({10}), 30: frozenset()}, "synthetic", "test")
        result = maximum_closure(mine, rules)
        self.assertEqual(result.exact_value, "43/4")
        self.assertEqual(result.evaluation.npv, 10.75)
        self.assertEqual(tuple(e.block_id for e in result.plan.extractions), (10, 20))

    def test_all_negative_empty_optimum_and_guards(self):
        mine = Mine((Block(1, 0, 0, 0, -3),))
        rules = Precedences({1: frozenset()}, "synthetic", "test")
        self.assertEqual(maximum_closure(mine, rules).plan.extractions, ())
        for parameters in (ProblemParameters(capacity=1), ProblemParameters(discount_rate=0.1), ProblemParameters(periods=2)):
            with self.assertRaises(ValueError):
                maximum_closure(mine, rules, parameters)

    def test_matches_exhaustive_search_on_small_dags(self):
        rng = random.Random(21)
        for _ in range(30):
            mine = Mine(tuple(Block(i, i, 0, 0, rng.randrange(-7, 10)) for i in range(7)))
            predecessors = {i: frozenset(j for j in range(i) if rng.random() < 0.25) for i in range(7)}
            rules = Precedences(predecessors, "synthetic", "exhaustive-test")
            feasible_values = []
            for size in range(8):
                for subset in combinations(range(7), size):
                    chosen = set(subset)
                    if all(predecessors[i] <= chosen for i in chosen):
                        feasible_values.append(sum(mine.by_id[i].value for i in chosen))
            result = maximum_closure(mine, rules)
            self.assertEqual(result.evaluation.npv, max(feasible_values))


if __name__ == "__main__":
    unittest.main()
