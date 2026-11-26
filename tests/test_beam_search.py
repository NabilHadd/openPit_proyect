"""Beam search over the cone view: optimum on the tiny mine, greedy with width 1,
determinism, the best plan of any level, the safety cap and the empty plan."""

from dataclasses import replace
import unittest

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.dataset import Block, Mine
from src.data.instance import MineInstance
from src.data.precedences import Precedences
from src.envs.mine_env import STOP, MineEnv
from src.evaluation.benchmark import maximum_closure
from src.evaluation.metrics import evaluate_plan, same_value
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine
from src.models.beam_search import BeamSearchEstimator, BeamSearchParams


def tiny():
    mine, rules = tiny_mine()
    return MineInstance(mine, rules)


def greedy_value(instance) -> float:
    """Independent greedy search: from the empty plan, always the child of highest VAN
    (first among ties), until no action is left; the best VAN seen (0 = empty plan)."""
    env = MineEnv(instance.mine, instance.problem, instance.precedences, instance.environment)
    state, value, best = env.initial_state(), 0.0, 0.0
    while True:
        children = []
        for action in env.cone_actions(state):
            if action != STOP:
                step = env.cone_step(state, action)
                children.append((value + step.reward, step))
        if not children:
            return best
        top = max(v for v, _ in children)
        value, step = next((v, s) for v, s in children if v == top)
        best = max(best, value)
        if step.terminal:
            return best
        state = step.state


class BeamSearchTests(unittest.TestCase):
    def test_reaches_the_optimum_of_the_tiny_mine(self):
        mine, rules = tiny_mine()
        optimum = enumerate_tiny(mine, rules, EnvironmentParameters()).closed_subset_optimum  # no forced start
        self.assertEqual(optimum, maximum_closure(mine, rules).evaluation.npv)
        for heuristic in ("npv", "optimistic"):
            with self.subTest(heuristic=heuristic):
                model = BeamSearchEstimator(width=10, heuristic=heuristic).fit(tiny())
                self.assertEqual(model.score(model.instance_), optimum)

    def test_width_one_is_the_greedy_search(self):
        mine, rules = tiny_mine()
        for problem in (ProblemParameters(), ProblemParameters(2, 0.1, 3)):
            instance = MineInstance(mine, rules, problem)
            with self.subTest(problem=problem):
                model = BeamSearchEstimator(width=1).fit(instance)
                self.assertTrue(same_value(model.score(instance), greedy_value(instance)))
                self.assertTrue(all(level["kept"] <= 1 for level in model.levels_))

    def test_deterministic_and_random_state_is_irrelevant(self):
        instance = MineInstance(*tiny_mine(), ProblemParameters(2, 0.1, 3))
        fits = [BeamSearchEstimator(width=3, heuristic="optimistic", random_state=seed).fit(instance) for seed in (0, 7)]
        self.assertTrue(BeamSearchEstimator.deterministic)
        self.assertEqual(fits[0].plan_, fits[1].plan_)
        self.assertEqual(fits[0].trace_, fits[1].trace_)
        self.assertEqual(fits[0].counters_, fits[1].counters_)

    def test_delivers_the_best_plan_of_any_level_and_traces_it(self):
        model = BeamSearchEstimator(width=2).fit(tiny())
        score = model.score(model.instance_)
        self.assertTrue(all(level["best_value"] is None or level["best_value"] <= score for level in model.levels_))
        self.assertEqual(model.trace_[0], ({"simulator_calls": 0, "expansions": 0}, 0.0))  # the empty plan
        self.assertEqual(model.trace_[-1][1], score)
        self.assertLessEqual(model.trace_[-1][0]["simulator_calls"], model.counters_["simulator_calls"])
        self.assertGreaterEqual(model.best_level_, 1)
        self.assertFalse(model.capped_)

    def test_safety_cap_stops_the_search_with_a_feasible_plan(self):
        model = BeamSearchEstimator(width=5, max_simulator_calls=4).fit(tiny())
        self.assertTrue(model.capped_)
        self.assertEqual(model.counters_["simulator_calls"], 4)
        instance = model.instance_
        self.assertTrue(evaluate_plan(instance.mine, model.plan_, instance.problem, instance.precedences).feasible)

    def test_a_mine_without_profit_delivers_the_empty_plan(self):
        mine = Mine((Block(1, 0, 0, 2, -3), Block(2, 1, 0, 2, -1), Block(3, 0, 0, 1, -5)))
        model = BeamSearchEstimator().fit(MineInstance(mine, Precedences({1: (), 2: (), 3: (1, 2)})))
        self.assertEqual((model.plan_.extractions, model.score(model.instance_)), ((), 0.0))

    def test_capacity_and_periods_give_feasible_discounted_plans(self):
        mine, rules = tiny_mine()
        instance = MineInstance(mine, rules, ProblemParameters(3, 0.2, 2), EnvironmentParameters(5, "finite"))
        for heuristic in ("npv", "optimistic"):
            with self.subTest(heuristic=heuristic):
                model = BeamSearchEstimator(width=4, heuristic=heuristic).fit(instance)
                evaluation = evaluate_plan(mine, model.plan_, instance.problem, rules)
                self.assertTrue(evaluation.feasible, evaluation.errors)
                self.assertTrue(all(u <= 2 for u in evaluation.period_usage))

    def test_invalid_parameters(self):
        for values in ({"width": 0}, {"width": 2.5}, {"heuristic": "random"}, {"max_simulator_calls": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BeamSearchParams(**values)
        with self.assertRaises(ValueError):
            BeamSearchEstimator(width=0).fit(tiny())
        self.assertEqual(replace(BeamSearchParams(), width=3).width, 3)


if __name__ == "__main__":
    unittest.main()
