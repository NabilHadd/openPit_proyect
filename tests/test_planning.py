"""Dyna-Q and prioritized sweeping: what each adds to tabular Q-learning.

With planning_steps = 0 both must be exactly Q-learning for the same seed (same Q table,
episodes, best plan, policy and exploration RNG); with planning they must still reach the
exhaustive optimum of the tiny mine.
"""

import unittest

from configs.config_settings import EnvironmentParameters, ProblemParameters
from golden_support import cases
from src.data.instance import MineInstance
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine
from src.models.dyna_q import DynaQEstimator, DynaQParams
from src.models.prioritized_sweeping import PrioritizedSweepingEstimator, PrioritizedSweepingParams
from src.models.q_learning import QLearningEstimator

PLANNERS = (DynaQEstimator, PrioritizedSweepingEstimator)


def same_fit(left, right):
    return (left.q_, left.history_, left.best_observed_, left.policy_, left.rng_.getstate(),
            [v for _, v in left.trace_]) == (right.q_, right.history_, right.best_observed_, right.policy_,
                                             right.rng_.getstate(), [v for _, v in right.trace_])


class PlanningTests(unittest.TestCase):
    def test_zero_planning_steps_is_exactly_q_learning(self):
        for name in ("tiny-physical-finite-one", "grid-physical-external-multi", "grid-physical-finite-one"):
            instance, params = cases()[name]
            for seed in (0, 1):
                baseline = QLearningEstimator(**params, random_state=seed).fit(instance)
                for planner in PLANNERS:
                    with self.subTest(case=name, seed=seed, model=planner.__name__):
                        model = planner(**params, planning_steps=0, random_state=seed).fit(instance)
                        self.assertTrue(same_fit(model, baseline))
                        self.assertEqual(model.counters_["planning_updates"], 0)

    def test_planning_reaches_the_exhaustive_optimum_of_the_tiny_mine(self):
        mine, rules = tiny_mine()
        for horizon in ("finite", "external"):
            environment = EnvironmentParameters(12 if horizon == "finite" else 50, horizon)
            target = enumerate_tiny(mine, rules, environment).optimum
            for planner in PLANNERS:
                for seed in range(5):
                    with self.subTest(horizon=horizon, model=planner.__name__, seed=seed):
                        model = planner(transitions=3000, planning_steps=5, policy_rollouts=5,
                                        random_state=seed).fit(MineInstance(mine, rules, ProblemParameters(), environment))
                        self.assertEqual(model.score(model.instance_), target)
                        self.assertTrue(all(r.evaluation.npv == target for r in model.policy_))

    def test_only_the_physical_key_and_valid_planning_parameters(self):
        for kind, values in ((DynaQParams, {"state": "position"}), (DynaQParams, {"planning_steps": -1}),
                             (DynaQParams, {"planning_steps": 1.5}), (PrioritizedSweepingParams, {"state": "position"}),
                             (PrioritizedSweepingParams, {"planning_steps": -1}),
                             (PrioritizedSweepingParams, {"theta": -0.1}),
                             (PrioritizedSweepingParams, {"theta": float("nan")}),
                             (PrioritizedSweepingParams, {"theta": True})):
            with self.subTest(kind=kind.__name__, values=values), self.assertRaisesRegex(ValueError, "|".join(values)):
                kind(**values)


class DynaQTests(unittest.TestCase):
    def test_planning_replays_lived_transitions_only(self):
        mine, rules = tiny_mine()
        model = DynaQEstimator(transitions=400, planning_steps=4, policy_rollouts=1).fit(MineInstance(mine, rules))
        counters = model.counters_
        self.assertEqual(counters["planning_updates"], 4 * 400)
        self.assertEqual(counters["updates"], 5 * 400)
        self.assertEqual(counters["simulator_calls"], 400)  # planning never calls the simulator
        self.assertEqual(set(model.model_), {(model.state_key(s), a) for (s, a) in model.q_})  # only pairs lived
        for (key, action), (state, *_rest) in model.model_.items():
            self.assertEqual(model.state_key(state), key)

    def test_the_planning_rng_is_separate_from_exploration(self):
        mine, rules = tiny_mine()
        model = DynaQEstimator(planning_steps=3)
        model._reset_learning()
        exploration = model.rng_.getstate()
        model._planning_rng.random()
        self.assertEqual(model.rng_.getstate(), exploration)


class PrioritizedSweepingTests(unittest.TestCase):
    def test_the_real_pair_is_queued_not_updated_directly(self):
        """With a threshold no priority exceeds, nothing is ever updated: the real
        transition goes through the queue (n > 0)."""
        mine, rules = tiny_mine()
        model = PrioritizedSweepingEstimator(transitions=300, planning_steps=3, theta=1e9,
                                             policy_rollouts=1).fit(MineInstance(mine, rules))
        self.assertEqual(model.q_, {})
        self.assertEqual((model.counters_["updates"], model.counters_["queue_pushes"]), (0, 0))

    def test_queue_keeps_the_highest_priority_and_skips_stale_entries(self):
        model = PrioritizedSweepingEstimator(theta=0.5, planning_steps=2)
        model._reset_learning()
        for priority in (0.4, 2.0, 1.0, 3.0):  # 0.4 <= theta; 1.0 < 2.0 is ignored; 3.0 replaces 2.0
            model._push("s", "a", priority)
        model._push("t", "b", 3.0)  # ties with ("s", "a"): order of arrival decides
        self.assertEqual(model.queue_pushes_, 3)
        self.assertEqual(model._queued, {("s", "a"): 3.0, ("t", "b"): 3.0})
        self.assertEqual([entry[2:] for entry in sorted(model._heap)], [("s", "a"), ("t", "b"), ("s", "a")])

    def test_predecessors_are_ordered_dicts_and_counters_add_up(self):
        mine, rules = tiny_mine()
        model = PrioritizedSweepingEstimator(transitions=500, planning_steps=4, policy_rollouts=1).fit(
            MineInstance(mine, rules))
        self.assertTrue(all(type(p) is dict for p in model.predecessors_.values()))
        counters = model.counters_
        self.assertLessEqual(counters["planning_updates"], 4 * 500)
        self.assertEqual(counters["updates"], counters["planning_updates"])  # every update comes from the queue
        self.assertGreaterEqual(counters["queue_pushes"], counters["planning_updates"])
        for key, pairs in model.predecessors_.items():
            for (before, action) in pairs:
                self.assertEqual(model.state_key(model.model_[before, action][2]), key)


if __name__ == "__main__":
    unittest.main()
