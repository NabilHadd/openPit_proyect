"""Shared-waste potential (src/training/potential.py), its use as the initial Q of the tabular
models, and DQN (src/models/dqn.py)."""

from dataclasses import replace
import unittest

import numpy as np

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.dataset import Block, Mine
from src.data.instance import MineInstance
from src.data.precedences import Precedences
from src.envs.mine_env import ADVANCE, STOP, MineEnv
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine
from src.models.dqn import ConeFeatures, DQNEstimator, DQNParams
from src.models.q_learning import QLearningEstimator
from src.training.potential import SharedWastePotential


def valley_mine() -> tuple[Mine, Precedences]:
    """Two ore blocks under shared waste: each cone alone loses (5 - 6), both win (10 - 9)."""
    mine = Mine((Block(1, 0, 0, 2, -3), Block(2, 1, 0, 2, -3), Block(3, 2, 0, 2, -3),
                 Block(4, 0, 0, 1, 5), Block(5, 1, 0, 1, 5)))
    return mine, Precedences({1: (), 2: (), 3: (), 4: (1, 2), 5: (2, 3)}, "valley", "test")


def environment(mine, rules, problem=ProblemParameters()) -> MineEnv:
    return MineEnv(mine, problem, rules, EnvironmentParameters(20, "external"))


class PotentialTests(unittest.TestCase):
    def test_shared_waste_is_split_among_the_ore_that_needs_it(self):
        env = environment(*valley_mine())
        phi = SharedWastePotential(env.blocks, env.cones, 0.0)
        start = env.initial_state()
        self.assertEqual(phi.share, {1: 1, 2: 2, 3: 1})
        self.assertAlmostEqual(phi(start), 1.0)             # each ore: 5 - 3 - 1.5 = 0.5
        after_shared = env.extract(start, 2).state           # the shared waste is paid
        self.assertAlmostEqual(phi(after_shared), 4.0)      # each ore: 5 - 3 = 2
        after_ore = env.extract(after_shared, 4).state       # ore 4 (and its waste 1) leaves the ground
        self.assertAlmostEqual(phi(after_ore), 2.0)

    def test_discounted_at_the_current_period_and_never_above_the_positive_value_left(self):
        env = environment(*valley_mine(), ProblemParameters(2, 0.1))
        phi = SharedWastePotential(env.blocks, env.cones, 0.1)
        state = env.cone_step(env.initial_state(), ADVANCE).state
        self.assertAlmostEqual(phi(state), 1.0 / 1.1)
        mine, rules = tiny_mine()
        env = environment(mine, rules)
        phi = SharedWastePotential(env.blocks, env.cones, 0.0)
        positive = sum(b.value for b in mine.blocks if b.reachable and b.value > 0)
        self.assertLessEqual(phi(env.initial_state()), positive)


class InitialQTests(unittest.TestCase):
    def test_a_pair_never_updated_starts_at_the_potential_and_stop_at_zero(self):
        mine, rules = valley_mine()
        model = QLearningEstimator(transitions=50, policy_rollouts=1, initial_q="shared_waste")
        model.fit(MineInstance(mine, rules, ProblemParameters(), EnvironmentParameters(20, "external")))
        start = environment(mine, rules).initial_state()
        unseen = replace(start, position=99)  # a state the agent never visited
        self.assertAlmostEqual(model.value(unseen, "north"), model._potential(unseen))
        self.assertEqual(model.value(unseen, STOP), 0.0)
        self.assertEqual(QLearningEstimator().initial_q, "zero")  # the default changes no result

    def test_invalid_initial_q(self):
        with self.assertRaisesRegex(ValueError, "initial_q"):
            QLearningEstimator(initial_q="huge", transitions=10).fit(MineInstance(*tiny_mine()))


class ConeFeaturesTests(unittest.TestCase):
    def test_actions_are_those_of_the_cone_view_and_rows_match_the_simulator(self):
        mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                     Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1),
                     Block(5, 5, 5, 3, -1, tonnes=1)))
        rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: (), 5: ()})
        for problem in (ProblemParameters(), ProblemParameters(2, 0.1, 2), ProblemParameters(3, 0.15, 4, "tonnes")):
            env = environment(mine, rules, problem)
            phi = SharedWastePotential(env.blocks, env.cones, problem.discount_rate)
            features = ConeFeatures(env, phi)
            stack, checked = [env.initial_state()], 0
            while stack:
                state = stack.pop()
                actions, rows = features(state)
                self.assertEqual(actions, tuple(a for a in env.cone_actions(state) if a != STOP))
                for action, row in zip(actions, rows):
                    step = env.cone_step(state, action)
                    with self.subTest(problem=problem, action=action):
                        self.assertAlmostEqual(row[0] * features.scale, step.reward)
                        self.assertAlmostEqual(row[1] * features.scale,
                                               (phi.undiscounted(step.state.extracted) - phi.undiscounted(state.extracted))
                                               / (1 + problem.discount_rate) ** state.period)
                        self.assertEqual(row[9], float(action == ADVANCE))
                    checked += 1
                    if not step.terminal:
                        stack.append(step.state)
            self.assertGreater(checked, 10)


class DQNTests(unittest.TestCase):
    def test_reaches_the_optimum_of_the_tiny_mine_and_of_the_valley(self):
        mine, rules = tiny_mine()
        tiny = MineInstance(mine, rules, ProblemParameters(), EnvironmentParameters(20, "external"))
        target = enumerate_tiny(mine, rules, EnvironmentParameters(12, "finite")).closed_subset_optimum
        valley = MineInstance(*valley_mine(), ProblemParameters(), EnvironmentParameters(20, "external"))
        for instance, optimum in ((tiny, target), (valley, 1.0)):
            for seed in range(3):
                with self.subTest(optimum=optimum, seed=seed):
                    model = DQNEstimator(transitions=600, warmup=100, replay_size=600, target_update=100,
                                         policy_rollouts=2, random_state=seed).fit(instance)
                    self.assertAlmostEqual(model.score(instance), optimum)

    def test_same_seed_same_result_and_counters(self):
        instance = MineInstance(*valley_mine(), ProblemParameters(2, 0.1, 3), EnvironmentParameters(20, "external"))
        runs = [DQNEstimator(**DQNParams.SMOKE, random_state=4).fit(instance) for _ in range(2)]
        self.assertEqual(runs[0].history_, runs[1].history_)
        self.assertEqual(runs[0].trace_, runs[1].trace_)
        self.assertEqual(runs[0].counters_["transitions"], DQNParams.SMOKE["transitions"])
        _, rows = runs[0].features_(MineEnv(instance.mine, instance.problem, instance.precedences,
                                            instance.environment).initial_state())
        self.assertTrue(np.array_equal(runs[0].q_values(rows), runs[1].q_values(rows)))

    def test_invalid_parameters(self):
        for values in ({"batch_size": 64, "replay_size": 32}, {"learning_rate": 0}, {"hidden_units": 0},
                       {"warmup": -1}, {"epsilon_schedule": "fast"}, {"gamma": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DQNParams(**values)


if __name__ == "__main__":
    unittest.main()
