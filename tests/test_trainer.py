from dataclasses import replace
import unittest
from unittest.mock import patch

from configs.config_settings import EnvironmentParameters, ProblemParameters
from configs.q_learning_settings import QLearningParams
from src.data.dataset import Mine
from src.data.precedences import Precedences
from src.envs.mine_env import MineEnv
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine
from src.models.q_learning import PositionKey, QLearning
from src.training.trainer import train


def run(mine, rules, *, params=QLearningParams(transitions=10_000), environment=EnvironmentParameters(12, "finite"),
        seed=0, problem=ProblemParameters()):
    return train(mine, rules, problem, environment, params, seed)


class TrainerTests(unittest.TestCase):
    def test_tiny_mine_gate_every_combination_learns_the_exhaustive_optimum(self):
        """Before scaling: on six blocks the best observed plan and every policy rollout must
        reach the optimum found by exhaustive enumeration, for ten seeds."""
        mine, rules = tiny_mine()
        for state in ("physical", "position"):
            for horizon in ("finite", "external"):
                # external + gamma=1: moving without extracting is free, so a tight cap
                # can interrupt a wandering rollout; the cap must leave room.
                environment = EnvironmentParameters(12 if horizon == "finite" else 50, horizon)
                target = enumerate_tiny(mine, rules, environment).optimum
                for seed in range(10):
                    with self.subTest(state=state, horizon=horizon, seed=seed):
                        result, _ = run(mine, rules, params=QLearningParams(state=state, transitions=10_000),
                                        environment=environment, seed=seed)
                        self.assertEqual(result.transitions, 10_000)
                        self.assertEqual(sum(e.steps for e in result.history), 10_000)
                        self.assertEqual(result.best_observed.evaluation.npv, target)
                        self.assertTrue(all(r.evaluation.feasible for r in result.policy))
                        if state == "physical":
                            self.assertTrue(all(r.evaluation.npv == target for r in result.policy))

    def test_position_key_with_gamma_one_overestimates_and_is_not_bound_checked(self):
        mine, rules = tiny_mine()
        _, model = run(mine, rules, params=QLearningParams(state="position", transitions=10_000, policy_rollouts=1),
                       environment=EnvironmentParameters(50, "external"))
        self.assertGreater(max(model.q.values()), sum(b.value for b in mine.blocks if b.value > 0))

    def test_position_key_keeps_the_clock_only_in_finite_mode(self):
        mine, rules = tiny_mine()
        model = QLearning(QLearningParams(state="position"), 0)
        for horizon, clock in (("finite", 11), ("external", 0)):
            env = MineEnv(mine, ProblemParameters(), rules, EnvironmentParameters(12, horizon))
            state, _, _ = env.step("start")
            self.assertEqual(model.state_key(state), PositionKey(state.position, 0, clock))
        physical = QLearning(QLearningParams(), 0)
        other = replace(state, extracted=frozenset())
        self.assertNotEqual(physical.state_key(state), physical.state_key(other))
        self.assertEqual(model.state_key(state), model.state_key(other))

    def test_real_end_does_not_bootstrap_but_interruptions_do(self):
        mine, rules = tiny_mine()

        class Primed(QLearning):
            def update(self, state, action, reward, following, actions, terminal):
                self.q[self.state_key(following), "stop"] = 7
                return super().update(state, action, reward, following, actions, terminal)

        start_value = 5  # value of the start block's cone in the tiny mine
        for horizon, cap, expected, end in (("finite", 12, start_value + 7, "budget"),
                                            ("finite", 1, start_value, "step_limit"),
                                            ("external", 1, start_value + 7, "budget")):
            with self.subTest(horizon=horizon, cap=cap), patch("src.training.trainer.QLearning", Primed):
                result, model = run(mine, rules, params=QLearningParams(transitions=1, policy_rollouts=1),
                                    environment=EnvironmentParameters(cap, horizon))
                self.assertEqual(next(v for (_, a), v in model.q.items() if a == "start"), expected)
                self.assertEqual(result.history[0].end, end)

    def test_external_cap_interrupts_episodes(self):
        mine, rules = tiny_mine()
        result, _ = run(mine, rules, params=QLearningParams(transitions=3, policy_rollouts=1),
                        environment=EnvironmentParameters(1, "external"))
        self.assertEqual([e.end for e in result.history], ["episode_cap", "episode_cap", "budget"])
        self.assertEqual(result.policy[0].end, "episode_cap")

    def test_seed_reproducibility_and_rollouts_do_not_touch_training(self):
        mine, rules = tiny_mine()
        left, left_model = run(mine, rules, params=QLearningParams(transitions=201, policy_rollouts=3), seed=3)
        right, right_model = run(mine, rules, params=QLearningParams(transitions=201, policy_rollouts=7), seed=3)
        self.assertEqual(left_model.q, right_model.q)
        self.assertEqual(left_model.rng.getstate(), right_model.rng.getstate())
        self.assertEqual(left.history, right.history)
        self.assertEqual(left.best_observed, right.best_observed)
        self.assertEqual(left.policy, right.policy[:3])

    def test_multi_period_capacity_and_discount_are_supported(self):
        mine, rules = tiny_mine()
        problem = ProblemParameters(periods=2, discount_rate=0.1, capacity=3)
        result, _ = run(mine, rules, problem=problem, params=QLearningParams(transitions=2_000, policy_rollouts=3),
                        environment=EnvironmentParameters(12, "external"))
        self.assertTrue(result.best_observed.evaluation.feasible)
        self.assertTrue(all(len([e for e in r.plan.extractions if e.period == p]) <= 3
                            for r in result.policy for p in (0, 1)))

    def test_invalid_parameters(self):
        for kind, values in ((QLearningParams, {"state": "full"}), (QLearningParams, {"alpha": 0}),
                             (QLearningParams, {"gamma": 1.1}), (QLearningParams, {"transitions": 0}),
                             (QLearningParams, {"policy_rollouts": 0}), (EnvironmentParameters, {"max_steps": 0}),
                             (EnvironmentParameters, {"horizon": "implicit"})):
            with self.subTest(kind=kind.__name__, values=values), self.assertRaises(ValueError):
                kind(**values)


if __name__ == "__main__":
    unittest.main()
