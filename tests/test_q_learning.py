"""Behavior of Q-learning; the common contract of every model is in test_estimator_contract.py."""

from dataclasses import replace
import unittest

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.instance import MineInstance
from src.envs.mine_env import MineEnv
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine
from src.models.q_learning import PositionKey, QLearningEstimator, QLearningParams


def fit(mine, rules, *, estimator=QLearningEstimator, environment=EnvironmentParameters(12, "finite"),
        problem=ProblemParameters(), **params):
    params.setdefault("transitions", 10_000)
    return estimator(**params).fit(MineInstance(mine, rules, problem, environment))


class QLearningTests(unittest.TestCase):
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
                        model = fit(mine, rules, state=state, environment=environment, random_state=seed)
                        self.assertEqual(model.n_transitions_, 10_000)
                        self.assertEqual(sum(e.steps for e in model.history_), 10_000)
                        self.assertEqual(model.best_observed_.evaluation.npv, target)
                        self.assertEqual(model.score(model.instance_), target)
                        self.assertTrue(all(r.evaluation.feasible for r in model.policy_))
                        if state == "physical":
                            self.assertTrue(all(r.evaluation.npv == target for r in model.policy_))

    def test_position_key_with_gamma_one_overestimates_and_is_not_bound_checked(self):
        mine, rules = tiny_mine()
        model = fit(mine, rules, state="position", policy_rollouts=1, environment=EnvironmentParameters(50, "external"))
        self.assertGreater(max(model.q_.values()), sum(b.value for b in mine.blocks if b.value > 0))

    def test_position_key_keeps_the_clock_only_in_finite_mode(self):
        mine, rules = tiny_mine()
        model = QLearningEstimator(state="position")
        for horizon, clock in (("finite", 11), ("external", 0)):
            env = MineEnv(mine, ProblemParameters(), rules, EnvironmentParameters(12, horizon))
            state, _, _ = env.step("start")
            self.assertEqual(model.state_key(state), PositionKey(state.position, 0, clock))
        physical = QLearningEstimator()
        other = replace(state, extracted=frozenset())
        self.assertNotEqual(physical.state_key(state), physical.state_key(other))
        self.assertEqual(model.state_key(state), model.state_key(other))

    def test_real_end_does_not_bootstrap_but_interruptions_do(self):
        mine, rules = tiny_mine()

        class Primed(QLearningEstimator):
            def update(self, state, action, reward, following, actions, terminal):
                self.q_[self.state_key(following), "stop"] = 7
                return super().update(state, action, reward, following, actions, terminal)

        start_value = 5  # value of the start block's cone in the tiny mine
        for horizon, cap, expected, end in (("finite", 12, start_value + 7, "budget"),
                                            ("finite", 1, start_value, "step_limit"),
                                            ("external", 1, start_value + 7, "budget")):
            with self.subTest(horizon=horizon, cap=cap):
                model = fit(mine, rules, estimator=Primed, transitions=1, policy_rollouts=1,
                            environment=EnvironmentParameters(cap, horizon))
                self.assertEqual(next(v for (_, a), v in model.q_.items() if a == "start"), expected)
                self.assertEqual(model.history_[0].end, end)

    def test_external_cap_interrupts_episodes(self):
        mine, rules = tiny_mine()
        model = fit(mine, rules, transitions=3, policy_rollouts=1, environment=EnvironmentParameters(1, "external"))
        self.assertEqual([e.end for e in model.history_], ["episode_cap", "episode_cap", "budget"])
        self.assertEqual(model.policy_[0].end, "episode_cap")

    def test_seed_reproducibility_and_rollouts_do_not_touch_training(self):
        mine, rules = tiny_mine()
        left = fit(mine, rules, transitions=201, policy_rollouts=3, random_state=3)
        right = fit(mine, rules, transitions=201, policy_rollouts=7, random_state=3)
        self.assertEqual(left.q_, right.q_)
        self.assertEqual(left.rng_.getstate(), right.rng_.getstate())
        self.assertEqual(left.history_, right.history_)
        self.assertEqual(left.best_observed_, right.best_observed_)
        self.assertEqual(left.policy_, right.policy_[:3])

    def test_multi_period_capacity_and_discount_are_supported(self):
        mine, rules = tiny_mine()
        problem = ProblemParameters(periods=2, discount_rate=0.1, capacity=3)
        model = fit(mine, rules, problem=problem, transitions=2_000, policy_rollouts=3,
                    environment=EnvironmentParameters(12, "external"))
        self.assertTrue(model.best_observed_.evaluation.feasible)
        self.assertTrue(all(len([e for e in r.plan.extractions if e.period == p]) <= 3
                            for r in model.policy_ for p in (0, 1)))

    def test_predict_delivers_the_best_observed_plan_of_its_own_instance_only(self):
        mine, rules = tiny_mine()
        model = fit(mine, rules, transitions=500, policy_rollouts=2)
        self.assertEqual(model.predict(model.instance_), model.best_observed_.plan)
        self.assertEqual(model.score(model.instance_), model.best_observed_.evaluation.npv)
        other = replace(model.instance_, environment=EnvironmentParameters(12, "external"))
        with self.assertRaisesRegex(ValueError, "misma instancia"):
            model.predict(other)

    def test_compare_reference_counts_the_policy_rollouts_that_match(self):
        mine, rules = tiny_mine()
        environment = EnvironmentParameters(12, "finite")
        optimum = enumerate_tiny(mine, rules, environment).optimum  # every rollout reaches it (see the gate)
        model = fit(mine, rules, environment=environment)
        self.assertEqual(model.compare_reference(optimum),
                         {"policy_matches": 20, "policy_rollouts": 20, "policy_always_matches": 1})
        self.assertEqual(model.compare_reference(optimum + 1),
                         {"policy_matches": 0, "policy_rollouts": 20, "policy_always_matches": 0})

    def test_observe_replaces_the_direct_update_and_counters_follow(self):
        """observe is the hook of TabularQ: overriding it decides whether the real
        transition updates Q at all (prioritized sweeping queues it instead)."""
        mine, rules = tiny_mine()
        model = fit(mine, rules, transitions=300, policy_rollouts=1)
        self.assertEqual(model.counters_, {"transitions": 300, "simulator_calls": 300, "updates": 300})
        self.assertEqual(model.trace_[-1][1], model.best_observed_.evaluation.npv)

        class Silent(QLearningEstimator):
            def observe(self, *transition):
                self.seen = getattr(self, "seen", 0) + 1

        silent = fit(mine, rules, estimator=Silent, transitions=300, policy_rollouts=1)
        self.assertEqual((silent.q_, silent.seen, silent.counters_["updates"]), ({}, 300, 0))

    def test_td_error_does_not_change_q(self):
        mine, rules = tiny_mine()
        model = QLearningEstimator(alpha=0.5)
        model._reset_learning()
        env = MineEnv(mine, ProblemParameters(), rules, EnvironmentParameters(12, "finite"))
        state = env.reset()
        following, reward, terminal = env.step("start")
        error = model.td_error(state, "start", reward, following, env.available_actions(), terminal)
        self.assertEqual((error, model.q_), (reward, {}))
        self.assertEqual(model.update(state, "start", reward, following, env.available_actions(), terminal), reward / 2)

    def test_invalid_parameters(self):
        for kind, values in ((QLearningParams, {"state": "full"}), (QLearningParams, {"alpha": 0}),
                             (QLearningParams, {"gamma": 1.1}), (QLearningParams, {"transitions": 0}),
                             (QLearningParams, {"policy_rollouts": 0}), (EnvironmentParameters, {"max_steps": 0}),
                             (EnvironmentParameters, {"horizon": "implicit"})):
            with self.subTest(kind=kind.__name__, values=values), self.assertRaises(ValueError):
                kind(**values)
        mine, rules = tiny_mine()
        for values in ({"alpha": 0}, {"epsilon_schedule": "cubic"}, {"random_state": -1}, {"random_state": True}):
            with self.subTest(estimator=values), self.assertRaises(ValueError):
                fit(mine, rules, **values)


if __name__ == "__main__":
    unittest.main()
