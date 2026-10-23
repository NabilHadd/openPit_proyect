import unittest
from unittest.mock import patch

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import grid9_z_up
from src.envs.spatial_cone_env import SpatialConeEnv
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine
from src.models.experimental_q_learning import ControlQLearning
from src.models.validated_q_learning import ValidationQLearning
from src.training.validation_runner import ValidationBudget, train_transition_budget


class ValidationLearningTests(unittest.TestCase):
    def test_ten_seeds_learn_and_reproduce_independent_tiny_optimum(self):
        six, six_rules = tiny_mine()
        four = Mine(six.blocks[:4])
        for mine, rules in ((four, grid9_z_up(four)), (six, six_rules)):
            spatial = SpatialParameters("max-z", 12, True)
            target = enumerate_tiny(mine, rules, spatial).optimum
            for seed in range(10):
                with self.subTest(blocks=len(mine.blocks), seed=seed):
                    run = train_transition_budget(mine, rules, spatial, LearningParameters(alpha=1, gamma=1, epsilon=.3),
                                                  seed, ValidationBudget(10000, 20))
                    self.assertEqual(run.training["transitions"], 10000)
                    self.assertEqual(sum(h["steps"] for h in run.training["history"]), 10000)
                    self.assertEqual(run.best_observed.evaluation.npv, target)
                    self.assertEqual(run.best_observed.phase, "learning")
                    self.assertTrue(all(p.path.evaluation.feasible and p.path.evaluation.npv == target for p in run.policy))
                    self.assertTrue(run.training["q_bounds"]["passed"])

    def test_evaluation_ties_are_uniform_reproducible_and_do_not_touch_training_rng(self):
        mine, rules = tiny_mine()
        env = SpatialConeEnv(mine, ProblemParameters(), rules, SpatialParameters("max-z", 12, True))
        env.step("start")
        model = ValidationQLearning(LearningParameters(), 7)
        actions = env.available_actions()
        before = model.q.copy(), model.rng.getstate()
        sampled = [model.select_action(env.state, actions, explore=False) for _ in range(200)]
        self.assertEqual(set(sampled), set(actions))
        self.assertEqual(before, (model.q, model.rng.getstate()))
        model.reset_evaluation(0)
        self.assertEqual(sampled, [model.select_action(env.state, actions, explore=False) for _ in range(200)])
        control = ValidationQLearning(LearningParameters(), 7)
        self.assertEqual([model.select_action(env.state, actions) for _ in range(20)],
                         [control.select_action(env.state, actions) for _ in range(20)])

    def test_true_terminal_vs_external_truncation_bootstrap(self):
        mine, rules = tiny_mine()
        class PrimedModel(ValidationQLearning):
            def update(self, state, action, reward, following, actions, terminal):
                self.q[self.state_key(following), "stop"] = 7
                return super().update(state, action, reward, following, actions, terminal)

        for mode, cap, expected, reason in (("finite", 12, 12, "transition_budget"),
                                            ("finite", 1, 5, "step_limit"),
                                            ("external", 1, 12, "transition_budget")):
            with self.subTest(mode=mode, cap=cap), patch("src.training.validation_runner.ValidationQLearning", PrimedModel):
                run = train_transition_budget(mine, rules, SpatialParameters("max-z", cap, True, mode),
                                              LearningParameters(alpha=1), 0, ValidationBudget(1, 1))
                self.assertEqual(next(v for (s, a), v in run.model.q.items() if a == "start"), expected)
                record = run.training["history"][0]
                self.assertEqual(record["reason"], reason)
                self.assertEqual(record["terminated"], reason == "step_limit")
                self.assertEqual(record["truncated"], reason != "step_limit")

    def test_external_episode_cap_is_not_stop_and_keeps_stationary_clock(self):
        mine, rules = tiny_mine()
        spatial = SpatialParameters("max-z", 1, True, "external")
        env = SpatialConeEnv(mine, ProblemParameters(), rules, spatial)
        state, _, terminal = env.step("start")
        self.assertEqual(state.steps_left, 0)
        self.assertFalse(terminal)
        self.assertIn("stop", env.available_actions())
        run = train_transition_budget(mine, rules, spatial, LearningParameters(alpha=1), 0, ValidationBudget(3, 1))
        self.assertEqual([r["reason"] for r in run.training["history"]],
                         ["external_episode_limit", "external_episode_limit", "transition_budget"])
        self.assertTrue(all(r["truncated"] and not r["terminated"] for r in run.training["history"]))
        self.assertTrue(run.policy[0].truncated)
        self.assertEqual(run.policy[0].reason, "external_episode_limit")
        self.assertIsNone(run.best_completed)

    def test_stop_is_a_true_terminal_and_not_a_navigation_cycle(self):
        mine, rules = tiny_mine()
        def scripted(model, state, actions, *, explore=True):
            return "start" if state.position is None else "stop"
        with patch.object(ValidationQLearning, "select_action", scripted):
            run = train_transition_budget(mine, rules, SpatialParameters("max-z", 12, True),
                                          LearningParameters(alpha=1), 0, ValidationBudget(2, 1))
        row = run.training["history"][0]
        self.assertEqual((row["reason"], row["terminated"], row["truncated"]), ("stop", True, False))
        self.assertEqual(row["decisions"].get("navigation_context_returns", 0), 0)

    def test_seed_reproducibility_ignoring_only_timings(self):
        mine, rules = tiny_mine()
        def execute():
            return train_transition_budget(mine, rules, SpatialParameters("max-z", 12, True),
                                           LearningParameters(alpha=1), 3, ValidationBudget(201, 3))
        left, right = execute(), execute()
        self.assertEqual(left.model.q, right.model.q)
        self.assertEqual(left.policy, right.policy)
        self.assertEqual({k: v for k, v in left.training.items() if k != "learning_seconds"},
                         {k: v for k, v in right.training.items() if k != "learning_seconds"})
        self.assertEqual(left.best_observed, right.best_observed)

    def test_observers_and_final_evaluation_do_not_change_learning(self):
        mine, rules = tiny_mine()
        spatial = SpatialParameters("max-z", 12, True)
        parameters = LearningParameters(alpha=1)
        run = train_transition_budget(mine, rules, spatial, parameters, 3, ValidationBudget(201, 3))
        plain = ControlQLearning(parameters, 3, "physical")
        env = SpatialConeEnv(mine, ProblemParameters(), rules, spatial)
        for _ in range(201):
            if env.state.terminal:
                env.reset()
            state = env.state
            action = plain.select_action(state, env.available_actions())
            following, reward, terminal = env.step(action)
            plain.update(state, action, reward, following, env.available_actions(), terminal)
        self.assertEqual(run.model.q, plain.q)
        self.assertEqual(run.model.rng.getstate(), plain.rng.getstate())

    def test_validation_rejects_gamma_discount_and_invalid_budgets(self):
        with self.assertRaises(ValueError):
            ValidationQLearning(LearningParameters(gamma=.7), 0)
        for args in ((0, 1), (1, 0), (True, 1)):
            with self.assertRaises(ValueError):
                ValidationBudget(*args)
        with self.assertRaises(ValueError):
            SpatialParameters("max-z", horizon_mode="implicit")


if __name__ == "__main__":
    unittest.main()
