import unittest

from src.config import LearningParameters, ProblemParameters
from src.data.dataset import Block, Mine
from src.data.transforms import Precedences, grid9_z_up
from src.envs.mine_env import ADVANCE, STOP, MineEnv, State
from src.evaluation.metrics import evaluate_plan
from src.models.q_learning import QLearning
from src.training.trainer import run_episode, train


def fixture(parameters=ProblemParameters()):
    mine = Mine((Block(10, 1, 1, 2, -4, tonnes=2), Block(20, 1, 1, 1, 15, tonnes=3)))
    return MineEnv(mine, parameters, grid9_z_up(mine))


class EnvTests(unittest.TestCase):
    def test_reset_trajectory_and_atomic_invalid(self):
        env = fixture()
        self.assertEqual(env.available_actions(), (10, STOP))
        initial, plan = env.state, env.plan
        for invalid in (20, 999, ADVANCE, True, 10.0):
            with self.assertRaises(ValueError):
                env.step(invalid)
            self.assertEqual((env.state, env.plan), (initial, plan))
        def trajectory():
            env.reset()
            return [env.step(10), env.step(20)]
        first = trajectory()
        self.assertEqual(first, trajectory())
        self.assertTrue(first[-1][-1])
        self.assertEqual(env.available_actions(), ())
        with self.assertRaises(ValueError):
            env.step(STOP)
        self.assertEqual(sum(t[1] for t in first), 11)

    def test_explicit_period_and_discount(self):
        env = fixture(ProblemParameters(periods=2, capacity=1, discount_rate=0.5))
        _, reward, done = env.step(10)
        self.assertEqual(reward, -4)
        self.assertFalse(done)
        self.assertEqual(env.available_actions(), (ADVANCE, STOP))
        self.assertEqual(env.step(ADVANCE)[1], 0)
        self.assertEqual(env.state.period, 1)
        self.assertEqual(env.step(20)[1], 10)
        self.assertEqual(evaluate_plan(env.mine, env.plan, env.parameters, env.precedences).npv, 6)

    def test_tonnage_and_stop(self):
        env = fixture(ProblemParameters(periods=2, capacity=3, capacity_unit="tonnes"))
        env.step(10)
        self.assertNotIn(20, env.available_actions())
        env.step(ADVANCE)
        self.assertIn(20, env.available_actions())
        self.assertTrue(env.step(STOP)[2])
        self.assertEqual(len(env.plan.extractions), 1)

    def test_last_period_exhaustion(self):
        env = fixture(ProblemParameters(capacity=1))
        self.assertTrue(env.step(10)[2])
        empty = fixture(ProblemParameters(capacity=1, capacity_unit="tonnes"))
        self.assertTrue(empty.state.terminal)


class LearningTests(unittest.TestCase):
    def test_update_and_terminal_no_bootstrap(self):
        model = QLearning(LearningParameters(alpha=0.5, gamma=0.9), 7)
        state, following = State(frozenset(), 0, 0), State(frozenset({10}), 0, 0)
        model.q[state, 10] = 2
        model.q[following, 20] = 10
        model.q[following, 999] = 1000
        self.assertAlmostEqual(model.update(state, 10, 4, following, (20,), False), 7.5)
        model.q[state, 10] = 2
        self.assertEqual(model.update(state, 10, 4, following, (20,), True), 3)
        self.assertEqual(model.update(state, STOP, -2, following, (), True), -1)

    def test_action_mask_exploration_and_no_mutation_on_eval(self):
        state = State(frozenset(), 0, 0)
        for epsilon in (0, 1):
            model = QLearning(LearningParameters(epsilon=epsilon), 10)
            model.q[state, 999] = 1000
            self.assertTrue(all(model.select_action(state, (10, STOP)) in (10, STOP) for _ in range(50)))
            before, rng_before = model.q.copy(), model.rng.getstate()
            model.select_action(state, (10, STOP), explore=False)
            self.assertEqual(model.q, before)
            self.assertEqual(model.rng.getstate(), rng_before)

    def test_seed_and_best_episode(self):
        parameters = LearningParameters(episodes=50)
        def experiment():
            env = fixture()
            model = QLearning(parameters, 4)
            result = train(env, model, parameters.episodes)
            before = model.q.copy()
            run_episode(env, model, learn=False)
            self.assertEqual(before, model.q)
            return result, model.q
        a, b = experiment(), experiment()
        self.assertEqual(a, b)
        result = a[0]
        self.assertEqual(result.best_evaluation.npv, max(r.evaluation.npv for r in result.history))

    def test_hand_optimum_without_private_data(self):
        mine = Mine((Block(1, 1, 1, 1, -4), Block(2, 2, 1, 1, 15)))
        prec = Precedences({1: frozenset(), 2: frozenset({1})}, "synthetic", "test")
        result = train(MineEnv(mine, ProblemParameters(), prec),
                       QLearning(LearningParameters(epsilon=0.4), 8), 300)
        self.assertEqual(result.best_evaluation.npv, 11)
        self.assertEqual(result.greedy_evaluation.npv, 11)


if __name__ == "__main__":
    unittest.main()
