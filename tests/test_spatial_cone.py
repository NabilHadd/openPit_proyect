import unittest

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import Precedences, grid9_z_up, grid9_z_down
from src.envs.mine_env import ADVANCE, STOP
from src.envs.spatial_cone_env import START, SpatialConeEnv, SpatialState
from src.evaluation.metrics import evaluate_plan
from src.evaluation.reporter import q_table_payload
from src.models.q_learning import SpatialQLearning
from src.training.trainer import run_episode, train


def layered(parameters=ProblemParameters(), max_steps=20, construction_up=False):
    mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                 Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1),
                 Block(5, 5, 5, 3, -1, tonnes=1)))
    return SpatialConeEnv(mine, parameters, grid9_z_up(mine), SpatialParameters("max-z", max_steps, construction_up))


class SpatialTests(unittest.TestCase):
    def test_start_reward_and_surface_tie_are_counted(self):
        env = layered()
        self.assertEqual(env.plan.extractions, ())
        self.assertEqual(env.available_actions(), (START,))
        state, reward, done = env.step(START)
        self.assertEqual((state.position, reward, done), (1, 5, False))
        self.assertEqual(tuple(e.block_id for e in env.plan.extractions), (1,))
        with self.assertRaises(ValueError):
            env.step(START)
        tied_mine = Mine((Block(9, 0, 0, 1, 7), Block(2, 1, 0, 1, 7)))
        tied_rules = Precedences({9: frozenset(), 2: frozenset()}, "synthetic", "test")
        tied = SpatialConeEnv(tied_mine, ProblemParameters(), tied_rules, SpatialParameters("max-z"))
        self.assertEqual(tied.start_id, 2)

    def test_transitive_cones_overlap_and_repeatability(self):
        env = layered()
        def trajectory():
            env.reset()
            return tuple(env.step(a) for a in (START, "down", "down", "up", "up", STOP))
        first = trajectory()
        self.assertEqual(first, trajectory())
        self.assertEqual([t[1] for t in first], [5, -6, 15, 0, 0, 0])
        self.assertEqual(tuple(e.block_id for e in env.plan.extractions), (1, 4, 2, 3))

    def test_shared_cone_extracts_each_block_once(self):
        mine = Mine((Block(1, 0, 0, 2, 9), Block(2, 1, 0, 2, -2),
                     Block(3, 0, 0, 1, 20), Block(4, 1, 0, 1, 30), Block(5, 5, 5, 2, -1)))
        env = SpatialConeEnv(mine, ProblemParameters(), grid9_z_up(mine), SpatialParameters("max-z"))
        rewards = [env.step(a)[1] for a in (START, "down", "south")]
        self.assertEqual(rewards, [9, 18, 30])
        self.assertEqual(len(env.plan.extractions), 4)
        result = evaluate_plan(mine, env.plan, env.parameters, env.precedences)
        self.assertTrue(result.feasible)
        self.assertEqual(result.npv, sum(rewards))

    def test_joint_capacity_rejection_is_atomic_and_advance_explicit(self):
        env = layered(ProblemParameters(periods=2, capacity=4, capacity_unit="tonnes", discount_rate=0.5))
        env.step(START)
        before = env.state, env.plan, env.action_history
        self.assertNotIn("down", env.available_actions())
        with self.assertRaises(ValueError):
            env.step("down")
        self.assertEqual((env.state, env.plan, env.action_history), before)
        self.assertEqual(env.step(ADVANCE)[1], 0)
        self.assertEqual(env.step("down")[1], -4)
        self.assertEqual(env.state.used, 4)
        self.assertNotIn("down", env.available_actions())

    def test_phase_masks_budget_and_terminal_bootstrap(self):
        env = layered(max_steps=4)
        env.reset(training=False)
        env.step(START)
        env.step("down")
        self.assertNotIn("up", env.available_actions())
        env.reset(training=True)
        env.step(START)
        env.step("down")
        self.assertIn("up", env.available_actions())
        env.step("up")
        previous = env.state
        following, reward, terminal = env.step("down")
        self.assertTrue(terminal)
        self.assertEqual(env.termination_reason, "step_limit")
        model = SpatialQLearning(LearningParameters(alpha=1, gamma=0.7), 0)
        model.q[model.state_key(following), "up"] = 999
        self.assertEqual(model.update(previous, "down", reward, following, (), True), 0)
        extension = layered(construction_up=True)
        extension.reset(training=False)
        extension.step(START)
        extension.step("down")
        self.assertIn("up", extension.available_actions())

    def test_compact_key_aliasing_does_not_replace_physical_mask(self):
        env = layered(ProblemParameters(periods=3, capacity=4, capacity_unit="tonnes"))
        # Same position/period, different feasible histories and current capacity usage.
        before = SpatialState(frozenset({1}), 2, 2, False, 1, 19, "learning")
        env.state = before
        model = SpatialQLearning(LearningParameters(), 2)
        other = SpatialState(frozenset({1, 2, 4}), 2, 0, False, 1, 19, "learning")
        self.assertEqual(model.state_key(before), model.state_key(other))
        self.assertNotIn("down", env.available_actions())
        env.state = other
        self.assertIn("down", env.available_actions())  # transit through an extracted block consumes zero
        model.q[model.state_key(before), "down"] = 10
        self.assertEqual(q_table_payload(model)["entries"][0]["state"], {"position": 1, "period": 2})

    def test_orientation_min_z_and_invalid_configuration(self):
        mine = Mine((Block(1, 0, 0, 1, 5), Block(2, 0, 0, 2, 9)))
        env = SpatialConeEnv(mine, ProblemParameters(), grid9_z_down(mine), SpatialParameters("min-z"))
        env.step(START)
        self.assertIn("down", env.available_actions())
        _, reward, terminal = env.step("down")
        self.assertEqual(reward, 9)
        self.assertTrue(terminal)
        for arguments in (("invalid", 5), ("max-z", 0)):
            with self.assertRaises(ValueError):
                SpatialParameters(*arguments)

    def test_training_seed_greedy_readonly_and_bound(self):
        parameters = LearningParameters(episodes=20, alpha=1, gamma=0.7, epsilon=0.3)
        def experiment():
            env, model = layered(max_steps=12), SpatialQLearning(parameters, 3)
            result = train(env, model, parameters.episodes)
            before = model.q.copy(), model.rng.getstate()
            run_episode(env, model, learn=False)
            self.assertEqual(before, (model.q, model.rng.getstate()))
            self.assertTrue(all(r.steps <= 12 for r in result.history))
            self.assertEqual(result.greedy_steps, len(result.greedy_actions))
            return result, model.q
        self.assertEqual(experiment(), experiment())


if __name__ == "__main__":
    unittest.main()
