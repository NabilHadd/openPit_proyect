import unittest
from unittest.mock import patch

from src.config import ProblemParameters, TrainingParameters
from src.data.dataset import Block, Mine
from src.data.precedences import Precedences
from src.envs.mine_env import MineEnv
from src.evaluation.metrics import evaluate_plan
from src.evaluation.tiny_oracle import enumerate_tiny, tiny_mine


class TinyOracleTests(unittest.TestCase):
    def test_independent_enumeration_and_environment_witness_agree(self):
        mine, rules = tiny_mine()
        spatial = TrainingParameters(1, 12, "finite")
        with patch.object(MineEnv, "step", side_effect=AssertionError("Oracle used the environment")):
            oracle = enumerate_tiny(mine, rules, spatial)
        self.assertEqual((oracle.optimum, oracle.closed_subset_optimum), (15, 15))
        self.assertEqual(set(oracle.block_ids), {1, 2, 3, 4})
        env = MineEnv(mine, ProblemParameters(), rules, spatial)
        rewards = [env.step(a)[1] for a in oracle.actions]
        self.assertIn(-5, rewards)
        self.assertEqual(sum(rewards), 15)
        self.assertTrue(env.state.terminal)
        self.assertTrue(evaluate_plan(mine, env.plan, env.parameters, rules).feasible)
        self.assertGreater(oracle.reachable_states, 50)

    def test_action_optimum_is_not_an_unreachable_subset_bound(self):
        mine, rules = tiny_mine()
        short = enumerate_tiny(mine, rules, TrainingParameters(1, 2, "finite"))
        self.assertEqual((short.optimum, short.closed_subset_optimum), (5, 15))
        mine = Mine((Block(1, 0, 0, 3, -2),))
        rules = Precedences({1: ()}, "explicit", "test")
        forced = enumerate_tiny(mine, rules, TrainingParameters(1, 2, "finite"))
        self.assertEqual((forced.optimum, forced.closed_subset_optimum), (-2, 0))

    def test_external_collection_cap_does_not_limit_the_task_optimum(self):
        mine, rules = tiny_mine()
        capped = enumerate_tiny(mine, rules, TrainingParameters(1, 1, "finite"))
        external = enumerate_tiny(mine, rules, TrainingParameters(1, 1, "external"))
        self.assertEqual(capped.optimum, 5)
        self.assertEqual(external.optimum, 15)
        self.assertGreater(len(external.actions), 1)

    def test_rejects_non_tiny_scenarios(self):
        mine, rules = tiny_mine()
        mine = Mine(tuple(Block(i, i, 0, 3, i) for i in range(7)))
        rules = Precedences({i: () for i in range(7)}, "explicit", "test")
        with self.assertRaises(ValueError):
            enumerate_tiny(mine, rules, TrainingParameters(1, 12, "finite"))


if __name__ == "__main__":
    unittest.main()
