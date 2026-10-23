"""Reachable-state checks independent of the agent and its Q projection."""

from collections import deque
from dataclasses import replace
import unittest

from src.config import ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import grid9_z_up
from src.envs.spatial_cone_env import SpatialConeEnv
from src.evaluation.metrics import evaluate_plan


class EnvironmentContractTests(unittest.TestCase):
    def test_every_reachable_transition_preserves_physics_and_reward_accounting(self):
        mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                     Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1)))
        rules = grid9_z_up(mine)
        for problem in (ProblemParameters(), ProblemParameters(3, 0.15, 3),
                        ProblemParameters(3, 0.15, 4, "tonnes")):
            def make(actions=()):
                env = SpatialConeEnv(mine, problem, rules, SpatialParameters("max-z", 8, True))
                total = 0
                for action in actions:
                    total += env.step(action)[1]
                return env, total

            initial, _ = make()
            queue = deque([(initial.state, ())])
            seen = {initial.state}
            transitions = collisions = 0
            while queue:
                state, prefix = queue.popleft()
                env, total = make(prefix)
                self.assertEqual(state, env.state)
                if state.terminal:
                    self.assertEqual(env.available_actions(), ())
                for action in env.available_actions():
                    left, left_total = make(prefix)
                    right, _ = make(prefix)
                    # Plans are historical output; neither ordering of already extracted
                    # blocks nor the learning/construction label may change the dynamics.
                    right.plan = replace(right.plan, extractions=tuple(reversed(right.plan.extractions)))
                    right.state = replace(right.state, phase="construction")
                    before = left.state
                    following, reward, terminal = left.step(action)
                    other, other_reward, other_terminal = right.step(action)
                    self.assertEqual(replace(following, phase="construction"), other)
                    self.assertEqual((reward, terminal, left.available_actions()),
                                     (other_reward, other_terminal, right.available_actions()))
                    ids = [e.block_id for e in left.plan.extractions]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), following.extracted)
                    self.assertLessEqual(before.extracted, following.extracted)
                    evaluation = evaluate_plan(mine, left.plan, problem, rules)
                    self.assertTrue(evaluation.feasible, evaluation.errors)
                    direct_sum = sum(mine.by_id[e.block_id].value / (1 + problem.discount_rate) ** e.period
                                     for e in left.plan.extractions)
                    self.assertAlmostEqual(left_total + reward, direct_sum)
                    self.assertAlmostEqual(left_total + reward, evaluation.npv)
                    if problem.capacity is not None:
                        self.assertAlmostEqual(following.used, evaluation.period_usage[following.period])
                    transitions += 1
                    if following not in seen:
                        seen.add(following)
                        queue.append((following, prefix + (action,)))
                    else:
                        collisions += 1
                snapshot = env.state, env.plan, env.action_history
                with self.assertRaises(ValueError):
                    env.step("invalid-action")
                self.assertEqual(snapshot, (env.state, env.plan, env.action_history))
            self.assertGreater(transitions, 20)
            self.assertGreater(collisions, 0)


if __name__ == "__main__":
    unittest.main()
