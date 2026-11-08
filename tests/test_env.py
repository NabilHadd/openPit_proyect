from collections import deque
from dataclasses import replace
import unittest

from src.config import ProblemParameters, TrainingParameters
from src.data.dataset import Block, Mine
from src.data.precedences import Precedences
from src.envs.mine_env import ADVANCE, START, STOP, MineEnv, State
from src.evaluation.metrics import evaluate_plan


def layered(problem=ProblemParameters(), max_steps=20, horizon="finite"):
    mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                 Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1),
                 Block(5, 5, 5, 3, -1, tonnes=1)))
    rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: (), 5: ()})
    return MineEnv(mine, problem, rules, TrainingParameters(1, max_steps, horizon))


class EnvironmentTests(unittest.TestCase):
    def test_start_reward_and_surface_tie(self):
        env = layered()
        self.assertEqual(env.available_actions(), (START,))
        state, reward, done = env.step(START)
        self.assertEqual((state.position, reward, done), (1, 5, False))
        with self.assertRaises(ValueError):
            env.step(START)
        tied = MineEnv(Mine((Block(9, 0, 0, 1, 7), Block(2, 1, 0, 1, 7))),
                       ProblemParameters(), Precedences({9: (), 2: ()}), TrainingParameters())
        self.assertEqual(tied.start_id, 2)

    def test_cones_revisits_and_six_directions_always(self):
        env = layered()
        rewards = [env.step(a)[1] for a in (START, "down", "down", "up", "up", STOP)]
        self.assertEqual(rewards, [5, -6, 15, 0, 0, 0])
        self.assertEqual(tuple(e.block_id for e in env.plan.extractions), (1, 4, 2, 3))
        env.reset()
        env.step(START)
        env.step("down")
        self.assertIn("up", env.available_actions())

    def test_capacity_rejection_is_atomic_and_advance_explicit(self):
        env = layered(ProblemParameters(periods=2, capacity=4, capacity_unit="tonnes", discount_rate=0.5))
        env.step(START)
        before = env.state, env.plan, env.action_history
        self.assertNotIn("down", env.available_actions())
        with self.assertRaises(ValueError):
            env.step("down")
        self.assertEqual((env.state, env.plan, env.action_history), before)
        self.assertEqual(env.step(ADVANCE)[1], 0)
        self.assertEqual(env.step("down")[1], -4)  # cone {4, 2}: (-2 - 4) / 1.5 in period 1
        self.assertEqual(env.state.used, 4)

    def test_finite_clock_is_in_state_and_external_clock_is_constant(self):
        finite = layered(max_steps=2)
        state, _, _ = finite.step(START)
        self.assertEqual(state.steps_left, 1)
        state, _, terminal = finite.step("down")
        self.assertTrue(terminal)
        self.assertEqual(finite.termination_reason, "step_limit")
        external = layered(max_steps=2, horizon="external")
        for action in (START, "down", "up"):
            state, _, terminal = external.step(action)
            self.assertEqual(state.steps_left, 0)
            self.assertFalse(terminal)  # the environment never ends an external episode by steps

    def test_unreachable_blocks_are_never_destinations(self):
        mine = Mine((Block(1, 0, 0, 2, 5), Block(4, 1, 0, 2, -1),
                     Block(2, 0, 0, 1, 50, reachable=False), Block(3, 1, 0, 1, 7)))
        env = MineEnv(mine, ProblemParameters(), Precedences({1: (), 4: (), 2: (1,), 3: (4,)}), TrainingParameters())
        env.step(START)
        self.assertNotIn("down", env.available_actions())
        self.assertEqual(env.step("south")[1], -1)
        _, reward, terminal = env.step("down")
        self.assertEqual((reward, terminal, env.termination_reason), (7, True, "all_extracted"))

    def test_every_reachable_transition_keeps_plans_feasible_and_rewards_equal_npv(self):
        mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                     Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1)))
        rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: ()})
        for problem in (ProblemParameters(), ProblemParameters(3, 0.15, 3), ProblemParameters(3, 0.15, 4, "tonnes")):
            def make(actions=()):
                env = MineEnv(mine, problem, rules, TrainingParameters(1, 8, "finite"))
                return env, sum(env.step(a)[1] for a in actions)
            queue, seen, transitions = deque([((), make()[0].state)]), set(), 0
            while queue:
                prefix, state = queue.popleft()
                env, _ = make(prefix)
                self.assertEqual(state, env.state)
                for action in env.available_actions():
                    left, total = make(prefix)
                    right, _ = make(prefix)
                    right.plan = replace(right.plan, extractions=tuple(reversed(right.plan.extractions)))
                    following, reward, _ = left.step(action)
                    self.assertEqual(following, right.step(action)[0])  # plan history never changes dynamics
                    evaluation = evaluate_plan(mine, left.plan, problem, rules)
                    self.assertTrue(evaluation.feasible, evaluation.errors)
                    self.assertAlmostEqual(total + reward, evaluation.npv)
                    transitions += 1
                    if following not in seen:
                        seen.add(following)
                        queue.append((prefix + (action,), following))
            self.assertGreater(transitions, 20)

    def test_state_has_no_phase(self):
        self.assertNotIn("phase", State.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
