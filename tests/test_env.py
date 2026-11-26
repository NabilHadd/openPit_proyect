from collections import deque
from dataclasses import replace
import unittest

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.dataset import Block, Mine
from src.data.precedences import Precedences
from src.envs.mine_env import ADVANCE, START, STOP, MineEnv, State
from src.evaluation.metrics import Extraction, Plan, evaluate_plan


def layered(problem=ProblemParameters(), max_steps=20, horizon="finite"):
    mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                 Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1),
                 Block(5, 5, 5, 3, -1, tonnes=1)))
    rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: (), 5: ()})
    return MineEnv(mine, problem, rules, EnvironmentParameters(max_steps, horizon))


class EnvironmentTests(unittest.TestCase):
    def test_start_reward_and_surface_tie(self):
        env = layered()
        self.assertEqual(env.available_actions(), (START,))
        state, reward, done = env.step(START)
        self.assertEqual((state.position, reward, done), (1, 5, False))
        with self.assertRaises(ValueError):
            env.step(START)
        tied = MineEnv(Mine((Block(9, 0, 0, 1, 7), Block(2, 1, 0, 1, 7))),
                       ProblemParameters(), Precedences({9: (), 2: ()}), EnvironmentParameters())
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
        env = MineEnv(mine, ProblemParameters(), Precedences({1: (), 4: (), 2: (1,), 3: (4,)}), EnvironmentParameters())
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
                env = MineEnv(mine, problem, rules, EnvironmentParameters(8, "finite"))
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

    def test_pure_transition_matches_the_live_episode_and_changes_nothing(self):
        """BFS over every reachable state with the pure simulator; each state is replayed
        on a live environment, whose step must give the same result."""
        mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                     Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1)))
        rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: ()})
        for problem in (ProblemParameters(), ProblemParameters(3, 0.15, 3), ProblemParameters(3, 0.15, 4, "tonnes")):
            for horizon in ("finite", "external"):
                simulator = MineEnv(mine, problem, rules, EnvironmentParameters(6, horizon))
                start = simulator.initial_state()
                self.assertEqual(start, simulator.reset())
                queue, seen, checked = deque([((), start)]), {start}, 0
                while queue:
                    prefix, state = queue.popleft()
                    live = MineEnv(mine, problem, rules, EnvironmentParameters(6, horizon))
                    for action in prefix:
                        live.step(action)
                    self.assertEqual(live.state, state)
                    self.assertEqual(simulator.actions(state), live.available_actions())
                    for action in simulator.actions(state):
                        before = simulator.state, simulator.plan, simulator.calls
                        result = simulator.transition(state, action)
                        self.assertEqual((simulator.state, simulator.plan, simulator.calls),
                                         (before[0], before[1], before[2] + 1))  # only the counter moves
                        replay = MineEnv(mine, problem, rules, EnvironmentParameters(6, horizon))
                        for previous in prefix:
                            replay.step(previous)
                        extractions = replay.plan.extractions
                        self.assertEqual(replay.step(action), (result.state, result.reward, result.terminal))
                        self.assertEqual(replay.termination_reason, result.reason)
                        self.assertEqual(replay.plan.extractions[len(extractions):],
                                         tuple(Extraction(i, result.state.period) for i in result.extracted))
                        checked += 1
                        if not result.terminal and result.state not in seen:
                            seen.add(result.state)
                            queue.append((prefix + (action,), result.state))
                self.assertGreater(checked, 10)

    def test_extracting_a_cone_equals_the_cursor_move_that_extracts_it(self):
        """On every reachable cursor state, extract(state, d) gives what the move to d gives
        (blocks, reward, capacity, period, end), except that the step clock does not run."""
        mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                     Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1)))
        rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: ()})
        compared = 0
        for problem in (ProblemParameters(), ProblemParameters(3, 0.15, 3), ProblemParameters(3, 0.15, 4, "tonnes")):
            for horizon in ("finite", "external"):
                env = MineEnv(mine, problem, rules, EnvironmentParameters(6, horizon))
                queue, seen = deque([env.initial_state()]), set()
                while queue:
                    state = queue.popleft()
                    for action in env.actions(state):
                        moved = env.transition(state, action)
                        destination = moved.state.position if action not in (ADVANCE, STOP) else None
                        if destination is not None and destination not in state.extracted:
                            self.assertIn(destination, env.cone_actions(state))
                            cone = env.extract(state, destination)
                            self.assertEqual((cone.extracted, cone.reward, cone.state.used, cone.state.period,
                                              cone.state.extracted, cone.state.position),
                                             (moved.extracted, moved.reward, moved.state.used, moved.state.period,
                                              moved.state.extracted, moved.state.position))
                            self.assertEqual(cone.state.steps_left, state.steps_left)
                            self.assertEqual(cone.reason, None if moved.reason == "step_limit" else moved.reason)
                            compared += 1
                        if not moved.terminal and moved.state not in seen:
                            seen.add(moved.state)
                            queue.append(moved.state)
        self.assertGreater(compared, 20)

    def test_cone_view_builds_feasible_plans_whose_rewards_are_their_npv(self):
        """Every sequence of cone actions (exhaustive on a small mine) gives a feasible plan
        and the sum of rewards is its verified VAN; calls counts each one."""
        mine = Mine((Block(1, 0, 0, 3, 5, tonnes=2), Block(2, 0, 0, 2, -4, tonnes=3),
                     Block(3, 0, 0, 1, 15, tonnes=1), Block(4, 0, 1, 3, -2, tonnes=1),
                     Block(5, 5, 5, 3, -1, tonnes=1)))
        rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: (), 5: ()})
        for problem in (ProblemParameters(), ProblemParameters(2, 0.1, 2), ProblemParameters(3, 0.15, 4, "tonnes")):
            env = MineEnv(mine, problem, rules, EnvironmentParameters(3, "finite"))
            stack, plans, keys = [(env.initial_state(), (), 0.0)], 0, set()
            while stack:
                state, plan, total = stack.pop()
                keys.add(env.search_key(state))
                for action in env.cone_actions(state):
                    calls = env.calls
                    result = env.cone_step(state, action)
                    self.assertEqual(env.calls, calls + 1)
                    extended = plan + tuple(Extraction(i, result.state.period) for i in result.extracted)
                    evaluation = evaluate_plan(mine, Plan(extended), problem, rules)
                    self.assertTrue(evaluation.feasible, evaluation.errors)
                    self.assertAlmostEqual(total + result.reward, evaluation.npv)
                    if result.terminal:
                        plans += 1
                        self.assertIn(result.reason, (STOP, "all_extracted"))  # never step_limit
                    else:
                        stack.append((result.state, extended, total + result.reward))
            self.assertGreater(plans, 10)
            self.assertTrue(all(len(key) == 3 for key in keys))

    def test_cone_view_rejects_what_it_does_not_offer(self):
        env = layered(ProblemParameters(periods=2, capacity=4, capacity_unit="tonnes"))
        state = env.initial_state()
        self.assertEqual(env.cone_actions(state), (1, 4, 5, ADVANCE, STOP))  # cones of 2 (6 t) and 3 (7 t) exceed 4 t
        for block in (2, 3, 99):
            with self.subTest(block=block), self.assertRaises(ValueError):
                env.extract(state, block)
        first = env.extract(state, 1).state
        with self.assertRaises(ValueError):
            env.extract(first, 1)  # already extracted
        done = env.cone_step(first, STOP).state
        self.assertEqual(env.cone_actions(done), ())
        with self.assertRaises(ValueError):
            env.cone_step(done, ADVANCE)
        moved = replace(first, position=4)
        self.assertEqual(env.search_key(first), env.search_key(moved))  # the cursor is not part of the key

    def test_state_has_no_phase(self):
        self.assertNotIn("phase", State.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
