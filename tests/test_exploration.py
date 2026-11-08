import unittest

from src.config import AgentParameters
from src.envs.mine_env import STOP
from src.models.exploration import SCHEDULES, epsilon_at
from src.models.q_learning import QLearning


class ExplorationTests(unittest.TestCase):
    def test_linear_goes_from_epsilon0_to_zero_over_the_budget(self):
        for transition, expected in ((0, 0.5), (50_000, 0.25), (75_000, 0.125), (100_000, 0.0)):
            with self.subTest(transition=transition):
                self.assertAlmostEqual(epsilon_at("linear", 0.5, transition, 100_000), expected)

    def test_constant_keeps_epsilon0_for_the_whole_budget(self):
        for transition in (0, 50_000, 100_000):
            with self.subTest(transition=transition):
                self.assertAlmostEqual(epsilon_at("constant", 0.5, transition, 100_000), 0.5)

    def test_every_registered_schedule_follows_the_contract(self):
        """Runs on every schedule in SCHEDULES, so a new shape is checked as soon as it is
        registered: callable as (epsilon0, progress), epsilon0 at p=0, always within
        [0, epsilon0] and, if it decreases, 0 at the end of the budget."""
        for name, schedule in SCHEDULES.items():
            with self.subTest(schedule=name):
                self.assertAlmostEqual(schedule(0.5, 0.0), 0.5)
                for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
                    self.assertTrue(0 <= schedule(0.5, progress) <= 0.5)
                if name != "constant":
                    self.assertAlmostEqual(schedule(0.5, 1.0), 0.0)

    def test_unknown_schedule_and_out_of_range_transition_are_rejected(self):
        with self.assertRaises(ValueError):
            AgentParameters(epsilon_schedule="cubic")
        for args in (("cubic", 0.5, 0, 10), ("linear", 0.5, 11, 10), ("linear", 0.5, -1, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                epsilon_at(*args)

    def test_random_exploration_never_chooses_stop(self):
        model = QLearning(AgentParameters(), 0)
        state = object()  # Q is empty: every action ties at 0
        actions = ("north", "down", STOP)
        chosen = {model.select_action(state, actions, epsilon=1.0) for _ in range(500)}
        self.assertEqual(chosen, {"north", "down"})

    def test_stop_is_still_chosen_when_it_is_the_greedy_action_or_the_only_one(self):
        model = QLearning(AgentParameters(), 0)
        state = object()
        model.q[state, STOP] = 10.0
        self.assertEqual({model.select_action(state, ("north", STOP), epsilon=0.0) for _ in range(50)}, {STOP})
        self.assertEqual(model.select_action(state, (STOP,), epsilon=1.0), STOP)


if __name__ == "__main__":
    unittest.main()
