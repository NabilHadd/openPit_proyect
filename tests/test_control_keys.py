from dataclasses import replace
import unittest

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import grid9_z_up
from src.envs.spatial_cone_env import SpatialConeEnv, SpatialState
from src.models.experimental_q_learning import ControlQLearning
from src.models.q_learning import SpatialQLearning


class ControlKeyTests(unittest.TestCase):
    def test_full_key_preserves_every_physical_variable_except_equivalent_phase(self):
        state = SpatialState(frozenset({1}), 0, 2.0, False, 1, 10, "learning")
        exact = ControlQLearning(LearningParameters(), 0, "physical")
        key = exact.state_key(state)
        self.assertEqual(key, exact.state_key(replace(state, phase="construction")))
        for change in ({"extracted": frozenset({1, 2})}, {"period": 1}, {"used": 3},
                       {"position": 2}, {"steps_left": 9}, {"terminal": True}):
            with self.subTest(change=change):
                self.assertNotEqual(key, exact.state_key(replace(state, **change)))

    def test_position_control_is_identical_to_existing_spatial_projection(self):
        parameters = LearningParameters(alpha=1, gamma=0.7, epsilon=0.3)
        control = ControlQLearning(parameters, 8, "position")
        legacy = SpatialQLearning(parameters, 8)
        a = SpatialState(frozenset(), 0, 0, False, None, 20, "learning")
        b = SpatialState(frozenset({1}), 0, 0, False, 1, 19, "learning")
        for _ in range(20):
            self.assertEqual(control.select_action(a, ("north", "stop")), legacy.select_action(a, ("north", "stop")))
            self.assertEqual(control.update(a, "north", 3, b, ("stop",), False),
                             legacy.update(a, "north", 3, b, ("stop",), False))
        self.assertEqual(control.q, legacy.q)
        self.assertEqual(control.rng.getstate(), legacy.rng.getstate())

    def test_construction_queries_trained_exact_key_and_six_direction_masks_match(self):
        mine = Mine((Block(1, 0, 0, 3, 5), Block(2, 0, 0, 2, -4), Block(3, 0, 0, 1, 15)))
        env = SpatialConeEnv(mine, ProblemParameters(), grid9_z_up(mine), SpatialParameters("max-z", 20, True))
        model = ControlQLearning(LearningParameters(alpha=1), 0, "physical")
        before = env.reset(training=True)
        following, reward, done = env.step("start")
        model.update(before, "start", reward, following, env.available_actions(), done)
        training_actions = env.available_actions()
        construction = env.reset(training=False)
        self.assertEqual(model.value(construction, "start"), 5)
        env.step("start")
        self.assertEqual(training_actions, env.available_actions())
        env.step("down")
        self.assertIn("up", env.available_actions())


if __name__ == "__main__":
    unittest.main()
