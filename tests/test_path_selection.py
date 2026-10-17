import unittest

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import Precedences, grid9_z_up
from src.envs.mine_env import STOP, MineEnv
from src.envs.spatial_cone_env import START, SpatialConeEnv
from src.models.base import BaseModel
from src.models.q_learning import SpatialQLearning
from src.training.trainer import PathObserver, run_episode, train


class ScriptedModel(BaseModel):
    def __init__(self, script):
        self.script = script
        self.index = 0
        self.updates = []

    def select_action(self, state, actions, *, explore=True):
        if not state.extracted:
            self.index = 0
        action = self.script[self.index]
        self.index += 1
        if action not in actions:
            raise AssertionError(f"Acción de prueba no disponible: {action}")
        return action

    def update(self, state, action, reward, next_state, next_actions, terminal):
        self.updates.append((action, reward, terminal))
        return 0.0


def chain(values, *, periods=1, rate=0.0):
    mine = Mine(tuple(Block(i + 1, i, 0, 0, value) for i, value in enumerate(values)))
    rules = Precedences({i + 1: frozenset({i}) if i else frozenset() for i in range(len(values))},
                        "synthetic-chain", "hand-calculated")
    return MineEnv(mine, ProblemParameters(periods=periods, discount_rate=rate), rules)


class PathSelectionTests(unittest.TestCase):
    def test_keeps_profitable_prefix_without_rewriting_completed_episode(self):
        model = ScriptedModel((1, 2))
        result = train(chain((10, -15)), model, 1)
        self.assertEqual(result.history[0].reward_sum, -5)
        self.assertEqual(result.best_evaluation.npv, -5)
        self.assertEqual(result.greedy_evaluation.npv, -5)
        self.assertEqual(result.selected_path.evaluation.npv, 10)
        self.assertEqual(result.selected_path.steps, 1)
        self.assertEqual(result.selected_path.actions, (1,))
        self.assertEqual(tuple(e.block_id for e in result.selected_path.plan.extractions), (1,))
        self.assertEqual(model.updates, [(1, 10, False), (2, -15, True)])

    def test_negative_intermediate_reward_does_not_end_or_prune_path(self):
        model = ScriptedModel((1, 2, 3, 4))
        result = train(chain((5, -10, 30, -50)), model, 1)
        self.assertEqual([u[1] for u in model.updates], [5, -10, 30, -50])
        self.assertEqual(result.history[0].steps, 4)
        self.assertEqual(result.best_evaluation.npv, -25)
        self.assertEqual(result.selected_path.evaluation.npv, 25)
        self.assertEqual(result.selected_path.actions, (1, 2, 3))

    def test_ties_keep_earliest_step_episode_and_phase(self):
        result = train(chain((5, 0, -1)), ScriptedModel((1, 2, 3)), 2)
        self.assertEqual(result.selected_path.evaluation.npv, 5)
        self.assertEqual(result.selected_path.steps, 1)
        self.assertEqual(result.selected_path.episode, 0)
        self.assertEqual(result.selected_path.phase, "learning")

    def test_all_negative_paths_do_not_inject_an_unobserved_empty_path(self):
        result = train(chain((-4, -5)), ScriptedModel((1, 2)), 1)
        self.assertEqual(result.best_evaluation.npv, -9)
        self.assertEqual(result.selected_path.evaluation.npv, -4)
        self.assertEqual(result.selected_path.actions, (1,))

    def test_greedy_prefix_can_win_without_becoming_a_training_episode(self):
        class DifferentPhases(ScriptedModel):
            def select_action(self, state, actions, *, explore=True):
                if not state.extracted:
                    return STOP if explore else 1
                return 2
        result = train(chain((7, -10)), DifferentPhases(()), 1)
        self.assertEqual(result.best_evaluation.npv, 0)
        self.assertEqual(result.greedy_evaluation.npv, -3)
        self.assertEqual(result.selected_path.evaluation.npv, 7)
        self.assertEqual(result.selected_path.phase, "construction")
        self.assertIsNone(result.selected_path.episode)

    def test_selected_path_uses_discounted_economic_value(self):
        from src.envs.mine_env import ADVANCE
        result = train(chain((-4, 15), periods=2, rate=0.5), ScriptedModel((1, ADVANCE, 2)), 1)
        self.assertEqual(result.selected_path.evaluation.npv, 6)
        self.assertEqual(result.selected_path.evaluation.raw_value, 11)

    def test_zero_step_terminal_has_a_verified_empty_path(self):
        mine = Mine((Block(1, 0, 0, 0, 9, tonnes=2),))
        rules = Precedences({1: frozenset()}, "synthetic", "test")
        env = MineEnv(mine, ProblemParameters(capacity=1, capacity_unit="tonnes"), rules)
        result = train(env, ScriptedModel(()), 1)
        self.assertEqual(result.selected_path.steps, 0)
        self.assertEqual(result.selected_path.actions, ())
        self.assertTrue(result.selected_path.evaluation.feasible)

    def test_observer_does_not_change_learning_or_rng(self):
        mine = Mine((Block(1, 0, 0, 3, 5), Block(2, 0, 0, 2, -4),
                     Block(3, 0, 0, 1, 15), Block(4, 0, 1, 3, -2)))
        parameters = LearningParameters(alpha=1, gamma=0.7, epsilon=0.3)
        def execute(observe):
            env = SpatialConeEnv(mine, ProblemParameters(), grid9_z_up(mine), SpatialParameters("max-z", 30))
            model = SpatialQLearning(parameters, 8)
            results = []
            for episode in range(20):
                observer = PathObserver(env, episode, "learning")
                results.append(run_episode(env, model, learn=True,
                                           on_transition=observer.observe if observe else None))
            return model.q, model.rng.getstate(), results
        self.assertEqual(execute(True), execute(False))

    def test_spatial_revisit_is_zero_and_prefix_keeps_cone_costs(self):
        mine = Mine((Block(1, 0, 0, 3, 5), Block(2, 0, 0, 2, -4),
                     Block(3, 0, 0, 1, 15), Block(4, 4, 4, 3, -1)))
        env = SpatialConeEnv(mine, ProblemParameters(), grid9_z_up(mine), SpatialParameters("max-z", 10))
        observer = PathObserver(env, 0, "learning")
        model = ScriptedModel((START, "down", "down", "up", STOP))
        total, steps = run_episode(env, model, learn=True, on_transition=observer.observe)
        self.assertEqual(total, 16)
        self.assertEqual(steps, 5)
        self.assertEqual([u[1] for u in model.updates], [5, -4, 15, 0, 0])
        self.assertEqual(observer.best.steps, 3)
        self.assertEqual(observer.best.evaluation.npv, 16)
        self.assertEqual(tuple(e.block_id for e in observer.best.plan.extractions), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
