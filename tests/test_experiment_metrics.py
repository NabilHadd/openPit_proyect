from dataclasses import replace
import unittest

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import grid9_z_up
from src.envs.spatial_cone_env import SpatialConeEnv, SpatialState
from src.evaluation.experiment_metrics import ControlDiagnostics
from src.models.experimental_q_learning import ControlQLearning
from src.training.trainer import train


class ExperimentMetricTests(unittest.TestCase):
    def test_hand_counted_reuse_singletons_and_navigation_returns(self):
        model = ControlQLearning(LearningParameters(), 0, "position")
        diagnostics = ControlDiagnostics(model.state_key)
        trace = diagnostics.new_episode("learning", 0)
        a = SpatialState(frozenset({1, 2}), 0, 0, False, 1, 10, "learning")
        b = replace(a, position=2, steps_left=9)
        a2 = replace(a, steps_left=8)
        b2 = replace(b, steps_left=7)
        for previous, action, following in ((a, "east", b), (b, "west", a2), (a2, "east", b2)):
            trace.observe(previous, action, 0, following)
        trace.finish("manual_partial_trace", 3)
        payload = diagnostics.payload(q_entries=2)
        coverage = payload["state_coverage"]
        self.assertEqual(coverage["learning_q_state_visits"], 3)
        self.assertEqual(coverage["unique_learning_q_states"], 2)
        self.assertAlmostEqual(coverage["q_state_reuse_fraction"], 1 / 3)
        self.assertEqual(coverage["q_states_visited_once_fraction"], 0.5)
        self.assertAlmostEqual(coverage["state_action_reuse_fraction"], 1 / 3)
        self.assertEqual(coverage["state_action_pairs_visited_once_fraction"], 0.5)
        self.assertEqual(coverage["q_rows_with_distinct_physical_states"], 1)
        self.assertEqual(coverage["q_rows_with_distinct_mining_contexts"], 0)
        self.assertEqual(payload["learning"]["zero_extraction_moves"], 3)
        self.assertEqual(payload["learning"]["navigation_cycle_returns"], 2)

    def test_zero_value_extraction_is_progress_and_stop_is_not_a_navigation_cycle(self):
        model = ControlQLearning(LearningParameters(), 0, "physical")
        diagnostics = ControlDiagnostics(model.state_key)
        trace = diagnostics.new_episode("learning", 0)
        empty = SpatialState(frozenset(), 0, 0, False, None, 3, "learning")
        mined = SpatialState(frozenset({1}), 0, 0, False, 1, 2, "learning")
        stopped = replace(mined, terminal=True, steps_left=1)
        trace.observe(empty, "start", 0, mined)
        trace.observe(mined, "stop", 0, stopped)
        trace.finish("stop", 2)
        row = diagnostics.payload(2)["episodes"][0]
        self.assertEqual(row["extracted_blocks"], 1)
        self.assertEqual(row["extraction_transitions"], 1)
        self.assertEqual(row["no_extraction_transitions"], 1)
        self.assertEqual(row["zero_extraction_moves"], 0)
        self.assertEqual(row["navigation_cycle_returns"], 0)

    def test_known_zero_is_not_an_unknown_state_and_action_coverage_is_separate(self):
        model = ControlQLearning(LearningParameters(), 0, "physical")
        diagnostics = ControlDiagnostics(model.state_key)
        state = SpatialState(frozenset({1}), 0, 0, False, 1, 10, "learning")
        following = replace(state, position=2, steps_left=9)
        learned = diagnostics.new_episode("learning", 0)
        model.update(state, "east", 0, following, ("stop",), False)
        learned.observe(state, "east", 0, following)
        greedy = diagnostics.new_episode("construction", None)
        same = replace(state, phase="construction")
        greedy.observe(same, "east", 0, following)
        self.assertEqual(greedy.stats.unseen_learning_q_state_visits, 0)
        self.assertEqual(greedy.stats.unseen_learning_state_action_visits, 0)
        greedy.observe(same, "west", 0, following)
        self.assertEqual(greedy.stats.unseen_learning_q_state_visits, 0)
        self.assertEqual(greedy.stats.unseen_learning_state_action_visits, 1)
        greedy.observe(replace(same, position=99), "stop", 0, following)
        self.assertEqual(greedy.stats.unseen_learning_q_state_visits, 1)

    def test_distinct_mining_histories_are_visible_under_compact_encoding(self):
        model = ControlQLearning(LearningParameters(), 0, "position")
        diagnostics = ControlDiagnostics(model.state_key)
        trace = diagnostics.new_episode("learning", 0)
        a = SpatialState(frozenset({1}), 0, 0, False, 1, 10, "learning")
        b = replace(a, extracted=frozenset({1, 2}), steps_left=9)
        trace.observe(a, "down", 0, b)
        trace.observe(b, "stop", 0, replace(b, terminal=True))
        self.assertEqual(diagnostics.payload(2)["state_coverage"]["q_rows_with_distinct_mining_contexts"], 1)

    def test_empty_denominators_are_null(self):
        model = ControlQLearning(LearningParameters(), 0, "physical")
        payload = ControlDiagnostics(model.state_key).payload(0)
        self.assertIsNone(payload["state_coverage"]["q_state_reuse_fraction"])
        self.assertIsNone(payload["learning"]["no_extraction_fraction"])

    def test_instrumentation_does_not_change_training_results_q_or_rng(self):
        mine = Mine((Block(1, 0, 0, 3, 5), Block(2, 0, 0, 2, -4),
                     Block(3, 0, 0, 1, 15), Block(4, 1, 0, 3, -1)))
        def execute(instrument):
            env = SpatialConeEnv(mine, ProblemParameters(), grid9_z_up(mine), SpatialParameters("max-z", 20, True))
            model = ControlQLearning(LearningParameters(alpha=1, gamma=1, epsilon=0.3), 8, "physical")
            diagnostics = ControlDiagnostics(model.state_key) if instrument else None
            result = train(env, model, 10, diagnostics=diagnostics)
            if diagnostics is not None:
                payload = diagnostics.payload(len(model.q))
                self.assertEqual(payload["learning"]["transitions"], sum(row.steps for row in result.history))
                self.assertEqual(payload["greedy"]["transitions"], result.greedy_steps)
                self.assertEqual(payload["state_coverage"]["q_rows_with_distinct_physical_states"], 0)
            return result, model.q, model.rng.getstate()
        self.assertEqual(execute(True), execute(False))


if __name__ == "__main__":
    unittest.main()
