"""Common contract of every model in MODELS (src/models/registry.py).

It runs on every registered model, so a new one is checked as soon as it is registered.
What a model does on its own is tested in tests/test_<model>.py.
"""

from dataclasses import asdict, fields
import json
from pathlib import Path
import tempfile
import unittest

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.instance import MineInstance
from src.evaluation.metrics import evaluate_plan
from src.evaluation.tiny_oracle import tiny_mine
from src.models.base import MineEstimator
from src.models.registry import MODELS


# Cheap hyperparameters for the contract; a model without an entry runs with its defaults.
FAST = {"q_learning": {"transitions": 500, "policy_rollouts": 2}}


def tiny_instance() -> MineInstance:
    mine, rules = tiny_mine()
    return MineInstance(mine, rules, ProblemParameters(), EnvironmentParameters(12, "finite"))


class EstimatorContractTests(unittest.TestCase):
    def test_hyperparameters_are_the_fields_of_the_settings_dataclass_plus_random_state(self):
        """One name everywhere: YAML key = dataclass field = __init__ argument, same defaults."""
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                self.assertTrue(issubclass(spec.estimator, MineEstimator))
                params = spec.estimator().get_params()
                self.assertIn("random_state", params)
                del params["random_state"]
                self.assertEqual(set(params), {f.name for f in fields(spec.params)})
                self.assertEqual(params, asdict(spec.params()))

    def test_clone_and_set_params_follow_scikit_learn(self):
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                estimator = spec.estimator(**FAST.get(name, {}))
                copy = clone(estimator)
                self.assertIsNot(copy, estimator)
                self.assertEqual(copy.get_params(), estimator.get_params())
                self.assertIs(copy.set_params(random_state=7), copy)
                self.assertEqual(copy.get_params()["random_state"], 7)

    def test_results_before_fit_are_an_error(self):
        instance = tiny_instance()
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                estimator = spec.estimator()
                with tempfile.TemporaryDirectory() as directory:
                    for call in (lambda: estimator.predict(instance), estimator.diagnostics,
                                 lambda: estimator.save_artifacts(Path(directory)),
                                 lambda: estimator.compare_reference(0.0)):
                        with self.assertRaises(NotFittedError):
                            call()

    def test_fit_predict_score_diagnostics_and_artifacts(self):
        instance = tiny_instance()
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                estimator = spec.estimator(**FAST.get(name, {}))
                before = estimator.get_params()
                self.assertIs(estimator.fit(instance), estimator)
                self.assertEqual(estimator.get_params(), before)  # fit never changes hyperparameters
                plan = estimator.predict(instance)
                evaluation = evaluate_plan(instance.mine, plan, instance.problem, instance.precedences)
                self.assertTrue(evaluation.feasible, evaluation.errors)
                self.assertEqual(estimator.score(instance), evaluation.npv)
                json.dumps(estimator.diagnostics(), allow_nan=False)  # must fit in report.json
                with tempfile.TemporaryDirectory() as directory:
                    estimator.save_artifacts(Path(directory))

    def test_compare_reference_returns_counts_and_changes_nothing(self):
        instance = tiny_instance()
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                estimator = spec.estimator(**FAST.get(name, {})).fit(instance)
                before = estimator.predict(instance), estimator.diagnostics()
                for reference in (estimator.score(instance), estimator.score(instance) + 1):
                    counts = estimator.compare_reference(reference)
                    self.assertIsInstance(counts, dict)
                    for key, value in counts.items():
                        self.assertIsInstance(key, str)
                        self.assertTrue(isinstance(value, int) and not isinstance(value, bool) and value >= 0, counts)
                self.assertEqual((estimator.predict(instance), estimator.diagnostics()), before)

    def test_same_random_state_gives_the_same_plan(self):
        instance = tiny_instance()
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                plans = [spec.estimator(**FAST.get(name, {}), random_state=3).fit(instance).predict(instance)
                         for _ in range(2)]
                self.assertEqual(plans[0], plans[1])


if __name__ == "__main__":
    unittest.main()
