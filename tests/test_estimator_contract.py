"""Common contract of every model in REGISTRY (src/models/register.py).

It runs on every registered model, so a new one is checked as soon as its module exists
(discover imports every module of src/models). Each model runs with the cheap values of
its params dataclass (SMOKE). What a model does on its own is tested in tests/test_<model>.py.
"""

from dataclasses import asdict, fields
import json
from pathlib import Path
import tempfile
import unittest

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from configs.config_settings import EnvironmentParameters, ProblemParameters
from configs.model_settings import parse_model_settings
from src.data.instance import MineInstance
from configs import PROJECT_ROOT
from src.evaluation.metrics import evaluate_plan
from src.evaluation.tiny_oracle import tiny_mine
from src.models.base import MineEstimator
from src.models.register import discover, load_settings
from src.runner import cost

MODELS = discover()


def smoke(spec, **extra) -> MineEstimator:
    return spec.estimator(**getattr(spec.params, "SMOKE", {}), **extra)


def tiny_instance() -> MineInstance:
    mine, rules = tiny_mine()
    return MineInstance(mine, rules, ProblemParameters(), EnvironmentParameters(12, "finite"))


def instances() -> dict[str, MineInstance]:
    """The simple case and one with 2 periods, capacity, discount and external horizon."""
    mine, rules = tiny_mine()
    return {"one-period-finite": tiny_instance(),
            "two-periods-capacity-external": MineInstance(mine, rules, ProblemParameters(2, 0.1, 3),
                                                          EnvironmentParameters(20, "external"))}


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
                self.assertIs(spec.estimator.Params, spec.params)

    def test_params_dataclass_declares_valid_fixed_and_smoke_and_its_yaml_loads(self):
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                names = {f.name for f in fields(spec.params)}
                self.assertLessEqual(set(getattr(spec.params, "FIXED", ())), names)
                self.assertLessEqual(set(getattr(spec.params, "SMOKE", {})), names)
                spec.params(**getattr(spec.params, "SMOKE", {}))  # valid values
                self.assertEqual(parse_model_settings({}, spec.params, name).params, spec.params())

    def test_clone_and_set_params_follow_scikit_learn(self):
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                estimator = smoke(spec)
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

    def test_every_registered_model_has_a_valid_yaml_in_configs(self):
        """The convention configs/<key>.yml, with the real files of the repository."""
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                path = PROJECT_ROOT / "configs" / f"{name}.yml"
                self.assertTrue(path.is_file(), f"falta {path}")
                self.assertIsInstance(load_settings(name, path).params, spec.params)

    def test_fit_predict_score_diagnostics_and_artifacts(self):
        for (name, spec), (case, instance) in ((m, c) for m in MODELS.items() for c in instances().items()):
            with self.subTest(model=name, instance=case):
                estimator = smoke(spec)
                before = estimator.get_params()
                self.assertIs(estimator.fit(instance), estimator)
                self.assertEqual(estimator.get_params(), before)  # fit never changes hyperparameters
                plan = estimator.predict(instance)
                evaluation = evaluate_plan(instance.mine, plan, instance.problem, instance.precedences)
                self.assertTrue(evaluation.feasible, evaluation.errors)
                self.assertEqual(estimator.score(instance), evaluation.npv)
                json.dumps(estimator.diagnostics(), allow_nan=False)  # must fit in report.json
                # counters_ and trace_ (snapshots of every counter at each improvement, ending at the
                # delivered plan) follow the contract that the runner checks
                spent = cost(estimator, estimator.score(instance))
                self.assertEqual(spent["unit"], estimator.cost_unit)
                self.assertEqual(spent["spent"], estimator.counters_)
                self.assertEqual(set(spent["to_best"]), set(estimator.counters_))
                json.dumps(spent, allow_nan=False)
                with tempfile.TemporaryDirectory() as directory:
                    estimator.save_artifacts(Path(directory))

    def test_compare_reference_returns_counts_and_changes_nothing(self):
        instance = tiny_instance()
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                estimator = smoke(spec).fit(instance)
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
                plans = [smoke(spec, random_state=3).fit(instance).predict(instance)
                         for _ in range(2)]
                self.assertEqual(plans[0], plans[1])


if __name__ == "__main__":
    unittest.main()
