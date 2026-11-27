"""Registry of models (src/models/register.py): discovery, settings and errors."""

from dataclasses import dataclass
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from src.models.base import MineEstimator
from src.models.q_learning import QLearningEstimator, QLearningParams
from src.models.register import REGISTRY, build_estimator, discover, load_settings, register, spec
from src.models.tabular import TabularQ


@dataclass(frozen=True)
class Nothing:
    pass


class RegisterTests(unittest.TestCase):
    def test_discover_finds_q_learning_with_its_params(self):
        models = discover()
        self.assertIs(models["q_learning"].estimator, QLearningEstimator)
        self.assertIs(models["q_learning"].params, QLearningParams)
        self.assertNotIn("tabular", models)  # the shared base is not a model

    def test_settings_and_estimator_come_from_the_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.yml"
            path.write_text(yaml.safe_dump({"params": {"alpha": 0.5, "transitions": 300}}))
            settings = load_settings("q_learning", path)
            estimator = build_estimator("q_learning", settings)
        self.assertIsInstance(estimator, QLearningEstimator)
        self.assertEqual((estimator.alpha, estimator.transitions, estimator.random_state), (0.5, 300, 0))

    def test_unknown_duplicate_and_invalid_registrations_are_errors(self):
        with self.assertRaisesRegex(ValueError, "sin implementación"):
            load_settings("ppo", "configs/q_learning.yml")
        with self.assertRaisesRegex(ValueError, "dos veces"):
            register("q_learning", params=QLearningParams)(type("Other", (QLearningEstimator,), {}))
        with self.assertRaises(TypeError):
            register("plain", params=Nothing)(object)
        with self.assertRaises(TypeError):
            register("no_dataclass", params=dict)
        self.assertNotIn("plain", REGISTRY)
        self.assertTrue(issubclass(REGISTRY["q_learning"].estimator, MineEstimator))

    def test_only_the_module_of_the_requested_model_is_imported(self):
        """A broken module of another model never stops this one (src/models/<key>.py only)."""
        real = importlib.import_module
        imported = []

        def fake(name, *args):
            imported.append(name)
            if name == "src.models.broken_model":
                raise SyntaxError("módulo roto")
            return real(name, *args)

        with mock.patch("src.models.register.importlib.import_module", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "src/models/broken_model.py no se pudo importar: SyntaxError"):
                spec("broken_model")
            self.assertIs(spec("q_learning").estimator, QLearningEstimator)
        self.assertEqual(imported, ["src.models.broken_model"])  # q_learning was already registered
        with self.assertRaisesRegex(ValueError, "no existe src/models/ppo.py"):
            spec("ppo")
        with self.assertRaisesRegex(ValueError, "no registra"):
            spec("exploration")  # a module of src/models that registers nothing
        with self.assertRaisesRegex(ValueError, "inválida"):
            spec("../x")

    def test_the_tabular_base_is_not_a_model(self):
        from src.evaluation.tiny_oracle import tiny_mine
        from src.data.instance import MineInstance
        mine, rules = tiny_mine()
        with self.assertRaisesRegex(TypeError, "no es un modelo registrado"):
            TabularQ().fit(MineInstance(mine, rules))


if __name__ == "__main__":
    unittest.main()
