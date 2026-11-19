"""The only module that knows every model: its name in configs/config.yml (models, model)
-> estimator class, hyperparameter dataclass and loader of its YAML.

A new model is registered here; tests/test_estimator_contract.py then checks it.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass

from configs.config_settings import RunConfig
from configs.q_learning_settings import QLearningParams, load_q_learning_settings
from src.models.base import MineEstimator
from src.models.q_learning import QLearningEstimator


@dataclass(frozen=True)
class ModelSpec:
    estimator: type[MineEstimator]
    params: type      # frozen dataclass of the hyperparameters: same field names as the estimator's __init__
    load: Callable    # path of the model YAML -> settings with .params and .distributions


MODELS: dict[str, ModelSpec] = {
    "q_learning": ModelSpec(QLearningEstimator, QLearningParams, load_q_learning_settings),
}


def load_models(config: RunConfig) -> dict:
    """Settings of every model listed in config.models, read before anything is trained so
    a broken or unknown model fails first."""
    unknown = sorted(set(config.models) - set(MODELS))
    if unknown:
        raise ValueError(f"Modelos sin implementación en models: {unknown}; disponibles: {sorted(MODELS)}")
    return {name: MODELS[name].load(path) for name, path in config.models.items()}


def build_estimator(name: str, settings) -> MineEstimator:
    """Unfitted estimator with the params of its YAML (random_state keeps its default)."""
    return MODELS[name].estimator(**asdict(settings.params))
