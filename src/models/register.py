"""Registry of models: key in configs/config.yml (models, model) -> estimator class and
params dataclass. A model registers itself in its own module:

    @dataclass(frozen=True)
    class DynaQParams(TabularParams):
        planning_steps: int = 5
        SMOKE: ClassVar[dict] = {"transitions": 500, "policy_rollouts": 2}

    @register("dyna_q", params=DynaQParams)
    class DynaQEstimator(TabularQ):
        ...

Adding a model is writing src/models/<key>.py and configs/<key>.yml; nothing else is
edited (config.yml only names it in train or experiments). One module per model key: the
modes import only src/models/<key>.py of the models they run. discover() imports every
module, so tests/test_estimator_contract.py checks a model as soon as its file exists.
"""

from dataclasses import asdict, dataclass, is_dataclass
import importlib
import pkgutil

from configs.config_settings import RunConfig
from configs.model_settings import ModelSettings, load_model_settings
from src.models.base import MineEstimator


@dataclass(frozen=True)
class ModelSpec:
    estimator: type[MineEstimator]
    params: type      # frozen dataclass of the hyperparameters: same field names as the estimator's __init__


REGISTRY: dict[str, ModelSpec] = {}


def register(key: str, *, params: type):
    """Class decorator: registers the estimator under key and sets estimator.Params."""
    if not isinstance(key, str) or not key:
        raise ValueError("La clave de un modelo debe ser un texto no vacío")
    if not is_dataclass(params):
        raise TypeError(f"params de {key} debe ser una dataclass de hiperparámetros")

    def decorate(cls):
        if not (isinstance(cls, type) and issubclass(cls, MineEstimator)):
            raise TypeError(f"{key}: solo se registran subclases de MineEstimator")
        previous = REGISTRY.get(key)
        if previous is not None and (previous.estimator.__module__, previous.estimator.__qualname__) != (
                cls.__module__, cls.__qualname__):
            raise ValueError(f"Modelo registrado dos veces: {key} ({previous.estimator.__qualname__} "
                             f"y {cls.__qualname__})")
        cls.Params = params
        REGISTRY[key] = ModelSpec(cls, params)
        return cls
    return decorate


def discover() -> dict[str, ModelSpec]:
    """Import every module of src/models so each one registers its models (tests use it to
    check every model). The modes import only the models they run (spec)."""
    package = importlib.import_module("src.models")
    for module in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        importlib.import_module(f"src.models.{module.name}")
    return REGISTRY


def spec(key: str) -> ModelSpec:
    """The registered model key. Convention: it lives in src/models/<key>.py, which is the
    only module imported, so a broken module of another model never stops this one."""
    if key in REGISTRY:
        return REGISTRY[key]
    if not isinstance(key, str) or not key.isidentifier():
        raise ValueError(f"Clave de modelo inválida: {key!r}")
    module = f"src.models.{key}"
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise ValueError(f"src/models/{key}.py no se pudo importar: {exc}") from exc
        raise ValueError(f"Modelo sin implementación: {key} (no existe src/models/{key}.py)") from None
    except Exception as exc:
        raise ValueError(f"src/models/{key}.py no se pudo importar: {type(exc).__name__}: {exc}") from exc
    if key not in REGISTRY:
        raise ValueError(f"Modelo sin implementación: src/models/{key}.py no registra @register({key!r}, ...)")
    return REGISTRY[key]


def load_settings(key: str, path) -> ModelSettings:
    """YAML of a registered model -> its params (validated by the dataclass) and distributions."""
    return load_model_settings(path, spec(key).params, key)


def load_models(config: RunConfig, keys) -> dict[str, ModelSettings]:
    """Settings of every model in keys, each from config.model_file(key) (configs/<key>.yml),
    read and validated before anything is trained so a broken or unknown model fails first."""
    settings = {}
    for key in keys:
        spec(key)
        path = config.model_file(key)
        if not path.is_file():
            raise ValueError(f"No existe el YAML del modelo {key}: {path} (convención: configs/<clave>.yml)")
        settings[key] = load_settings(key, path)
    return settings


def build_estimator(key: str, settings: ModelSettings) -> MineEstimator:
    """Unfitted estimator with the params of its YAML (random_state keeps its default)."""
    return spec(key).estimator(**asdict(settings.params))
