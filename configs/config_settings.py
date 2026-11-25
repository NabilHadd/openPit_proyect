"""Configuration shared by every model (configs/config.yml).

Everything that defines the task lives here: data, reference solution, output, seeds,
the mining problem and how MineEnv presents it. How a model learns lives in the model's
own YAML, listed in `models`.
"""

from dataclasses import asdict, dataclass, field, fields
import math

from configs import check_keys, integer, read_yaml, section


DEFAULT_CONFIG = "configs/config.yml"


@dataclass(frozen=True)
class ProblemParameters:
    """The mining problem: periods, economic discount and capacity per period."""
    periods: int = 1
    discount_rate: float = 0.0
    capacity: float | None = None
    capacity_unit: str = "blocks"

    def __post_init__(self):
        integer(self.periods, "periods", 1)
        if not math.isfinite(self.discount_rate) or self.discount_rate < 0:
            raise ValueError("discount_rate debe ser finita y no negativa")
        if self.capacity_unit not in {"blocks", "tonnes"}:
            raise ValueError("capacity_unit debe ser blocks o tonnes")
        if self.capacity is not None and (not math.isfinite(self.capacity) or self.capacity <= 0):
            raise ValueError("capacity debe ser finita y positiva")


@dataclass(frozen=True)
class EnvironmentParameters:
    """How MineEnv presents the problem to the models that walk through it.

    horizon=finite: max_steps is a rule of the task and the steps left are part of the state.
    horizon=external: max_steps only interrupts an episode; the agent never sees a clock.
    """
    max_steps: int = 500
    horizon: str = "external"

    def __post_init__(self):
        integer(self.max_steps, "max_steps", 1)
        if self.horizon not in {"finite", "external"}:
            raise ValueError("horizon debe ser finite o external")


def _check_seeds(seeds, name: str, *, allow_empty: bool = False):
    if not seeds and not allow_empty:
        raise ValueError(f"{name} debe ser una lista no vacía")
    for seed in seeds:
        integer(seed, f"cada semilla de {name}", 0)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Las semillas de {name} deben ser distintas")


@dataclass(frozen=True)
class TuningParameters:
    """Random search of the tuning mode, shared by every model so each one is tuned with the
    same budget. The distributions themselves live in each model's YAML.

    n_iter: configurations drawn from the model's distributions.
    search_seed: seed of that draw, so every run tries the same configurations.
    seeds: training seeds of every configuration. They never repeat the run seeds: tuning
        and comparing with the same seeds would pick hyperparameters fitted to those seeds.
    """
    n_iter: int = 20
    search_seed: int = 0
    seeds: tuple[int, ...] = ()

    def __post_init__(self):
        integer(self.n_iter, "tuning.n_iter", 1)
        integer(self.search_seed, "tuning.search_seed", 0)
        _check_seeds(self.seeds, "tuning.seeds", allow_empty=True)


@dataclass(frozen=True)
class RunConfig:
    data: str
    model: str                # model to run: a key of models
    models: dict[str, str]    # model name -> path of its YAML
    solution: str | None = None
    output: str = "artifacts/reports"
    seeds: tuple[int, ...] = (0,)
    problem: ProblemParameters = field(default_factory=ProblemParameters)
    environment: EnvironmentParameters = field(default_factory=EnvironmentParameters)
    tuning: TuningParameters = field(default_factory=TuningParameters)

    def __post_init__(self):
        _check_seeds(self.seeds, "seeds")
        if not isinstance(self.models, dict) or not self.models or not all(
                isinstance(name, str) and name and isinstance(path, str) and path
                for name, path in self.models.items()):
            raise ValueError("models debe asociar cada nombre de modelo a la ruta de su YAML")
        if not isinstance(self.model, str) or self.model not in self.models:
            raise ValueError(f"model debe ser una clave de models: {sorted(self.models)}")
        shared = sorted(set(self.seeds) & set(self.tuning.seeds))
        if shared:
            raise ValueError(f"tuning.seeds no puede repetir semillas de seeds: {shared}")

    def as_dict(self) -> dict:
        return asdict(self)


TOP_LEVEL = {"data", "solution", "output", "seeds", "problem", "environment", "model", "models", "tuning"}


def _seed_list(values, name: str) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} debe ser una lista, por ejemplo [0, 1, 2]")
    return tuple(values)


def parse_config(raw) -> RunConfig:
    check_keys(raw, TOP_LEVEL, "la configuración")
    for key in ("data", "model", "models"):
        if not raw.get(key):
            raise ValueError(f"La configuración debe indicar {key}")
    tuning = check_keys(raw.get("tuning") or {}, {f.name for f in fields(TuningParameters)}, "tuning")
    return RunConfig(data=str(raw["data"]), model=raw["model"], models=raw["models"],
                     solution=None if raw.get("solution") is None else str(raw["solution"]),
                     output=str(raw.get("output", "artifacts/reports")),
                     seeds=_seed_list(raw.get("seeds", [0]), "seeds"),
                     problem=section(ProblemParameters, raw.get("problem"), "problem"),
                     environment=section(EnvironmentParameters, raw.get("environment"), "environment"),
                     tuning=TuningParameters(**{**tuning, "seeds": _seed_list(tuning.get("seeds", []), "tuning.seeds")}))


def load_config(path) -> RunConfig:
    return parse_config(read_yaml(path))
