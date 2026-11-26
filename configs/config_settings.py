"""Configuration shared by every model (configs/config.yml).

Everything that defines the task lives here: data, reference solution, output, seeds,
the mining problem and how MineEnv presents it, and which models the modes run (train,
experiments). How a model learns lives in its own YAML, found by convention:
configs/<key>.yml, next to this file.
"""

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import re

from configs import PROJECT_ROOT, check_keys, integer, project_path, read_yaml, section


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


def _check_key(key, where: str):
    """A model key is also the name of its module (src/models/<key>.py) and of its YAML."""
    if not isinstance(key, str) or not key.isidentifier():
        raise ValueError(f"{where} debe ser la clave de un modelo (nombre de módulo, p. ej. q_learning): {key!r}")


@dataclass(frozen=True)
class RunConfig:
    """train: the model of the train mode; experiments: the models of the experiments mode.
    The YAML of model <key> is <key>.yml in the folder of this configuration (configs/).
    name: optional label added at the end of the output folder names."""
    data: str
    train: str | None = None
    experiments: tuple[str, ...] = ()
    name: str | None = None
    solution: str | None = None
    output: str = "artifacts"
    seeds: tuple[int, ...] = (0,)
    problem: ProblemParameters = field(default_factory=ProblemParameters)
    environment: EnvironmentParameters = field(default_factory=EnvironmentParameters)
    tuning: TuningParameters = field(default_factory=TuningParameters)
    source: str | None = None  # path of the configuration file it was read from

    def __post_init__(self):
        _check_seeds(self.seeds, "seeds")
        if self.train is None and not self.experiments:
            raise ValueError("La configuración debe indicar train (un modelo) o experiments (lista de modelos)")
        if self.train is not None:
            _check_key(self.train, "train")
        for key in self.experiments:
            _check_key(key, "cada elemento de experiments")
        if len(set(self.experiments)) != len(self.experiments):
            raise ValueError("experiments no puede repetir modelos")
        if self.name is not None and (not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.name)):
            raise ValueError("name solo admite letras, dígitos, _ y - (va en el nombre de la carpeta)")
        shared = sorted(set(self.seeds) & set(self.tuning.seeds))
        if shared:
            raise ValueError(f"tuning.seeds no puede repetir semillas de seeds: {shared}")

    def model_file(self, key: str) -> Path:
        """By convention, configs/<key>.yml: next to the configuration file."""
        folder = Path(self.source).parent if self.source is not None else PROJECT_ROOT / "configs"
        return folder / f"{key}.yml"

    def as_dict(self) -> dict:
        return asdict(self)


TOP_LEVEL = {"data", "solution", "output", "seeds", "problem", "environment", "train", "experiments", "name", "tuning"}


def _seed_list(values, name: str) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} debe ser una lista, por ejemplo [0, 1, 2]")
    return tuple(values)


def parse_config(raw, source: str | None = None) -> RunConfig:
    check_keys(raw, TOP_LEVEL, "la configuración")
    if not raw.get("data"):
        raise ValueError("La configuración debe indicar data")
    experiments = raw.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("experiments debe ser una lista de modelos, por ejemplo [q_learning]")
    tuning = check_keys(raw.get("tuning") or {}, {f.name for f in fields(TuningParameters)}, "tuning")
    return RunConfig(data=str(raw["data"]), train=raw.get("train"), experiments=tuple(experiments),
                     name=raw.get("name"),
                     solution=None if raw.get("solution") is None else str(raw["solution"]),
                     output=str(raw.get("output", "artifacts")),
                     seeds=_seed_list(raw.get("seeds", [0]), "seeds"),
                     problem=section(ProblemParameters, raw.get("problem"), "problem"),
                     environment=section(EnvironmentParameters, raw.get("environment"), "environment"),
                     tuning=TuningParameters(**{**tuning, "seeds": _seed_list(tuning.get("seeds", []), "tuning.seeds")}),
                     source=source)


def load_config(path) -> RunConfig:
    return parse_config(read_yaml(path), str(project_path(path)))
