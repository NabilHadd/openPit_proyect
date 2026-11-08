"""Run configuration: one YAML file. Paths are relative to the repository, never to the caller."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import yaml

from src.models.exploration import SCHEDULES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/config.yml"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _integer(value, name: str, minimum: int):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}")


@dataclass(frozen=True)
class ProblemParameters:
    """The mining problem: periods, economic discount and capacity per period."""
    periods: int = 1
    discount_rate: float = 0.0
    capacity: float | None = None
    capacity_unit: str = "blocks"

    def __post_init__(self):
        _integer(self.periods, "periods", 1)
        if not math.isfinite(self.discount_rate) or self.discount_rate < 0:
            raise ValueError("discount_rate debe ser finita y no negativa")
        if self.capacity_unit not in {"blocks", "tonnes"}:
            raise ValueError("capacity_unit debe ser blocks o tonnes")
        if self.capacity is not None and (not math.isfinite(self.capacity) or self.capacity <= 0):
            raise ValueError("capacity debe ser finita y positiva")


@dataclass(frozen=True)
class AgentParameters:
    """Q-learning hyperparameters and what the Q table remembers of a state."""
    state: str = "physical"
    alpha: float = 1.0
    gamma: float = 1.0
    epsilon: float = 0.3              # initial epsilon of the schedule (fixed epsilon with "constant")
    epsilon_schedule: str = "constant"   # shape of the decrease, see src/models/exploration.py

    def __post_init__(self):
        if self.state not in {"physical", "position"}:
            raise ValueError("state debe ser physical o position")
        if self.epsilon_schedule not in SCHEDULES:
            raise ValueError(f"epsilon_schedule debe ser uno de {sorted(SCHEDULES)}")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha debe estar en (0, 1]")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("gamma y epsilon deben estar en [0, 1]")


@dataclass(frozen=True)
class TrainingParameters:
    """transitions = total training budget; max_steps = limit of one episode.

    horizon=finite: max_steps is a rule of the task and the steps left are part of the state.
    horizon=external: max_steps only interrupts an episode; the agent never sees a clock.
    """
    transitions: int = 100_000
    max_steps: int = 500
    horizon: str = "external"

    def __post_init__(self):
        _integer(self.transitions, "transitions", 1)
        _integer(self.max_steps, "max_steps", 1)
        if self.horizon not in {"finite", "external"}:
            raise ValueError("horizon debe ser finite o external")


@dataclass(frozen=True)
class EvaluationParameters:
    policy_rollouts: int = 20

    def __post_init__(self):
        _integer(self.policy_rollouts, "policy_rollouts", 1)


@dataclass(frozen=True)
class RunConfig:
    data: str
    solution: str | None = None
    output: str = "artifacts/reports"
    seeds: tuple[int, ...] = (0,)
    problem: ProblemParameters = field(default_factory=ProblemParameters)
    agent: AgentParameters = field(default_factory=AgentParameters)
    training: TrainingParameters = field(default_factory=TrainingParameters)
    evaluation: EvaluationParameters = field(default_factory=EvaluationParameters)

    def __post_init__(self):
        if not self.seeds:
            raise ValueError("seeds debe ser una lista no vacía")
        for seed in self.seeds:
            _integer(seed, "cada semilla", 0)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Las semillas deben ser distintas")

    def as_dict(self) -> dict:
        return {**asdict(self), "seeds": list(self.seeds)}


SECTIONS = {"problem": ProblemParameters, "agent": AgentParameters,
            "training": TrainingParameters, "evaluation": EvaluationParameters}
TOP_LEVEL = {"data", "solution", "output", "seeds", *SECTIONS}


def _section(name: str, values) -> object:
    kind = SECTIONS[name]
    if values is None:
        return kind()
    if not isinstance(values, dict):
        raise ValueError(f"La sección {name} debe ser un diccionario")
    unknown = set(values) - set(kind.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Claves desconocidas en {name}: {sorted(unknown)}")
    return kind(**values)


def parse_config(raw) -> RunConfig:
    """Strict: unknown keys are an error, so a typo is never silently ignored."""
    if not isinstance(raw, dict):
        raise ValueError("La configuración debe ser un diccionario YAML")
    unknown = set(raw) - TOP_LEVEL
    if unknown:
        raise ValueError(f"Claves desconocidas en la configuración: {sorted(unknown)}")
    if not raw.get("data"):
        raise ValueError("La configuración debe indicar data")
    seeds = raw.get("seeds", [0])
    if not isinstance(seeds, list):
        raise ValueError("seeds debe ser una lista, por ejemplo [0, 1, 2]")
    return RunConfig(data=str(raw["data"]),
                     solution=None if raw.get("solution") is None else str(raw["solution"]),
                     output=str(raw.get("output", "artifacts/reports")), seeds=tuple(seeds),
                     **{name: _section(name, raw.get(name)) for name in SECTIONS})


def load_config(path: str | Path) -> RunConfig:
    file = project_path(path)
    if not file.is_file():
        raise ValueError(f"No existe la configuración {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML inválido en {file}: {exc}") from exc
    return parse_config(raw)
