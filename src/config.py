"""Paths are relative to the repository, never to the caller's working directory."""

from dataclasses import dataclass
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class ProblemParameters:
    periods: int = 1
    discount_rate: float = 0.0
    capacity: float | None = None
    capacity_unit: str = "blocks"

    def __post_init__(self):
        if isinstance(self.periods, bool) or not isinstance(self.periods, int) or self.periods < 1:
            raise ValueError("periods debe ser un entero positivo")
        if not math.isfinite(self.discount_rate) or self.discount_rate < 0:
            raise ValueError("discount_rate debe ser finita y no negativa")
        if self.capacity_unit not in {"blocks", "tonnes"}:
            raise ValueError("capacity_unit debe ser blocks o tonnes")
        if self.capacity is not None and (not math.isfinite(self.capacity) or self.capacity <= 0):
            raise ValueError("capacity debe ser finita y positiva")


@dataclass(frozen=True)
class LearningParameters:
    episodes: int = 100
    alpha: float = 0.2
    gamma: float = 1.0
    epsilon: float = 0.2

    def __post_init__(self):
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int) or self.episodes < 1:
            raise ValueError("episodes debe ser un entero positivo")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha debe estar en (0, 1]")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("gamma y epsilon deben estar en [0, 1]")


@dataclass(frozen=True)
class SpatialParameters:
    surface: str
    max_steps: int = 500
    construction_up: bool = False

    def __post_init__(self):
        if self.surface not in {"min-z", "max-z"}:
            raise ValueError("surface debe ser min-z o max-z")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps debe ser un entero positivo")
        if not isinstance(self.construction_up, bool):
            raise ValueError("construction_up debe ser booleano")
