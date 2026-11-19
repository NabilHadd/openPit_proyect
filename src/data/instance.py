"""The problem every model receives, built once per run from the common configuration."""

from dataclasses import dataclass, field

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.dataset import Mine
from src.data.precedences import Precedences


@dataclass(frozen=True)
class MineInstance:
    """The mine, its precedence rule, the mining problem and how MineEnv presents it.

    Models never modify it: a transformation a model needs happens inside that model, and
    every plan is evaluated against this instance."""
    mine: Mine
    precedences: Precedences
    problem: ProblemParameters = field(default_factory=ProblemParameters)
    environment: EnvironmentParameters = field(default_factory=EnvironmentParameters)
