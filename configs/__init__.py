"""Configuration package: the common YAML plus one YAML per model.

    config.yml    + config_settings.py  shared by every model: data, problem, environment, seeds
    <model>.yml   + model_settings.py   hyperparameters of one model and its tuning distributions;
                                        the params dataclass lives in src/models/<model>.py

Paths are relative to the repository, never to the caller. Every reader is strict: an
unknown key is an error, so a typo is never silently ignored. Nothing in configs/
imports from src/.
"""

from dataclasses import dataclass, fields
import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def integer(value, name: str, minimum: int):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}")


def read_yaml(path: str | Path):
    """Raw content of a YAML file; the parse_* functions check its shape."""
    file = project_path(path)
    if not file.is_file():
        raise ValueError(f"No existe la configuración {file}")
    try:
        return yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML inválido en {file}: {exc}") from exc


def check_keys(values, allowed, where: str) -> dict:
    """values must be a mapping whose keys all belong to allowed."""
    if not isinstance(values, dict):
        raise ValueError(f"{where} debe ser un diccionario")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Claves desconocidas en {where}: {sorted(unknown)}")
    return values


def section(kind, values, where: str):
    """YAML mapping -> frozen dataclass `kind`; a missing section gives its defaults."""
    if values is None:
        return kind()
    return kind(**check_keys(values, {f.name for f in fields(kind)}, where))


DISTRIBUTIONS = {"uniform": ("low", "high"), "loguniform": ("low", "high"), "randint": ("low", "high"),
                 "choice": ("values",)}
CONTINUOUS = ("uniform", "loguniform")  # never for an integer hyperparameter


def _real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Distribution:
    """Random-search distribution of one hyperparameter, as written in a model YAML.

    uniform: any value in [low, high], all equally likely.
    loguniform: uniform over the logarithm in [low, high] (low > 0): every order of
        magnitude is equally likely, for parameters whose scale matters more than the value.
    randint: any integer in [low, high], both included, all equally likely (integer
        hyperparameters, e.g. planning steps or hidden units).
    choice: one of `values`, all equally likely.

    Only data: the tuning mode turns it into a sampler, and it can go into report.json.
    """
    distribution: str
    low: float | None = None
    high: float | None = None
    values: tuple = ()

    def __post_init__(self):
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution debe ser una de {sorted(DISTRIBUTIONS)}")
        if self.distribution == "choice":
            if not self.values or self.low is not None or self.high is not None:
                raise ValueError("choice requiere solo values, una lista no vacía")
            return
        if self.distribution == "randint":
            if self.values or not all(isinstance(v, int) and not isinstance(v, bool) for v in (self.low, self.high)):
                raise ValueError("randint requiere low y high enteros")
        elif self.values or not _real(self.low) or not _real(self.high):
            raise ValueError(f"{self.distribution} requiere low y high numéricos y finitos")
        if not self.low < self.high:
            raise ValueError("low debe ser menor que high")
        if self.distribution == "loguniform" and self.low <= 0:
            raise ValueError("loguniform requiere low > 0")


def distribution(values, where: str) -> Distribution:
    """YAML mapping {distribution: ..., low/high | values} -> Distribution."""
    kind = values.get("distribution") if isinstance(values, dict) else None
    if not isinstance(kind, str) or kind not in DISTRIBUTIONS:
        raise ValueError(f"{where} debe indicar distribution: una de {sorted(DISTRIBUTIONS)}")
    required = DISTRIBUTIONS[kind]
    check_keys(values, {"distribution", *required}, where)
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{where}: {kind} requiere {missing}")
    if kind == "choice" and not isinstance(values["values"], list):
        raise ValueError(f"{where}.values debe ser una lista")
    try:
        if kind == "choice":
            return Distribution(kind, values=tuple(values["values"]))
        return Distribution(kind, low=values["low"], high=values["high"])
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
