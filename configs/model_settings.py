"""Configuration of any model (configs/<model>.yml): hyperparameters and their distributions.

    params:                 values a run uses (fields of the model's params dataclass)
    tuning:
      distributions:        search space of the tuning mode, one entry per tuned field

The params dataclass lives in the model's own module (src/models/<model>.py) and holds
every rule of its hyperparameters in __post_init__. Two optional class attributes:
    FIXED: fields that are never tuned (budget, evaluation): equal for every configuration.
    SMOKE: cheap values for the contract tests (tests/test_estimator_contract.py).

This module knows no model: it receives the dataclass, so configs/ imports nothing from src/.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import get_type_hints

from configs import CONTINUOUS, Distribution, check_keys, distribution, read_yaml, section


@dataclass(frozen=True)
class ModelSettings:
    params: object                                   # instance of the model's params dataclass
    distributions: dict[str, Distribution] = field(default_factory=dict)


def _distributions(values, params, where: str) -> dict[str, Distribution]:
    """Checked when the file is read, not in the middle of a tuning run: every value of a
    choice and both ends of a range must be valid in params. A numeric rule must therefore
    be an interval, so a range whose ends are valid is valid everywhere. An integer field
    only takes randint or choice: a continuous draw would fail in the middle of a search."""
    kind = type(params)
    types = get_type_hints(kind)
    check_keys(values, {f.name for f in fields(kind)}, where)
    fixed = [name for name in getattr(kind, "FIXED", ()) if name in values]
    if fixed:
        raise ValueError(f"{where}: {fixed} no se ajustan (FIXED de {kind.__name__}): son presupuesto o "
                         "evaluación, iguales para toda configuración probada")
    result = {}
    for name, spec in values.items():
        candidates = distribution(spec, f"{where}.{name}")
        integer_field = types.get(name) is int
        if integer_field and candidates.distribution in CONTINUOUS:
            raise ValueError(f"{where}.{name} es entero: use randint o choice, no {candidates.distribution}")
        if candidates.distribution == "randint" and not integer_field:
            raise ValueError(f"{where}.{name} no es entero: randint solo se usa en hiperparámetros enteros")
        for candidate in candidates.values or (candidates.low, candidates.high):
            try:
                kind(**{**asdict(params), name: candidate})
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}.{name}: {candidate!r} no es un valor válido ({exc})") from None
        result[name] = candidates
    return result


def parse_model_settings(raw, params_cls, where: str) -> ModelSettings:
    """YAML content -> ModelSettings; a missing section gives the dataclass defaults."""
    check_keys(raw, {"params", "tuning"}, where)
    params = section(params_cls, raw.get("params"), f"{where}.params")
    tuning = check_keys(raw.get("tuning") or {}, {"distributions"}, f"{where}.tuning")
    return ModelSettings(params, _distributions(tuning.get("distributions") or {}, params,
                                                f"{where}.tuning.distributions"))


def load_model_settings(path, params_cls, where: str) -> ModelSettings:
    return parse_model_settings(read_yaml(path), params_cls, where)
