"""Independent plan verifier; never repairs or silently reorders a submitted plan."""

from dataclasses import dataclass
import math

from src.config import ProblemParameters
from src.data.dataset import Mine
from src.data.transforms import Precedences
from src.training.rewards import discounted_value


@dataclass(frozen=True)
class Extraction:
    block_id: int
    period: int = 0


@dataclass(frozen=True)
class Plan:
    extractions: tuple[Extraction, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    feasible: bool | None
    raw_value: float
    npv: float
    errors: tuple[str, ...]
    precedence_status: str
    period_usage: tuple[float, ...]


def block_usage(block, parameters: ProblemParameters) -> float:
    if parameters.capacity_unit == "blocks":
        return 1.0
    if block.tonnes is None:
        raise ValueError("Capacidad en toneladas requiere tonelaje por bloque; no se infiere del conteo")
    return block.tonnes


def validate_capacity_data(mine: Mine, parameters: ProblemParameters):
    if parameters.capacity is not None:
        for block in mine.blocks:
            block_usage(block, parameters)


def evaluate_plan(mine: Mine, plan: Plan, parameters: ProblemParameters,
                  precedences: Precedences | None) -> Evaluation:
    if precedences is not None:
        precedences.validate(mine)
    validate_capacity_data(mine, parameters)
    by_id = mine.by_id
    seen, errors = set(), []
    raw_value = npv = 0.0
    usage = [0.0] * parameters.periods
    last_period = 0
    for index, extraction in enumerate(plan.extractions):
        block_id, period = extraction.block_id, extraction.period
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id not in by_id:
            errors.append(f"Paso {index}: bloque desconocido {block_id!r}")
            continue
        if block_id in seen:
            errors.append(f"Paso {index}: bloque duplicado {block_id}")
            continue
        if isinstance(period, bool) or not isinstance(period, int) or not 0 <= period < parameters.periods:
            errors.append(f"Paso {index}: período inválido {period!r}")
            continue
        if period < last_period:
            errors.append(f"Paso {index}: períodos no monótonos")
        last_period = period
        if precedences is not None:
            missing = precedences.predecessors[block_id] - seen
            if missing:
                errors.append(f"Bloque {block_id}: faltan predecesores anteriores en la secuencia {sorted(missing)}")
        block = by_id[block_id]
        seen.add(block_id)
        raw_value += block.value
        npv += discounted_value(block.value, period, parameters.discount_rate)
        if parameters.capacity is not None:
            usage[period] += block_usage(block, parameters)
    if parameters.capacity is not None:
        for period, used in enumerate(usage):
            if used > parameters.capacity and not math.isclose(used, parameters.capacity, rel_tol=1e-12):
                errors.append(f"Período {period}: capacidad excedida ({used} > {parameters.capacity})")
    status = "unverified" if precedences is None else ("conditional" if precedences.provisional else "specified")
    feasible = False if errors else (None if precedences is None else True)
    return Evaluation(feasible, raw_value, npv, tuple(errors), status, tuple(usage))
