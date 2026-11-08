"""Decreasing exploration: epsilon is recomputed at every training transition.

Except for constant wich is for comparison every schedule goes from epsilon0 (first transition)
to 0 (end of the budget) and is written in terms of the progress p = t / T, where t is the current
transition and T the total budget. The speed of the decrease is therefore set by the budget itself and the
schedule adds no hidden constants: the only choices are epsilon0 and the shape.

Designed to hold more shapes (e.g. logarithmic, exponential, slow). To add one, write a
function (epsilon0, progress) -> epsilon that returns epsilon0 at p=0 and 0 at p=1, and
register it in SCHEDULES; `epsilon_schedule` in the YAML then accepts its name.
"""

from typing import Protocol

class Schedule(Protocol):
    """Contract of every epsilon schedule: epsilon at progress p in [0, 1]."""
    def __call__(self, epsilon0:float, progress: float) -> float : ...


def linear(epsilon0: float, progress: float)->float:
    return epsilon0*(1-progress)

def constant(epsilon0: float, _progress: float)->float:
    return epsilon0



SCHEDULES: dict[str, Schedule] = {"constant": constant, "linear": linear}


def epsilon_at(schedule: str, epsilon0: float, transition: int, total: int) -> float:
    """Epsilon for training transition `transition` (0-based) of a budget of `total`."""
    if schedule not in SCHEDULES:
        raise ValueError(f"epsilon_schedule debe ser uno de {sorted(SCHEDULES)}")
    if total < 1 or not 0 <= transition <= total:
        raise ValueError("La transición debe estar entre 0 y el presupuesto total")
    return SCHEDULES[schedule](epsilon0, transition / total)