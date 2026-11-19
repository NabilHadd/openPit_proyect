"""Q-learning configuration (configs/q_learning.yml): hyperparameters and their distributions.

`params` are the hyperparameters a run uses. `tuning.distributions` is the search space of
the tuning mode (random search): one distribution per hyperparameter to tune; the ones
left out keep their value of `params`.
"""

from dataclasses import asdict, dataclass, field, fields

from configs import Distribution, check_keys, distribution, integer, read_yaml, section
from src.models.exploration import SCHEDULES


FIXED = ("transitions", "policy_rollouts")  # budget and evaluation: equal for every configuration tried


@dataclass(frozen=True)
class QLearningParams:
    """Hyperparameters of tabular Q-learning.

    state: what the Q table remembers of a state (physical: position + extracted blocks;
    position: only the position). transitions: training budget per seed.
    policy_rollouts: plans built with the learned policy after training.
    """
    state: str = "physical"
    alpha: float = 1.0
    gamma: float = 1.0
    epsilon: float = 0.3                # initial epsilon of the schedule (fixed epsilon with "constant")
    epsilon_schedule: str = "constant"  # shape of the decrease, see src/models/exploration.py
    transitions: int = 100_000
    policy_rollouts: int = 20

    def __post_init__(self):
        if self.state not in {"physical", "position"}:
            raise ValueError("state debe ser physical o position")
        if self.epsilon_schedule not in SCHEDULES:
            raise ValueError(f"epsilon_schedule debe ser uno de {sorted(SCHEDULES)}")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha debe estar en (0, 1]")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("gamma y epsilon deben estar en [0, 1]")
        integer(self.transitions, "transitions", 1)
        integer(self.policy_rollouts, "policy_rollouts", 1)


@dataclass(frozen=True)
class QLearningSettings:
    params: QLearningParams = field(default_factory=QLearningParams)
    distributions: dict[str, Distribution] = field(default_factory=dict)


def _distributions(values, params: QLearningParams, where: str) -> dict[str, Distribution]:
    """Checked when the file is read, not in the middle of a tuning run: every value of a
    choice and both ends of a range must be valid in params. The numeric rules of
    QLearningParams are intervals, so a range whose ends are valid is valid everywhere."""
    check_keys(values, {f.name for f in fields(QLearningParams)}, where)
    fixed = [name for name in FIXED if name in values]
    if fixed:
        raise ValueError(f"{where}: {fixed} no se ajustan; transitions es el presupuesto y policy_rollouts "
                         "la evaluación, iguales para toda configuración probada")
    result = {}
    for name, spec in values.items():
        candidates = distribution(spec, f"{where}.{name}")
        for candidate in candidates.values or (candidates.low, candidates.high):
            try:
                QLearningParams(**{**asdict(params), name: candidate})
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}.{name}: {candidate!r} no es un valor válido ({exc})") from None
        result[name] = candidates
    return result


def parse_q_learning_settings(raw, where: str = "q_learning") -> QLearningSettings:
    check_keys(raw, {"params", "tuning"}, where)
    params = section(QLearningParams, raw.get("params"), f"{where}.params")
    tuning = check_keys(raw.get("tuning") or {}, {"distributions"}, f"{where}.tuning")
    return QLearningSettings(params, _distributions(tuning.get("distributions") or {}, params,
                                                    f"{where}.tuning.distributions"))


def load_q_learning_settings(path) -> QLearningSettings:
    return parse_q_learning_settings(read_yaml(path))
