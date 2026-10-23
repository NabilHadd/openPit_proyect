"""Streaming 2x2 diagnostic suite. Never imports a reference or benchmark."""

from dataclasses import dataclass, replace
from time import perf_counter

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.envs.spatial_cone_env import SpatialConeEnv
from src.evaluation.experiment_metrics import ControlDiagnostics
from src.models.experimental_q_learning import ControlQLearning
from src.training.trainer import TrainingResult, train


@dataclass(frozen=True)
class ControlSpec:
    id: str
    state_encoding: str
    gamma: float


CONTROL_SPECS = (
    ControlSpec("A_position_g07", "position", 0.7),
    ControlSpec("B_position_g1", "position", 1.0),
    ControlSpec("C_physical_g07", "physical", 0.7),
    ControlSpec("D_physical_g1", "physical", 1.0),
)


@dataclass(frozen=True)
class ControlRun:
    spec: ControlSpec
    seed: int
    learning: LearningParameters
    generation_seconds: float
    diagnostic_summary_seconds: float
    result: TrainingResult
    model: ControlQLearning
    diagnostics: dict


def validate_controls(problem: ProblemParameters, spatial: SpatialParameters, seeds):
    if problem != ProblemParameters():
        raise ValueError("La suite reducida requiere un período sin capacidad ni descuento")
    if not spatial.construction_up:
        raise ValueError("El control exige seis direcciones idénticas en aprendizaje y evaluación")
    if spatial.horizon_mode != "finite":
        raise ValueError("Los controles históricos requieren el horizonte finito original")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Se requieren semillas distintas y una lista no vacía")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Cada semilla debe ser un entero")


def iter_control_runs(mine, precedences, problem: ProblemParameters, spatial: SpatialParameters,
                      learning: LearningParameters, seeds: tuple[int, ...]):
    """Yield one seed at a time so callers need not retain forty Q tables."""
    validate_controls(problem, spatial, seeds)
    for spec in CONTROL_SPECS:
        effective = replace(learning, gamma=spec.gamma)
        for seed in seeds:
            started = perf_counter()
            env = SpatialConeEnv(mine, problem, precedences, spatial)
            model = ControlQLearning(effective, seed, spec.state_encoding)
            diagnostics = ControlDiagnostics(model.state_key)
            result = train(env, model, effective.episodes, diagnostics=diagnostics)
            generation_seconds = perf_counter() - started
            summarized = perf_counter()
            payload = diagnostics.payload(len(model.q))
            summary_seconds = perf_counter() - summarized
            yield ControlRun(spec, seed, effective, generation_seconds, summary_seconds,
                             result, model, payload)
