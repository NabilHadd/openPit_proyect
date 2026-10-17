"""Experiments receive the mine and rules only, never workbook selections."""

from dataclasses import dataclass
from time import perf_counter

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Mine
from src.data.transforms import Precedences
from src.envs.mine_env import MineEnv
from src.envs.spatial_cone_env import SpatialConeEnv
from src.models.q_learning import QLearning, SpatialQLearning
from src.training.trainer import TrainingResult, train


@dataclass(frozen=True)
class Experiment:
    seed: int
    generation_seconds: float
    result: TrainingResult
    model: QLearning


def run_experiments(mine: Mine, precedences: Precedences, problem: ProblemParameters,
                    learning: LearningParameters, seeds: tuple[int, ...],
                    spatial: SpatialParameters | None = None) -> tuple[Experiment, ...]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Se requiere una lista no vacía de semillas distintas")
    experiments = []
    for seed in seeds:
        start = perf_counter()
        env = (MineEnv(mine, problem, precedences) if spatial is None else
               SpatialConeEnv(mine, problem, precedences, spatial))
        model = (QLearning if spatial is None else SpatialQLearning)(learning, seed)
        result = train(env, model, learning.episodes)
        experiments.append(Experiment(seed, perf_counter() - start, result, model))
    return tuple(experiments)
