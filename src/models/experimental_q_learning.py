"""Diagnostic controls only; not registered as a final or paper-replica model."""

from collections.abc import Hashable
from dataclasses import replace

from src.envs.mine_env import State
from src.envs.spatial_cone_env import SpatialState
from src.models.q_learning import PositionKey, QLearning


def control_physical_key(state: State) -> SpatialState:
    if not isinstance(state, SpatialState):
        raise ValueError("El control requiere SpatialState y el entorno de conos")
    # Valid ONLY when learning/construction have identical six-direction dynamics.
    return replace(state, phase="experimental-six-direction-control")


class ControlQLearning(QLearning):
    """Same Q update/RNG for both encodings; phase normalization is control-specific."""

    def __init__(self, parameters, seed: int, state_encoding: str):
        if state_encoding not in {"position", "physical"}:
            raise ValueError("state_encoding experimental debe ser position o physical")
        super().__init__(parameters, seed)
        self.state_encoding = state_encoding
        self.name = f"experimental-control-{state_encoding}-q-learning"

    def state_key(self, state: State) -> Hashable:
        physical = control_physical_key(state)
        if self.state_encoding == "position":
            return PositionKey(physical.position, physical.period)
        return physical
