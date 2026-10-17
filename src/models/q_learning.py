"""Sparse full-physical-state Q-learning. Experimental, NOT a verified paper replica."""

import random
from collections.abc import Hashable
from dataclasses import dataclass

from src.config import LearningParameters
from src.envs.mine_env import Action, State
from src.models.base import BaseModel


class QLearning(BaseModel):
    name = "experimental-full-state-q-learning"

    def __init__(self, parameters: LearningParameters, seed: int):
        self.parameters = parameters
        self.seed = seed
        self.rng = random.Random(seed)
        self.q: dict[tuple[Hashable, Action], float] = {}

    def value(self, state: State, action: Action) -> float:
        return self.q.get((self.state_key(state), action), 0.0)

    def state_key(self, state: State) -> Hashable:
        return state

    def select_action(self, state: State, actions: tuple[Action, ...], *, explore: bool = True) -> Action:
        if not actions:
            raise ValueError("No hay acciones disponibles")
        if explore and self.rng.random() < self.parameters.epsilon:
            return self.rng.choice(actions)
        best = max(self.value(state, a) for a in actions)
        ties = tuple(a for a in actions if self.value(state, a) == best)
        return self.rng.choice(ties) if explore else ties[0]

    def update(self, state: State, action: Action, reward: float, next_state: State,
               next_actions: tuple[Action, ...], terminal: bool) -> float:
        if not terminal and not next_actions:
            raise ValueError("Transición no terminal sin acciones siguientes")
        bootstrap = 0.0 if terminal else max(self.value(next_state, a) for a in next_actions)
        previous = self.value(state, action)
        updated = previous + self.parameters.alpha * (reward + self.parameters.gamma * bootstrap - previous)
        self.q[self.state_key(state), action] = updated
        return updated


@dataclass(frozen=True)
class PositionKey:
    position: int | None
    period: int


class SpatialQLearning(QLearning):
    """Position/period approximation with aliasing; not a full Markov-state table."""

    name = "paper-inspired-spatial-cone-q-learning"

    def state_key(self, state):
        return PositionKey(state.position, state.period)
