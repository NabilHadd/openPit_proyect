"""Tabular Q-learning over the spatial mine environment."""

import random
from collections.abc import Hashable
from dataclasses import dataclass

from src.config import AgentParameters
from src.envs.mine_env import STOP, State
from src.models.base import BaseModel


@dataclass(frozen=True)
class PositionKey:
    """Compact key: forgets what was extracted. steps_left keeps the task clock in finite mode
    (it is always 0 in external mode), so a step limit never contradicts the key."""
    position: int | None
    period: int
    steps_left: int


class QLearning(BaseModel):
    name = "spatial-cone-q-learning"

    def __init__(self, parameters: AgentParameters, seed: int):
        self.parameters = parameters
        self.seed = seed
        self.rng = random.Random(seed)
        self.q: dict[tuple[Hashable, str], float] = {}

    def state_key(self, state: State) -> Hashable:
        if self.parameters.state == "position":
            return PositionKey(state.position, state.period, state.steps_left)
        return state

    def value(self, state: State, action: str) -> float:
        return self.q.get((self.state_key(state), action), 0.0)

    def best_actions(self, state: State, actions: tuple[str, ...]) -> tuple[str, ...]:
        """Every action tied at the maximum Q value."""
        if not actions:
            raise ValueError("No hay acciones disponibles")
        values = [self.value(state, a) for a in actions]
        best = max(values)
        return tuple(a for a, v in zip(actions, values) if v == best)

    def select_action(self, state: State, actions: tuple[str, ...], epsilon: float) -> str:
        """Training choice: epsilon-greedy, ties broken at random with the training RNG.

        epsilon is given by the trainer at every transition (decreasing schedule). A random
        exploration move never chooses STOP: stopping at random cut episodes short (~13
        steps) and deep blocks were rarely reached. STOP is still chosen when it is the
        action of highest Q."""
        if not actions:
            raise ValueError("No hay acciones disponibles")
        if self.rng.random() < epsilon:
            moves = tuple(a for a in actions if a != STOP)
            return self.rng.choice(moves or actions)
        return self.rng.choice(self.best_actions(state, actions))

    def update(self, state: State, action: str, reward: float, next_state: State,
               next_actions: tuple[str, ...], terminal: bool) -> float:
        """terminal=True only when the task really ends; an interruption still bootstraps."""
        if not terminal and not next_actions:
            raise ValueError("Transición no terminal sin acciones siguientes")
        bootstrap = 0.0 if terminal else max(self.value(next_state, a) for a in next_actions)
        previous = self.value(state, action)
        updated = previous + self.parameters.alpha * (reward + self.parameters.gamma * bootstrap - previous)
        self.q[self.state_key(state), action] = updated
        return updated
