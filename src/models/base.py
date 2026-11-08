"""Minimal learning contract; models do not apply mining rules."""

from abc import ABC, abstractmethod

from src.envs.mine_env import State


class BaseModel(ABC):
    @abstractmethod
    def select_action(self, state: State, actions: tuple[str, ...]) -> str:
        raise NotImplementedError

    @abstractmethod
    def best_actions(self, state: State, actions: tuple[str, ...]) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: State, action: str, reward: float, next_state: State,
               next_actions: tuple[str, ...], terminal: bool) -> float:
        raise NotImplementedError
