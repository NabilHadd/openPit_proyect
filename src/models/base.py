"""Minimal policy/learning contract; models do not apply mining rules."""

from abc import ABC, abstractmethod

from src.envs.mine_env import Action, State


class BaseModel(ABC):
    @abstractmethod
    def select_action(self, state: State, actions: tuple[Action, ...], *, explore: bool = True) -> Action:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: State, action: Action, reward: float, next_state: State,
               next_actions: tuple[Action, ...], terminal: bool) -> float:
        raise NotImplementedError
