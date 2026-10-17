"""Deterministic experimental MDP, not a reconstruction of the missing paper."""

from dataclasses import dataclass
import math

from src.config import ProblemParameters
from src.data.dataset import Mine
from src.data.transforms import Precedences
from src.evaluation.metrics import Extraction, Plan, block_usage, validate_capacity_data
from src.training.rewards import extraction_reward


STOP = "stop"
ADVANCE = "advance_period"
Action = int | str


@dataclass(frozen=True)
class State:
    extracted: frozenset[int]
    period: int
    used: float
    terminal: bool = False


class MineEnv:
    def __init__(self, mine: Mine, parameters: ProblemParameters, precedences: Precedences):
        precedences.validate(mine)
        validate_capacity_data(mine, parameters)
        self.mine = mine
        self.parameters = parameters
        self.precedences = precedences
        self.blocks = mine.by_id
        self.reset()

    def reset(self, *, training: bool = True) -> State:
        self.state = State(frozenset(), 0, 0.0)
        self.plan = Plan()
        self._finish_if_exhausted()
        return self.state

    @property
    def episode_step_bound(self) -> int:
        return len(self.mine.blocks) + self.parameters.periods

    def _extractable(self) -> tuple[int, ...]:
        available = []
        for block_id in sorted(self.blocks):
            if block_id in self.state.extracted:
                continue
            if not self.precedences.predecessors[block_id] <= self.state.extracted:
                continue
            if self.parameters.capacity is not None:
                used = self.state.used + block_usage(self.blocks[block_id], self.parameters)
                capacity = self.parameters.capacity
                if used > capacity and not math.isclose(used, capacity, rel_tol=1e-12):
                    continue
            available.append(block_id)
        return tuple(available)

    def _finish_if_exhausted(self):
        exhausted = len(self.state.extracted) == len(self.blocks)
        no_future = self.state.period == self.parameters.periods - 1 and not self._extractable()
        if exhausted or no_future:
            self.state = State(self.state.extracted, self.state.period, self.state.used, True)

    def available_actions(self) -> tuple[Action, ...]:
        if self.state.terminal:
            return ()
        actions = self._extractable()
        if self.state.period + 1 < self.parameters.periods:
            actions += (ADVANCE,)
        return actions + (STOP,)

    def step(self, action: Action) -> tuple[State, float, bool]:
        if isinstance(action, bool) or not isinstance(action, (int, str)) or action not in self.available_actions():
            raise ValueError(f"Acción no disponible: {action!r}")
        state = self.state
        if action == STOP:
            self.state = State(state.extracted, state.period, state.used, True)
            return self.state, 0.0, True
        if action == ADVANCE:
            self.state = State(state.extracted, state.period + 1, 0.0)
            reward = 0.0
        else:
            block = self.blocks[action]
            reward = extraction_reward(block.value, state.period, self.parameters.discount_rate)
            used = state.used
            if self.parameters.capacity is not None:
                used += block_usage(block, self.parameters)
            # Compute all potentially failing values before committing the transition.
            plan = Plan(self.plan.extractions + (Extraction(action, state.period),))
            next_state = State(state.extracted | {action}, state.period, used)
            self.plan = plan
            self.state = next_state
        self._finish_if_exhausted()
        return self.state, reward, self.state.terminal
