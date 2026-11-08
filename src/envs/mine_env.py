"""Spatial mine environment: a cursor moves through the grid and extracts whole cones.

Arriving at a block extracts it together with every pending block of its cone (its
transitive predecessors), so every intermediate plan is feasible. Revisiting extracted
blocks only moves the cursor. The same six directions apply when learning and when
building a plan with the learned policy.
"""

from dataclasses import dataclass
import math

from src.config import ProblemParameters, TrainingParameters
from src.data.dataset import Mine
from src.data.precedences import Precedences, order_selection
from src.evaluation.metrics import Extraction, Plan, block_usage, validate_capacity_data
from src.training.rewards import extraction_reward


START = "start"
STOP = "stop"
ADVANCE = "advance_period"
DIRECTIONS = ("north", "south", "west", "east", "down", "up")
OFFSETS = {"north": (-1, 0, 0), "south": (1, 0, 0), "west": (0, -1, 0),
           "east": (0, 1, 0), "down": (0, 0, -1), "up": (0, 0, 1)}  # standard data: higher z = shallower


@dataclass(frozen=True)
class State:
    """Complete physical state. steps_left is the task clock (finite) or always 0 (external)."""
    extracted: frozenset[int]
    period: int
    used: float
    position: int | None
    steps_left: int
    terminal: bool = False


class MineEnv:
    def __init__(self, mine: Mine, problem: ProblemParameters, precedences: Precedences,
                 training: TrainingParameters):
        precedences.validate(mine)
        validate_capacity_data(mine, problem)
        self.mine, self.parameters, self.precedences, self.training = mine, problem, precedences, training
        self.blocks = mine.by_id
        self.finite = training.horizon == "finite"
        self.reachable_count = sum(b.reachable for b in mine.blocks)
        self.order = order_selection(mine.by_id, precedences)
        self.cones = {}
        for block_id in self.order:
            cone = {block_id}
            for predecessor in precedences.predecessors[block_id]:
                cone.update(self.cones[predecessor])
            self.cones[block_id] = frozenset(cone)
        self.positions = {(b.x, b.y, b.z): b.id for b in mine.blocks}
        surface_z = max(b.z for b in mine.blocks)
        self.start_id = min((b for b in mine.blocks if b.z == surface_z and b.reachable),
                            key=lambda b: (-b.value, b.id)).id
        self.reset()

    def reset(self) -> State:
        self.state = State(frozenset(), 0, 0.0, None, self.training.max_steps if self.finite else 0)
        self.plan = Plan()
        self.action_history = ()
        self.termination_reason = None
        if not self._fits(self._new_blocks(self.start_id)):
            raise ValueError("El cono del bloque inicial no cabe en la capacidad de un período")
        return self.state

    def _new_blocks(self, destination: int) -> tuple[int, ...]:
        pending = self.cones[destination] - self.state.extracted
        return tuple(i for i in self.order if i in pending)

    def _usage(self, new_blocks) -> float:
        if self.parameters.capacity is None:
            return 0.0
        return sum(block_usage(self.blocks[i], self.parameters) for i in new_blocks)

    def _fits(self, new_blocks) -> bool:
        if self.parameters.capacity is None:
            return True
        total = self.state.used + self._usage(new_blocks)
        return total <= self.parameters.capacity or math.isclose(total, self.parameters.capacity, rel_tol=1e-12)

    def destination(self, action: str) -> int | None:
        if self.state.position is None:
            return self.start_id if action == START else None
        if action not in OFFSETS:
            return None
        block = self.blocks[self.state.position]
        dx, dy, dz = OFFSETS[action]
        return self.positions.get((block.x + dx, block.y + dy, block.z + dz))

    def available_actions(self) -> tuple[str, ...]:
        if self.state.terminal:
            return ()
        if self.state.position is None:
            return (START,)
        actions = tuple(a for a in DIRECTIONS if (dest := self.destination(a)) is not None
                        and self.blocks[dest].reachable and self._fits(self._new_blocks(dest)))
        if self.state.period + 1 < self.parameters.periods:
            actions += (ADVANCE,)
        return actions + (STOP,)

    def step(self, action: str) -> tuple[State, float, bool]:
        if not isinstance(action, str) or action not in self.available_actions():
            raise ValueError(f"Acción no disponible: {action!r}")
        state = self.state
        used, period, position = state.used, state.period, state.position
        new_blocks, reward = (), 0.0
        reason = STOP if action == STOP else None
        if action == ADVANCE:
            period, used = period + 1, 0.0
        elif action != STOP:
            position = self.destination(action)
            new_blocks = self._new_blocks(position)
            reward = sum(extraction_reward(self.blocks[i].value, period, self.parameters.discount_rate)
                         for i in new_blocks)
            used += self._usage(new_blocks)
        extracted = state.extracted | frozenset(new_blocks)
        steps_left = state.steps_left - 1 if self.finite else 0
        if reason is None:
            if len(extracted) == self.reachable_count:
                reason = "all_extracted"
            elif self.finite and steps_left == 0:
                reason = "step_limit"
        self.plan = Plan(self.plan.extractions + tuple(Extraction(i, period) for i in new_blocks))
        self.state = State(extracted, period, used, position, steps_left, reason is not None)
        self.action_history += (action,)
        self.termination_reason = reason
        return self.state, reward, self.state.terminal
