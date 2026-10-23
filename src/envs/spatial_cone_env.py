"""Paper-inspired exploration and atomic cone extraction, with explicit assumptions.

The complete physical state stays here; a compact Q projection is NOT this MDP.
Training allows up; construction omits it unless explicitly enabled as an extension.
"""

from dataclasses import dataclass
import math

from src.config import SpatialParameters
from src.data.transforms import order_selection
from src.envs.mine_env import ADVANCE, STOP, MineEnv, State
from src.evaluation.metrics import Extraction, Plan, block_usage
from src.training.rewards import extraction_reward


START = "start"
DIRECTIONS = ("north", "south", "west", "east", "down", "up")


@dataclass(frozen=True)
class SpatialState(State):
    position: int | None = None
    steps_left: int = 0
    phase: str = "learning"


class SpatialConeEnv(MineEnv):
    def __init__(self, mine, parameters, precedences, spatial: SpatialParameters):
        self.spatial = spatial
        precedences.validate(mine)
        self.order = order_selection(mine.by_id, precedences)
        self.cones = {}
        for block_id in self.order:
            cone = {block_id}
            for predecessor in precedences.predecessors[block_id]:
                cone.update(self.cones[predecessor])
            self.cones[block_id] = frozenset(cone)
        self.positions = {(b.x, b.y, b.z): b.id for b in mine.blocks}
        surface_z = (min if spatial.surface == "min-z" else max)(b.z for b in mine.blocks)
        self.start_id = min((b for b in mine.blocks if b.z == surface_z), key=lambda b: (-b.value, b.id)).id
        self.down_dz = 1 if spatial.surface == "min-z" else -1
        super().__init__(mine, parameters, precedences)

    @property
    def episode_step_bound(self) -> int:
        return self.spatial.max_steps

    def reset(self, *, training: bool = True) -> SpatialState:
        clock = self.spatial.max_steps if self.spatial.horizon_mode == "finite" else 0
        self.state = SpatialState(frozenset(), 0, 0.0, False, None, clock,
                                  "learning" if training else "construction")
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
        block = self.blocks[self.state.position]
        offsets = {"north": (-1, 0, 0), "south": (1, 0, 0), "west": (0, -1, 0),
                   "east": (0, 1, 0), "down": (0, 0, self.down_dz), "up": (0, 0, -self.down_dz)}
        if action not in offsets:
            return None
        dx, dy, dz = offsets[action]
        return self.positions.get((block.x + dx, block.y + dy, block.z + dz))

    def available_actions(self) -> tuple[str, ...]:
        if self.state.terminal:
            return ()
        if self.state.position is None:
            return (START,)
        directions = DIRECTIONS if self.state.phase == "learning" or self.spatial.construction_up else DIRECTIONS[:-1]
        actions = tuple(a for a in directions if (dest := self.destination(a)) is not None and self._fits(self._new_blocks(dest)))
        if self.state.period + 1 < self.parameters.periods:
            actions += (ADVANCE,)
        return actions + (STOP,)

    def step(self, action: str) -> tuple[SpatialState, float, bool]:
        if not isinstance(action, str) or action not in self.available_actions():
            raise ValueError(f"Acción espacial no disponible: {action!r}")
        state = self.state
        used, period, position = state.used, state.period, state.position
        new_blocks, reward = (), 0.0
        reason = "stop" if action == STOP else None
        if action == ADVANCE:
            period, used = period + 1, 0.0
        elif action != STOP:
            position = self.destination(action)
            new_blocks = self._new_blocks(position)
            reward = sum(extraction_reward(self.blocks[i].value, period, self.parameters.discount_rate) for i in new_blocks)
            used += self._usage(new_blocks)
        plan = Plan(self.plan.extractions + tuple(Extraction(i, period) for i in new_blocks))
        extracted = state.extracted | frozenset(new_blocks)
        steps_left = state.steps_left - 1 if self.spatial.horizon_mode == "finite" else 0
        if reason is None:
            if len(extracted) == len(self.blocks):
                reason = "all_extracted"
            elif self.spatial.horizon_mode == "finite" and steps_left == 0:
                reason = "step_limit"
        next_state = SpatialState(extracted, period, used, reason is not None, position, steps_left, state.phase)
        self.plan, self.state = plan, next_state
        self.action_history += (action,)
        self.termination_reason = reason
        return self.state, reward, self.state.terminal
