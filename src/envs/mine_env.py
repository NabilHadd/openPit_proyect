"""Spatial mine environment: a cursor moves through the grid and extracts whole cones.

Arriving at a block extracts it together with every pending block of its cone (its
transitive predecessors), so every intermediate plan is feasible. Revisiting extracted
blocks only moves the cursor. The same six directions apply when learning and when
building a plan with the learned policy.

Two ways to use it:
- Pure simulator, for models that evaluate transitions from arbitrary states (search,
  model-based planning): initial_state(), actions(state) and transition(state, action)
  never change the environment; transition only counts its calls in `calls`.
- Live episode, as a thin wrapper of the simulator: reset() and step(action) keep the
  current state, the plan built so far, the actions taken and why the episode ended.
- Cone view, for search methods (beam search): cone_actions(state), extract(state, block)
  and cone_step(state, action), pure as well, with no cursor and no step clock; they share
  the core of transition (reward, capacity, end). search_key(state) deduplicates nodes.

`calls` counts simulated transitions (transition, extract, cone_step, step included), not
the calls to actions() or cone_actions(), which also cost time.

State does not carry the plan (which period each block left in), so a model that keeps
several partial plans stores each one itself, from Step.extracted.
"""

from dataclasses import dataclass
import math

from configs.config_settings import EnvironmentParameters, ProblemParameters
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


@dataclass(frozen=True)
class Step:
    """Result of one transition. extracted: the new blocks, in a valid order, all taken out
    in state.period; reason: why the episode ended (None if it goes on)."""
    state: State
    reward: float
    terminal: bool
    extracted: tuple[int, ...]
    reason: str | None


class MineEnv:
    def __init__(self, mine: Mine, problem: ProblemParameters, precedences: Precedences,
                 environment: EnvironmentParameters):
        precedences.validate(mine)
        validate_capacity_data(mine, problem)
        self.mine, self.parameters, self.precedences, self.environment = mine, problem, precedences, environment
        self.blocks = mine.by_id
        self.finite = environment.horizon == "finite"
        self.reachable_count = sum(b.reachable for b in mine.blocks)
        self.order = order_selection(mine.by_id, precedences)
        self.cones = {}
        for block_id in self.order:
            cone = {block_id}
            for predecessor in precedences.predecessors[block_id]:
                cone.update(self.cones[predecessor])
            self.cones[block_id] = frozenset(cone)
        rank = {block_id: index for index, block_id in enumerate(self.order)}
        # each cone already in the global valid order, so pending blocks are a filter of it
        self.cone_order = {block_id: tuple(sorted(cone, key=rank.__getitem__)) for block_id, cone in self.cones.items()}
        self.positions = {(b.x, b.y, b.z): b.id for b in mine.blocks}
        surface_z = max(b.z for b in mine.blocks)
        self.start_id = min((b for b in mine.blocks if b.z == surface_z and b.reachable),
                            key=lambda b: (-b.value, b.id)).id
        self.calls = 0  # transitions simulated (step included) since the environment was built
        self.reset()

    # --- pure simulator -------------------------------------------------------------------

    def initial_state(self) -> State:
        return State(frozenset(), 0, 0.0, None, self.environment.max_steps if self.finite else 0)

    def new_blocks(self, state: State, destination: int) -> tuple[int, ...]:
        """Pending blocks of the cone of destination, in a valid extraction order."""
        return tuple(i for i in self.cone_order[destination] if i not in state.extracted)

    def _usage(self, new_blocks) -> float:
        if self.parameters.capacity is None:
            return 0.0
        return sum(block_usage(self.blocks[i], self.parameters) for i in new_blocks)

    def fits(self, state: State, new_blocks) -> bool:
        if self.parameters.capacity is None:
            return True
        total = state.used + self._usage(new_blocks)
        return total <= self.parameters.capacity or math.isclose(total, self.parameters.capacity, rel_tol=1e-12)

    def destination(self, state: State, action: str) -> int | None:
        if state.position is None:
            return self.start_id if action == START else None
        if action not in OFFSETS:
            return None
        block = self.blocks[state.position]
        dx, dy, dz = OFFSETS[action]
        return self.positions.get((block.x + dx, block.y + dy, block.z + dz))

    def actions(self, state: State) -> tuple[str, ...]:
        if state.terminal:
            return ()
        if state.position is None:
            return (START,)
        actions = tuple(a for a in DIRECTIONS if (dest := self.destination(state, a)) is not None
                        and self.blocks[dest].reachable and self.fits(state, self.new_blocks(state, dest)))
        if state.period + 1 < self.parameters.periods:
            actions += (ADVANCE,)
        return actions + (STOP,)

    def transition(self, state: State, action: str) -> Step:
        """Cursor view. Pure: the environment is not changed, only `calls` is counted.
        action must be one of actions(state); step checks it, a caller of transition is trusted."""
        if action in (ADVANCE, STOP):
            return self._apply(state, action, state.position, tick=True)
        return self._apply(state, action, self.destination(state, action), tick=True)

    def _apply(self, state: State, action: str, position: int | None, *, tick: bool) -> Step:
        """Core shared by both views: extract the pending cone of position (unless action is
        ADVANCE or STOP), with discounted reward, capacity used and the end of the episode.
        tick: whether the step clock runs (cursor view only)."""
        self.calls += 1
        used, period = state.used, state.period
        new_blocks, reward = (), 0.0
        reason = STOP if action == STOP else None
        if action == ADVANCE:
            period, used = period + 1, 0.0
        elif action != STOP:
            new_blocks = self.new_blocks(state, position)
            reward = sum(extraction_reward(self.blocks[i].value, period, self.parameters.discount_rate)
                         for i in new_blocks)
            used += self._usage(new_blocks)
        extracted = state.extracted | frozenset(new_blocks)
        steps_left = (state.steps_left - 1 if tick else state.steps_left) if self.finite else 0
        if reason is None:
            if len(extracted) == self.reachable_count:
                reason = "all_extracted"
            elif self.finite and tick and steps_left == 0:
                reason = "step_limit"
        following = State(extracted, period, used, position, steps_left, reason is not None)
        return Step(following, reward, following.terminal, new_blocks, reason)

    # --- cone view (search) ---------------------------------------------------------------

    def cone_actions(self, state: State) -> tuple[int | str, ...]:
        """Cone view, for search methods: every reachable pending block whose pending cone
        fits the capacity left (in the global valid order), then ADVANCE and STOP. There is
        no cursor and no forced start: any cone can be the first one."""
        if state.terminal:
            return ()
        blocks = tuple(b for b in self.order if self.blocks[b].reachable and b not in state.extracted
                       and self.fits(state, self.new_blocks(state, b)))
        tail = (ADVANCE,) if state.period + 1 < self.parameters.periods else ()
        return blocks + tail + (STOP,)

    def extract(self, state: State, block: int) -> Step:
        """Cone view: extract the pending cone of block in the current period. Pure, counts
        one call. The step clock does not run (it is a rule of the cursor walk), so the only
        ends are STOP and all_extracted. position becomes block, but search_key ignores it."""
        if state.terminal:
            raise ValueError("El estado es terminal")
        if block not in self.blocks or not self.blocks[block].reachable or block in state.extracted:
            raise ValueError(f"Bloque no extraíble: {block!r}")
        if not self.fits(state, self.new_blocks(state, block)):
            raise ValueError(f"El cono pendiente de {block} no cabe en la capacidad del período")
        return self._apply(state, "extract", block, tick=False)

    def cone_step(self, state: State, action: int | str) -> Step:
        """One action of cone_actions: a block id (extract its cone), ADVANCE or STOP."""
        if action in (ADVANCE, STOP):
            if action not in self.cone_actions(state):
                raise ValueError(f"Acción no disponible: {action!r}")
            return self._apply(state, action, state.position, tick=False)
        return self.extract(state, action)

    @staticmethod
    def search_key(state: State) -> tuple:
        """Key to deduplicate search nodes: what decides the future (extracted, period and
        capacity used), without the cursor position or the step clock. used is rounded so
        the same blocks summed in another order give the same key. Two nodes with the same
        key can carry plans of different VAN (blocks in other periods): keep the best."""
        return state.extracted, state.period, round(state.used, 9)

    # --- live episode ---------------------------------------------------------------------

    def reset(self) -> State:
        self.state = self.initial_state()
        self.plan = Plan()
        self.action_history = ()
        self.termination_reason = None
        if not self.fits(self.state, self.new_blocks(self.state, self.start_id)):
            raise ValueError("El cono del bloque inicial no cabe en la capacidad de un período")
        return self.state

    def available_actions(self) -> tuple[str, ...]:
        return self.actions(self.state)

    def step(self, action: str) -> tuple[State, float, bool]:
        if not isinstance(action, str) or action not in self.available_actions():
            raise ValueError(f"Acción no disponible: {action!r}")
        result = self.transition(self.state, action)
        self.plan = Plan(self.plan.extractions + tuple(Extraction(i, result.state.period) for i in result.extracted))
        self.state = result.state
        self.action_history += (action,)
        self.termination_reason = result.reason
        return result.state, result.reward, result.terminal
