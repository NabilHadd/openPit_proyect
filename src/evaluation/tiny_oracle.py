"""Exhaustive tiny-mine oracle, independent of environment transitions and Q-learning."""

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from math import fsum

from src.config import SpatialParameters
from src.data.dataset import Block, Mine
from src.data.transforms import Precedences


def tiny_mine() -> tuple[Mine, Precedences]:
    """Profitable ore behind a loss, plus optional losses that should be avoided."""
    mine = Mine((Block(1, 0, 0, 3, 5), Block(2, 0, 0, 2, -4), Block(3, 0, 0, 1, 15),
                 Block(4, 1, 0, 3, -1), Block(5, 2, 0, 3, -8), Block(6, 1, 0, 2, -6)))
    rules = Precedences({1: (), 2: (1, 4), 3: (2,), 4: (), 5: (), 6: (1, 4, 5)},
                        "tiny-explicit", "Synthetic six-block validation fixture")
    return mine, rules


@dataclass(frozen=True)
class TinyOracleResult:
    optimum: float
    closed_subset_optimum: float
    actions: tuple[str, ...]
    block_ids: tuple[int, ...]
    reachable_states: int
    transitions: int


def enumerate_tiny(mine: Mine, rules: Precedences, spatial: SpatialParameters) -> TinyOracleResult:
    """Enumerate closed subsets AND reachable spatial states.

    The subset optimum need not be attainable from the forced start or within the
    horizon. External mode enumerates stationary states without a countdown or
    collection cap. The action optimum is the target, not the subset bound.
    Only the one-period, uncapacitated, economically undiscounted case is supported.
    """
    if not 1 <= len(mine.blocks) <= 6:
        raise ValueError("La enumeración exhaustiva admite entre uno y seis bloques")
    if not spatial.construction_up:
        raise ValueError("La validación diminuta exige seis direcciones en ambas fases")
    rules.validate(mine)
    blocks = mine.by_id
    ids = tuple(blocks)
    subset_best = float("-inf")
    for size in range(len(ids) + 1):
        for subset in combinations(ids, size):
            chosen = frozenset(subset)
            if all(rules.predecessors[i] <= chosen for i in chosen):
                subset_best = max(subset_best, fsum(blocks[i].value for i in chosen))

    surface_z = (max if spatial.surface == "max-z" else min)(b.z for b in mine.blocks)
    start = min((b for b in mine.blocks if b.z == surface_z), key=lambda b: (-b.value, b.id)).id
    positions = {(b.x, b.y, b.z): b.id for b in mine.blocks}
    dz = -1 if spatial.surface == "max-z" else 1
    offsets = {"north": (-1, 0, 0), "south": (1, 0, 0), "west": (0, -1, 0),
               "east": (0, 1, 0), "down": (0, 0, dz), "up": (0, 0, -dz)}
    finite = spatial.horizon_mode == "finite"
    initial = frozenset(), None, spatial.max_steps if finite else 0
    queue = deque([(initial, ())])
    seen = {initial}
    best = float("-inf")
    best_actions = best_ids = ()
    transitions = 0
    while queue:
        (taken, position, remaining), route = queue.popleft()
        if position is None:
            destinations = [("start", start)]
        else:
            block = blocks[position]
            destinations = [(a, positions[(block.x + dx, block.y + dy, block.z + z)])
                            for a, (dx, dy, z) in offsets.items()
                            if (block.x + dx, block.y + dy, block.z + z) in positions]
            transitions += 1  # Stop is always available after the forced start.
            total = fsum(blocks[i].value for i in taken)
            if total > best:
                best, best_actions, best_ids = total, route + ("stop",), tuple(sorted(taken))
        for action, destination in destinations:
            transitions += 1
            closure, pending = set(taken), [destination]
            while pending:
                current = pending.pop()
                if current not in closure:
                    closure.add(current)
                    pending.extend(rules.predecessors[current])
            following = frozenset(closure), destination, remaining - 1 if finite else 0
            next_route = route + (action,)
            terminal = (finite and remaining == 1) or len(closure) == len(blocks)
            if terminal:
                total = fsum(blocks[i].value for i in closure)
                if total > best:
                    best, best_actions, best_ids = total, next_route, tuple(sorted(closure))
            elif following not in seen:
                seen.add(following)
                queue.append((following, next_route))
    return TinyOracleResult(best, subset_best, best_actions, best_ids, len(seen), transitions)
