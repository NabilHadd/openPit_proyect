"""Precedences between blocks: the project's single pit rule, plus the internal structure.

Rule given by the client: 45° slope, i.e. (with cubic blocks) a block requires the
9 blocks directly above it. Blocks without a complete overlying cone are unreachable
(`Block.reachable`, computed by `load_mine`); their predecessor list is irrelevant.

`Precedences` can also be built by hand (synthetic DAGs in tests and the tiny oracle).
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.data.dataset import Mine


PIT_RULE = "pit-9-above"
PIT_RULE_SOURCE = "Cliente (Aldo): talud de 45°, 9 bloques superiores; bloques de borde inalcanzables"


@dataclass(frozen=True)
class Precedences:
    predecessors: Mapping[int, frozenset[int]]
    rule: str = "explicit"
    source: str = "synthetic"

    def __post_init__(self):
        object.__setattr__(self, "predecessors", MappingProxyType({
            k: frozenset(v) for k, v in self.predecessors.items()
        }))

    def validate(self, mine: Mine):
        ids = set(mine.by_id)
        if set(self.predecessors) != ids:
            raise ValueError("Las precedencias deben definir todos los bloques")
        done = set()
        for block_id, predecessors in self.predecessors.items():
            if block_id in predecessors or not predecessors <= ids:
                raise ValueError("Predecesores desconocidos o autorreferencia")
        while len(done) < len(ids):
            ready = {i for i in ids - done if self.predecessors[i] <= done}
            if not ready:
                raise ValueError("Ciclo de precedencias")
            done.update(ready)


def pit_precedences(mine: Mine) -> Precedences:
    """Mine -> Precedences: each block requires the (up to) 9 blocks at z+1 with |dx|,|dy| <= 1.

    Reachable blocks always have all 9; edge blocks keep the ones that exist, but they
    are unreachable and can never be extracted.
    """
    positions = {(b.x, b.y, b.z): b.id for b in mine.blocks}
    predecessors = {
        b.id: frozenset(positions[(b.x + dx, b.y + dy, b.z + 1)]
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        if (b.x + dx, b.y + dy, b.z + 1) in positions)
        for b in mine.blocks
    }
    result = Precedences(predecessors, PIT_RULE, PIT_RULE_SOURCE)
    result.validate(mine)
    return result


def order_selection(ids, precedences: Precedences, extracted=frozenset()) -> tuple[int, ...]:
    """Construct a witness order for an unordered set, NOT an order supplied by a source.

    `extracted` holds blocks already removed earlier (e.g. in previous periods).
    """
    remaining, ordered = set(ids), []
    while remaining:
        ready = sorted(i for i in remaining if precedences.predecessors[i] <= set(ordered) | set(extracted))
        if not ready:
            raise ValueError("Selección no cerrada bajo precedencias (o ciclo)")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return tuple(ordered)
