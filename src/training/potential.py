"""Shared-waste potential: an estimate of the value still obtainable from a state.

Every ore block (reachable, value > 0) pays for the waste of its cone, but a waste block
that lies in the cones of several ore blocks is paid once in reality, so its cost is split
in equal parts among them:

    share(w)  = number of ore blocks whose cone contains the waste block w
    cost(o,s) = sum of |value(w)| / share(w) over the waste w of cone(o) still in the ground
    Phi(s)    = sum over ore o still in the ground of max(0, value(o) - cost(o, s)),
                discounted at the current period

Extracting the waste over an ore body raises Phi (that ore is now cheaper) and extracting
the ore lowers Phi by what it was worth, so Phi points to the ore worth digging without
solving the problem: it ignores capacity and periods, and the split is a rule of thumb,
not an optimum. Phi is never above the positive value still in the ground.

Used by the tabular models as the initial value of Q (initial_q: shared_waste, equivalent
to potential-based reward shaping, Wiewiora 2003) and by DQN as a feature.
"""

from src.training.rewards import discounted_value


class SharedWastePotential:
    def __init__(self, blocks: dict, cones: dict, discount_rate: float):
        """blocks: id -> Block; cones: id -> frozenset of the block and its predecessors."""
        ore = sorted(i for i, b in blocks.items() if b.reachable and b.value > 0)
        share: dict[int, int] = {}
        for o in ore:
            for w in cones[o]:
                if blocks[w].value < 0:
                    share[w] = share.get(w, 0) + 1
        self.discount_rate = discount_rate
        # (ore id, ore value, ((waste id, its part of the waste cost), ...)), in id order
        self.terms = tuple((o, blocks[o].value,
                            tuple((w, -blocks[w].value / share[w]) for w in sorted(cones[o]) if blocks[w].value < 0))
                           for o in ore)
        self.share = share
        self._cache: dict[frozenset, float] = {}

    def undiscounted(self, extracted: frozenset) -> float:
        value = self._cache.get(extracted)
        if value is None:
            value = 0.0
            for o, ore_value, wastes in self.terms:
                if o in extracted:
                    continue
                net = ore_value - sum(cost for w, cost in wastes if w not in extracted)
                if net > 0:
                    value += net
            self._cache[extracted] = value
        return value

    def __call__(self, state) -> float:
        """Phi of a State (its extracted blocks and period)."""
        return discounted_value(self.undiscounted(state.extracted), state.period, self.discount_rate)
