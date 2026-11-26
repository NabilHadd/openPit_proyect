"""Beam search over the cone view of the mine environment: direct search, no learning.

A node is a partial plan: its state, its extractions (block, period) and its VAN g. Level 0
is the empty plan. Expanding a node applies every action of MineEnv.cone_actions: extract
the pending cone of a block (the whole cone at once, so every plan is feasible) or advance
the period; STOP adds nothing (the node itself is already a plan). Children with the same
search_key (same extracted blocks, period and capacity used: the same future) are merged,
keeping the one of highest g. The `width` children of highest heuristic form the next level.
The search ends when a level is empty (no node has actions left) or when the safety cap
max_simulator_calls is reached. predict delivers the best plan seen at ANY level, not only
the last one; the empty plan (VAN 0) is the starting best, so a plan worse than extracting
nothing is never delivered.

Heuristics (how children are ranked; the delivered plan is always ranked by its real VAN):
- npv (default): g, the VAN of the partial plan. Because an action extracts a whole cone,
  its cost in waste is already inside g: a deep ore body behind waste is credited as soon
  as its cone is net positive.
- optimistic: g + the positive value still in the ground (pending reachable blocks of
  positive value, discounted at the current period): an upper bound of every completion,
  ignoring capacity. It does not punish a node for not having mined ore yet, only for the
  waste it took out; it ranks first the plans that keep the most potential value.

Deterministic: no RNG; ties are broken by order of generation (nodes of the previous level
in rank order, actions in the environment's fixed block order), so random_state is accepted
only for the common contract and `deterministic = True` (the experiments mode runs one seed).
Cost: cost_unit = simulator_calls (calls of extract/cone_step); expansions counts nodes expanded.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from sklearn.utils.validation import check_is_fitted

from configs import integer
from src.data.instance import MineInstance
from src.envs.mine_env import STOP, MineEnv, State
from src.evaluation.metrics import Extraction, Plan, checked_evaluation, same_value
from src.evaluation.reporter import write_json, write_plan_csv
from src.models.base import MineEstimator
from src.models.register import register
from src.training.rewards import discounted_value

HEURISTICS = ("npv", "optimistic")


@dataclass(frozen=True)
class BeamSearchParams:
    """width: nodes kept per level (B >= 1; B = 1 is the greedy search). heuristic: how
    children are ranked (npv | optimistic). max_simulator_calls: safety cap of the budget."""
    width: int = 10
    heuristic: str = "npv"
    max_simulator_calls: int = 2_000_000

    FIXED: ClassVar[tuple[str, ...]] = ("max_simulator_calls",)
    SMOKE: ClassVar[dict] = {"width": 2}

    def __post_init__(self):
        integer(self.width, "width", 1)
        if self.heuristic not in HEURISTICS:
            raise ValueError(f"heuristic debe ser una de {list(HEURISTICS)}")
        integer(self.max_simulator_calls, "max_simulator_calls", 1)


@dataclass(frozen=True)
class Node:
    state: State
    extractions: tuple[Extraction, ...]
    value: float   # g: VAN of the partial plan


@register("beam_search", params=BeamSearchParams)
class BeamSearchEstimator(MineEstimator):
    cost_unit = "simulator_calls"
    deterministic = True

    def __init__(self, width=10, heuristic="npv", max_simulator_calls=2_000_000, random_state=0):
        self.width = width
        self.heuristic = heuristic
        self.max_simulator_calls = max_simulator_calls
        self.random_state = random_state

    def fit(self, instance: MineInstance) -> "BeamSearchEstimator":
        self._check_params()
        env = MineEnv(instance.mine, instance.problem, instance.precedences, instance.environment)
        positive = {b.id: b.value for b in instance.mine.blocks if b.reachable and b.value > 0}
        rate = instance.problem.discount_rate

        def score(node: Node) -> float:
            if self.heuristic == "npv":
                return node.value
            period = node.state.period
            return node.value + sum(discounted_value(v, period, rate) for i, v in positive.items()
                                    if i not in node.state.extracted)

        self._start_trace()
        expansions, levels, capped = 0, [], False
        best = Node(env.initial_state(), (), 0.0)
        best_level = 0
        self._offer({"simulator_calls": 0, "expansions": 0}, 0.0)
        level = [best]
        while level and not capped:
            children: dict = {}  # search_key -> best child; insertion order = order of generation
            for node in level:
                expansions += 1
                for action in env.cone_actions(node.state):
                    if action == STOP:
                        continue
                    if env.calls >= self.max_simulator_calls:
                        capped = True
                        break
                    step = env.cone_step(node.state, action)
                    child = Node(step.state, node.extractions + tuple(Extraction(i, step.state.period)
                                                                      for i in step.extracted),
                                 node.value + step.reward)
                    if child.value > best.value and not same_value(child.value, best.value):
                        best, best_level = child, len(levels) + 1
                        self._offer({"simulator_calls": env.calls, "expansions": expansions}, child.value)
                    if step.terminal:
                        continue
                    key = env.search_key(step.state)
                    kept = children.get(key)
                    if kept is None or child.value > kept.value and not same_value(child.value, kept.value):
                        children[key] = child
                if capped:
                    break
            ranked = sorted(children.values(), key=score, reverse=True)  # stable: ties keep generation order
            level = ranked[:self.width]
            levels.append({"depth": len(levels) + 1, "children": len(children), "kept": len(level),
                           "best_value": max((n.value for n in level), default=None)})

        plan = Plan(best.extractions)
        evaluation = checked_evaluation(instance, plan)
        if not same_value(evaluation.npv, best.value):
            raise RuntimeError("El mejor plan de la búsqueda no coincide con su VAN verificado")
        self.instance_ = instance
        self.plan_ = plan
        self.evaluation_ = evaluation
        self.levels_ = tuple(levels)
        self.best_level_ = best_level
        self.capped_ = capped
        self.trace_ = tuple(self._trace)
        self.counters_ = {"simulator_calls": env.calls, "expansions": expansions}
        return self

    def predict(self, instance: MineInstance) -> Plan:
        """The best plan seen during the search; it belongs to the instance of fit."""
        check_is_fitted(self, "plan_")
        self._check_instance(instance)
        return self.plan_

    def diagnostics(self) -> dict:
        check_is_fitted(self, "plan_")
        return {"search": {"levels": len(self.levels_), "best_level": self.best_level_,
                           "capped": self.capped_, "blocks": len(self.plan_.extractions),
                           "npv": self.evaluation_.npv, **self.counters_}}

    def save_artifacts(self, folder: Path) -> None:
        """plan.csv (the delivered plan) and search.json (size and best value of every level)."""
        check_is_fitted(self, "plan_")
        write_plan_csv(Path(folder) / "plan.csv", self.plan_)
        write_json(Path(folder) / "search.json", {"levels": list(self.levels_), "best_level": self.best_level_,
                                                  "capped": self.capped_, "counters": self.counters_})
