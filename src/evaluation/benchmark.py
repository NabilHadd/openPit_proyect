"""Exact maximum closure for the one-period, undiscounted, uncapacitated case.

Independent diagnostic only: no reference selection and no training imports.
Decimal representations of input values are converted to rational capacities.
"""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from src.config import ProblemParameters
from src.data.dataset import Mine
from src.data.transforms import Precedences, order_selection
from src.evaluation.metrics import Evaluation, Extraction, Plan, evaluate_plan


@dataclass(frozen=True)
class ClosureResult:
    plan: Plan
    evaluation: Evaluation
    exact_value: str
    exact_cut_capacity: str


def maximum_closure(mine: Mine, precedences: Precedences,
                    parameters: ProblemParameters = ProblemParameters()) -> ClosureResult:
    if parameters != ProblemParameters():
        raise ValueError("Máximo cierre solo admite un período sin capacidad ni descuento")
    precedences.validate(mine)
    ids = sorted(mine.by_id)
    indices = {block_id: i for i, block_id in enumerate(ids)}
    source, sink = len(ids), len(ids) + 1
    residual = [{} for _ in range(len(ids) + 2)]
    values = {b.id: Fraction(str(b.value)) for b in mine.blocks}
    positive = sum((v for v in values.values() if v > 0), Fraction(0))
    infinity = positive + 1

    def add_edge(u, v, capacity):
        residual[u][v] = residual[u].get(v, Fraction(0)) + capacity
        residual[v].setdefault(u, Fraction(0))

    for block_id in ids:
        i, value = indices[block_id], values[block_id]
        if value > 0:
            add_edge(source, i, value)
        elif value < 0:
            add_edge(i, sink, -value)
        for predecessor in sorted(precedences.predecessors[block_id]):
            add_edge(i, indices[predecessor], infinity)

    flow = Fraction(0)
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v, capacity in residual[u].items():
                if capacity > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            reachable = set(parent)
            break
        path, v = [], sink
        while v != source:
            u = parent[v]
            path.append((u, v))
            v = u
        increment = min(residual[u][v] for u, v in path)
        for u, v in path:
            residual[u][v] -= increment
            residual[v][u] += increment
        flow += increment

    selected = {block_id for block_id in ids if indices[block_id] in reachable}
    plan = Plan(tuple(Extraction(i) for i in order_selection(selected, precedences)))
    evaluation = evaluate_plan(mine, plan, parameters, precedences)
    objective = sum((values[i] for i in selected), Fraction(0))
    if evaluation.feasible is not True or objective != positive - flow:
        raise RuntimeError("El certificado de máximo cierre no coincide con el plan verificado")
    return ClosureResult(plan, evaluation, str(objective), str(flow))
