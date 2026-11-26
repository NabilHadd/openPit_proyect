"""Shared loop of every mode that trains a model: one fit per seed, with its time, score,
cost, diagnostics and artifacts, plus the comparison with the reference once every seed is done.

Nothing here knows a particular model: it only uses the MineEstimator contract (including
counters_, trace_ and cost_unit). train uses it now; tuning and experiments reuse it.
"""

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from scipy.stats import t as student_t  # scipy is a dependency of scikit-learn
from sklearn.base import clone

from src.data.instance import MineInstance
from src.evaluation.metrics import same_value
from src.models.base import MineEstimator


@dataclass(frozen=True)
class SeedRun:
    seed: int
    estimator: MineEstimator  # fitted with random_state=seed
    seconds: float            # time of fit
    score: float              # verified VAN of the delivered plan
    blocks: int               # blocks of the delivered plan
    artifacts: str | None     # folder of save_artifacts, relative to the run directory


def hyperparameters(estimator: MineEstimator) -> dict:
    """get_params without random_state, which every seed sets on its own."""
    return {name: value for name, value in estimator.get_params().items() if name != "random_state"}


def fit_seeds(instance: MineInstance, estimator: MineEstimator, seeds, directory: Path | None = None) -> list[SeedRun]:
    """One fit per seed on a clone of estimator, which stays unfitted. With directory, each
    seed's artifacts go to directory/seed-<n>/."""
    runs = []
    for seed in seeds:
        fitted = clone(estimator).set_params(random_state=seed)
        started = perf_counter()
        fitted.fit(instance)
        seconds = perf_counter() - started
        plan = fitted.predict(instance)
        artifacts = None
        if directory is not None:
            folder = Path(directory) / f"seed-{seed}"
            folder.mkdir()
            fitted.save_artifacts(folder)
            artifacts = folder.name
        runs.append(SeedRun(seed, fitted, seconds, fitted.score(instance), len(plan.extractions), artifacts))
    return runs


def _is_counts(values, unit: str) -> bool:
    return isinstance(values, dict) and unit in values and all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0 for k, v in values.items())


def cost(estimator: MineEstimator, score: float) -> dict:
    """What fit spent (every counter) and the counters at the moment the delivered plan was
    first found; cost_unit names the counter the model's budget is measured in.

    Checks the contract of counters_ and trace_: every snapshot holds the unit, the cost in
    that unit never decreases, the VAN always improves and the last one is the score."""
    name, unit = type(estimator).__name__, estimator.cost_unit
    counters, trace = estimator.counters_, estimator.trace_
    if not _is_counts(counters, unit):
        raise RuntimeError(f"{name}.counters_ debe ser un diccionario de conteos que incluya cost_unit={unit!r}")
    if not trace or not all(isinstance(entry, tuple) and len(entry) == 2 and _is_counts(entry[0], unit)
                            for entry in trace):
        raise RuntimeError(f"{name}.trace_ debe ser una tupla no vacía de (contadores, VAN)")
    costs, values = [entry[0][unit] for entry in trace], [entry[1] for entry in trace]
    if costs != sorted(costs) or costs[-1] > counters[unit] or any(
            b <= a or same_value(a, b) for a, b in zip(values, values[1:])):
        raise RuntimeError(f"{name}.trace_: el costo no puede bajar y el VAN debe mejorar en cada entrada")
    if not same_value(values[-1], score):
        raise RuntimeError(f"{name}.trace_ termina en {values[-1]} pero el plan entregado vale {score}")
    return {"unit": unit, "spent": dict(counters), "to_best": dict(trace[-1][0])}


def seed_row(run: SeedRun) -> dict:
    """Common figures of one seed for report.json; what is model-specific goes in diagnostics."""
    return {"seed": run.seed, "seconds": run.seconds, "score": run.score, "blocks": run.blocks,
            "cost": cost(run.estimator, run.score), "artifacts": run.artifacts,
            "diagnostics": run.estimator.diagnostics()}


def mean_interval(values) -> dict:
    """Mean, sample standard deviation and 95% confidence interval of the mean (Student's t).
    With a single value the deviation and the interval are null; with none, everything is."""
    values = list(values)
    if not values:
        return {"mean": None, "std": None, "ci95": None}
    center = mean(values)
    if len(values) < 2:
        return {"mean": center, "std": None, "ci95": None}
    spread = stdev(values)
    half = float(student_t.ppf(0.975, len(values) - 1)) * spread / math.sqrt(len(values))
    return {"mean": center, "std": spread, "ci95": [center - half, center + half]}


def seeds_summary(runs: list[SeedRun]) -> dict:
    """Score over seeds and cost in the model's unit (cost_unit)."""
    scores = [run.score for run in runs]
    interval = mean_interval(scores)
    units = {run.estimator.cost_unit for run in runs}
    if len(units) != 1:
        raise RuntimeError(f"Las semillas de un mismo modelo deben medir su costo en la misma unidad: {sorted(units)}")
    unit = units.pop()
    costs = [cost(run.estimator, run.score) for run in runs]
    to_best = mean_interval(c["to_best"][unit] for c in costs)
    return {"seeds": len(runs), "score_max": max(scores), "score_mean": interval["mean"], "score_min": min(scores),
            "score_std": interval["std"], "score_ci95": interval["ci95"], "cost_unit": unit,
            "cost_spent_mean": mean(c["spent"][unit] for c in costs),
            "cost_to_best_mean": to_best["mean"], "cost_to_best_std": to_best["std"]}


def gap_percent(score: float, reference_npv: float) -> float | None:
    """100 * (reference - score) / |reference|: 0 = matches it, > 0 = below it. Null when the
    reference is 0 (no relative gap exists)."""
    if reference_npv == 0:
        return None
    return 100 * (reference_npv - score) / abs(reference_npv)


def cost_to_reach(trace, target: float) -> dict | None:
    """Counters at the first improvement whose VAN reached target (None if never)."""
    for counters, npv in trace:
        if npv > target or same_value(npv, target):
            return dict(counters)
    return None


def _counts(counts, model: str) -> dict[str, int]:
    if not isinstance(counts, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for key, value in counts.items()):
        raise RuntimeError(f"{model}.compare_reference debe devolver un diccionario de conteos (enteros >= 0)")
    return counts


def compare_reference(runs: list[SeedRun], rows: list[dict], reference_npv: float) -> dict:
    """Only after every seed has been trained and saved, so the reference never reaches
    training: adds to each row whether its plan matches the reference, its gap, the cost at
    which it first reached the reference (from trace_) and the model's own counts, and
    returns their totals and means."""
    totals = Counter()
    for run, row in zip(runs, rows, strict=True):
        counts = _counts(run.estimator.compare_reference(reference_npv), type(run.estimator).__name__)
        row["matches_reference"] = same_value(run.score, reference_npv)
        row["gap_percent"] = 0.0 if row["matches_reference"] and reference_npv != 0 else gap_percent(run.score, reference_npv)
        row["cost_to_reference"] = cost_to_reach(run.estimator.trace_, reference_npv)
        row["reference"] = counts
        totals.update(counts)
    gaps = [row["gap_percent"] for row in rows]
    by_seed = [None if row["cost_to_reference"] is None else row["cost_to_reference"][run.estimator.cost_unit]
               for run, row in zip(runs, rows)]
    reached = mean_interval(v for v in by_seed if v is not None)
    return {"reference_npv": reference_npv, "matches_reference": sum(row["matches_reference"] for row in rows),
            "gap_percent_mean": None if None in gaps else mean(gaps),
            "reached_reference": sum(v is not None for v in by_seed),
            # mean and std only over the seeds that reached it (see reached_reference); per seed, in cost_unit
            "cost_to_reference_mean": reached["mean"], "cost_to_reference_std": reached["std"],
            "cost_to_reference_by_seed": by_seed, "reference": dict(totals)}
