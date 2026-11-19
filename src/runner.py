"""Shared loop of every mode that trains a model: one fit per seed, with its time, score,
diagnostics and artifacts, plus the comparison with the reference once every seed is done.

Nothing here knows a particular model: it only uses the MineEstimator contract. train uses
it now; tuning and experiments reuse it.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

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


def seed_row(run: SeedRun) -> dict:
    """Common figures of one seed for report.json; what is model-specific goes in diagnostics."""
    return {"seed": run.seed, "seconds": run.seconds, "score": run.score, "blocks": run.blocks,
            "artifacts": run.artifacts, "diagnostics": run.estimator.diagnostics()}


def seeds_summary(runs: list[SeedRun]) -> dict:
    scores = [run.score for run in runs]
    return {"seeds": len(runs), "score_max": max(scores), "score_mean": mean(scores), "score_min": min(scores)}


def _counts(counts, model: str) -> dict[str, int]:
    if not isinstance(counts, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for key, value in counts.items()):
        raise RuntimeError(f"{model}.compare_reference debe devolver un diccionario de conteos (enteros >= 0)")
    return counts


def compare_reference(runs: list[SeedRun], rows: list[dict], reference_npv: float) -> dict:
    """Only after every seed has been trained and saved: adds to each row whether its plan
    matches the reference and the model's own counts, and returns their totals."""
    totals = Counter()
    for run, row in zip(runs, rows, strict=True):
        counts = _counts(run.estimator.compare_reference(reference_npv), type(run.estimator).__name__)
        row["matches_reference"] = same_value(run.score, reference_npv)
        row["reference"] = counts
        totals.update(counts)
    return {"reference_npv": reference_npv, "matches_reference": sum(row["matches_reference"] for row in rows),
            "reference": dict(totals)}
