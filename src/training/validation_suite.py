"""Sequential validation orchestration with replaceable training, benchmark and storage."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from src.config import LearningParameters, SpatialParameters
from src.data.dataset import Mine
from src.data.transforms import Precedences
from src.evaluation.benchmark import maximum_closure
from src.evaluation.tiny_oracle import TinyOracleResult, enumerate_tiny
from src.evaluation.validation_artifacts import FileValidationArtifacts
from src.evaluation.validation_report import attach_benchmark_metrics, summarize_run, summarize_suite
from src.training.validation_runner import ValidationBudget, ValidationRun, train_transition_budget


class ValidationArtifacts(Protocol):
    """Stores snapshots synchronously without retaining runs or the mutable report."""

    def start_suite(self, mine, rules, spatial, learning, seeds, budget, output: Path) -> tuple[Path, dict]: ...

    def save_run(self, directory: Path, mine, rules, run: ValidationRun) -> tuple[str, dict[str, str]]: ...

    def write_report(self, directory: Path, report: dict) -> None: ...


Trainer = Callable[[Mine, Precedences, SpatialParameters, LearningParameters, int, ValidationBudget], ValidationRun]
Benchmark = Callable[[Mine, Precedences, SpatialParameters], tuple[float, dict, TinyOracleResult | None]]


def evaluate_validation_benchmark(mine: Mine, rules: Precedences, spatial: SpatialParameters) -> tuple[float, dict, TinyOracleResult | None]:
    if len(mine.blocks) <= 6:
        oracle = enumerate_tiny(mine, rules, spatial)
        return oracle.optimum, {"kind": "independent-exhaustive-action-oracle", **asdict(oracle), "used_by_agent": False}, oracle
    closure = maximum_closure(mine, rules)
    target = closure.evaluation.npv
    return target, {"kind": "economic-upper-bound", "value": target,
                    "spatial_attainability": "not_certified_by_this_run", "used_by_agent": False}, None


def run_validation_suite(mine, rules, spatial, learning, seeds, budget, output: Path, *,
                         trainer: Trainer | None = None, benchmark: Benchmark | None = None,
                         store: ValidationArtifacts | None = None) -> Path:
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError("Se requieren semillas enteras distintas y una lista no vacía")
    if learning.gamma != 1 or not spatial.construction_up:
        raise ValueError("La validación requiere gamma=1 y seis direcciones en ambas fases")
    trainer = train_transition_budget if trainer is None else trainer
    benchmark = evaluate_validation_benchmark if benchmark is None else benchmark
    store = FileValidationArtifacts() if store is None else store
    directory, report = store.start_suite(mine, rules, spatial, learning, seeds, budget, output)
    try:
        for seed in seeds:
            run = trainer(mine, rules, spatial, learning, seed, budget)
            if run.seed != seed or run.training["transitions"] != budget.transitions:
                raise RuntimeError("El ejecutor cambió la semilla o el presupuesto de transiciones")
            folder, exports = store.save_run(directory, mine, rules, run)
            report["runs"].append(summarize_run(run, directory=folder, exports=exports))
            report["completed_runs"] += 1
            store.write_report(directory, report)
            del run
        if report["completed_runs"] != report["expected_runs"]:
            raise RuntimeError("Suite incompleta: no se calcula una tasa de éxito parcial")
        target, payload, oracle = benchmark(mine, rules, spatial)
        report["benchmark"] = payload
        for row in report["runs"]:
            attach_benchmark_metrics(row, target=target, oracle=oracle, finite_horizon=spatial.horizon_mode == "finite")
        summary = summarize_suite(report["runs"], evaluation_rollouts=budget.evaluation_rollouts,
                                  training_transitions=budget.transitions, tiny=len(mine.blocks) <= 6)
        report.update(status="completed", summary=summary)
        store.write_report(directory, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        store.write_report(directory, report)
        raise
    return directory
