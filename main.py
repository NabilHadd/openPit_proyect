"""Entry point. Every value lives in the YAML given by --config (and the model YAMLs found
by convention, configs/<key>.yml); the terminal only carries the mode, boolean flags and,
for evaluate-plan, the plan to evaluate.

    python main.py train --config configs/config.yml          # the model of `train`
    python main.py experiments --config configs/config.yml    # every model of `experiments`
    python main.py evaluate-plan --config configs/config.yml [--plan RUTA]

Output folders (local time; _2, _3... if the name exists; `name` of the YAML at the end):
    <output>/train/<YYYY-MM-DD_HH-MM>_train_<model>[_name]/{config.yml, <model>.yml, report.json, seed-N/}
    <output>/experiments/<YYYY-MM-DD_HH-MM>_experiments[_name]/{config.yml, summary.csv, summary.json,
        <model>/{<model>.yml, report.json, seed-N/}}
    <output>/evaluate-plan/<YYYY-MM-DD_HH-MM>_evaluate-plan[_name]/{config.yml, report.json, plan.csv}
"""

import argparse
from pathlib import Path
import shutil
from time import perf_counter
import traceback

from configs import project_path
from configs.config_settings import DEFAULT_CONFIG, RunConfig, load_config
from src.data.dataset import Solution, load_mine, load_solution
from src.data.instance import MineInstance
from src.data.precedences import order_selection, pit_precedences
from src.evaluation.metrics import Extraction, Plan, evaluate_plan
from src.evaluation.reporter import (code_provenance, create_run_directory, mine_metadata, runtime_metadata, write_json,
                                     write_plan_csv, write_rows_csv)
from src.models.base import MineEstimator
from src.models.register import build_estimator, load_models
from src.runner import compare_reference, fit_seeds, hyperparameters, seed_row, seeds_summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Planificación minera de una mina a cielo abierto")
    modes = result.add_subparsers(dest="mode", required=True)
    for name, text in (("train", "Entrenar el modelo de train una vez por semilla y exportar sus resultados"),
                       ("experiments", "Entrenar cada modelo de experiments con las mismas semillas y resumirlos"),
                       ("evaluate-plan", "Calcular el VAN y verificar la factibilidad de un plan")):
        mode = modes.add_parser(name, help=text, description=text)
        mode.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML de configuración; default={DEFAULT_CONFIG}")
        if name == "evaluate-plan":
            mode.add_argument("--plan", help="CSV del plan (order,id,period); sin --plan se evalúa la solución del YAML")
    return result


def file_metadata(solution: Solution) -> dict:
    return {"path": solution.source, "sha256": solution.data_sha256, "ordered": solution.ordered,
            "blocks": len(solution.entries)}


def plan_report(instance: MineInstance, solution: Solution) -> tuple[dict, Plan]:
    """Evaluate a plan file. If it has no order, a valid witness order is built period by
    period, only to verify it."""
    order_error = None
    if solution.ordered:
        extractions = tuple(Extraction(e.block_id, e.period) for e in solution.entries)
    else:
        extractions, done = [], set()
        for period in sorted({e.period for e in solution.entries}):
            ids = sorted(e.block_id for e in solution.entries if e.period == period)
            if order_error is None:
                try:
                    ids = order_selection(ids, instance.precedences, done)
                except ValueError as exc:
                    order_error = str(exc)
            extractions.extend(Extraction(i, period) for i in ids)
            done.update(ids)
        extractions = tuple(extractions)
    plan = Plan(extractions)
    evaluation = evaluate_plan(instance.mine, plan, instance.problem, instance.precedences)
    return {"file": file_metadata(solution), "feasible": evaluation.feasible, "npv": evaluation.npv,
            "raw_value": evaluation.raw_value, "period_usage": evaluation.period_usage,
            "errors": evaluation.errors[:20], "order": "del archivo" if solution.ordered else
            ("testigo construido para verificar" if order_error is None else f"sin orden válido: {order_error}")}, plan


def base_report(mode: str, config: RunConfig, instance: MineInstance) -> dict:
    return {"schema_version": 5, "mode": mode, "config": config.as_dict(), "code": code_provenance(),
            "runtime": runtime_metadata(), "data": mine_metadata(instance.mine),
            "precedence_rule": instance.precedences.rule}


def load_inputs(config: RunConfig) -> MineInstance:
    mine = load_mine(project_path(config.data))
    return MineInstance(mine, pit_precedences(mine), config.problem, config.environment)


def solution_path(config: RunConfig) -> Path | None:
    if config.solution is None:
        return None
    path = project_path(config.solution)
    if not path.is_file():
        raise ValueError(f"No existe la solución {path}; corrija solution en el YAML o use solution: null")
    return path


def copy_config(config: RunConfig, directory: Path):
    """Exact copy of the configuration file the run was launched with."""
    if config.source is not None:
        shutil.copyfile(config.source, directory / "config.yml")


def model_report(mode: str, config: RunConfig, instance: MineInstance, key: str, estimator: MineEstimator,
                 seeds, directory: Path, reference_file: Path | None) -> dict:
    """Train one model once per seed into directory, compare it with the reference and write
    directory/report.json. The reference is read only after every seed has been trained and
    saved; it never reaches the model."""
    shutil.copyfile(config.model_file(key), directory / f"{key}.yml")
    report = base_report(mode, config, instance)
    report["model"] = {"name": key, "file": str(config.model_file(key)), "params": hyperparameters(estimator),
                       "deterministic": type(estimator).deterministic, "cost_unit": type(estimator).cost_unit}
    runs = fit_seeds(instance, estimator, seeds, directory)
    rows = [seed_row(run) for run in runs]
    summary = seeds_summary(runs)
    reference = None
    if reference_file is not None:
        reference, _ = plan_report(instance, load_solution(reference_file, instance.mine))
        if reference["feasible"]:
            summary.update(compare_reference(runs, rows, reference["npv"]))
    report.update(status="ok", reference=reference, seeds=rows, summary=summary)
    write_json(directory / "report.json", report)
    return report


def print_model(report: dict):
    rows, summary = report["seeds"], report["summary"]
    for row in rows:
        print(f"  Semilla {row['seed']}: VAN del plan entregado={row['score']:g} ({row['blocks']} bloques)")
    if "reference_npv" in summary:
        counts = "".join(f"; {name}={total}" for name, total in summary["reference"].items())
        gap = summary["gap_percent_mean"]
        print(f"  Referencia {summary['reference_npv']:g}: el plan entregado la iguala en "
              f"{summary['matches_reference']}/{len(rows)} semillas{counts}"
              + ("" if gap is None else f"; brecha media {gap:.2f} %"))
    elif report["reference"] is not None:
        print("  La solución de referencia no es factible; no se compara")
    else:
        print("  Sin solución de referencia (solution: null)")


def run_train(config: RunConfig) -> Path:
    if config.train is None:
        raise ValueError("train requiere el campo train en la configuración (un modelo)")
    key = config.train
    settings = load_models(config, [key])[key]  # its YAML, before loading data or training
    estimator = build_estimator(key, settings)
    instance = load_inputs(config)
    reference_file = solution_path(config)
    directory = create_run_directory(project_path(config.output) / "train", f"train_{key}", config.name)
    copy_config(config, directory)
    report = model_report("train", config, instance, key, estimator, config.seeds, directory, reference_file)
    print(f"Reporte: {directory / 'report.json'}")
    print_model(report)
    return directory


SUMMARY_COLUMNS = ("model", "status", "seeds", "score_mean", "score_std", "ci95_low", "ci95_high",
                   "gap_percent_mean", "reached_reference", "cost_unit", "cost_to_reference_mean",
                   "cost_to_reference_std", "cost_to_best_mean", "cost_to_best_std", "cost_spent_mean",
                   "seconds", "params", "error")


def summary_entry(key: str, estimator: MineEstimator, seeds, seconds: float, report: dict | None,
                  error: dict | None) -> dict:
    """One model of summary.json; summary_row flattens it for summary.csv."""
    entry = {"model": key, "status": "ok" if error is None else "error", "seeds": list(seeds),
             "deterministic": type(estimator).deterministic, "cost_unit": type(estimator).cost_unit,
             "params": hyperparameters(estimator), "seconds": seconds, "error": error}
    if report is not None:
        entry["summary"] = report["summary"]
        entry["reference_npv"] = report["summary"].get("reference_npv")
    return entry


def summary_row(entry: dict) -> dict:
    summary = entry.get("summary", {})
    ci = summary.get("score_ci95") or (None, None)
    n = len(entry["seeds"]) if entry["status"] == "ok" else None
    reached = summary.get("reached_reference")
    return {"model": entry["model"], "status": entry["status"], "seeds": n,
            "score_mean": summary.get("score_mean"), "score_std": summary.get("score_std"),
            "ci95_low": ci[0], "ci95_high": ci[1], "gap_percent_mean": summary.get("gap_percent_mean"),
            "reached_reference": None if reached is None else f"{reached}/{n}", "cost_unit": entry["cost_unit"],
            "cost_to_reference_mean": summary.get("cost_to_reference_mean"),
            "cost_to_reference_std": summary.get("cost_to_reference_std"),
            "cost_to_best_mean": summary.get("cost_to_best_mean"), "cost_to_best_std": summary.get("cost_to_best_std"),
            "cost_spent_mean": summary.get("cost_spent_mean"), "seconds": entry["seconds"],
            "params": "; ".join(f"{k}={v}" for k, v in entry["params"].items()),
            "error": None if entry["error"] is None else f"{entry['error']['type']}: {entry['error']['message']}"}


def run_experiments(config: RunConfig) -> Path:
    """Every model of experiments, with the seeds of the YAML (a deterministic model only with
    the first one). Every model YAML is read and validated before anything is trained; a model
    that fails while training is recorded in the summary and the next one goes on."""
    if not config.experiments:
        raise ValueError("experiments requiere una lista no vacía de modelos en la configuración")
    settings = load_models(config, config.experiments)
    estimators = {key: build_estimator(key, settings[key]) for key in config.experiments}
    instance = load_inputs(config)
    reference_file = solution_path(config)
    directory = create_run_directory(project_path(config.output) / "experiments", "experiments", config.name)
    copy_config(config, directory)
    entries = []
    for key, estimator in estimators.items():
        seeds = config.seeds[:1] if type(estimator).deterministic else config.seeds
        folder = directory / key
        folder.mkdir()
        print(f"Modelo {key} ({len(seeds)} semilla{'s' if len(seeds) != 1 else ''})")
        started, report, error = perf_counter(), None, None
        try:
            report = model_report("experiments", config, instance, key, estimator, seeds, folder, reference_file)
        except Exception as exc:  # recorded, and the next model goes on
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            failed = base_report("experiments", config, instance)
            failed.update(model={"name": key, "params": hyperparameters(estimator)}, status="error", error=error)
            write_json(folder / "report.json", failed)
        entries.append(summary_entry(key, estimator, seeds, perf_counter() - started, report, error))
        write_json(directory / "summary.json", {"schema_version": 5, "mode": "experiments",
                                                "seeds": list(config.seeds), "models": entries})
        write_rows_csv(directory / "summary.csv", SUMMARY_COLUMNS, [summary_row(e) for e in entries])
        if error is None:
            print_model(report)
        else:
            print(f"  Error: {error['type']}: {error['message']}")
    print(f"Resumen: {directory / 'summary.csv'}")
    return directory


def run_evaluate_plan(config: RunConfig, plan: str | None = None) -> Path:
    instance = load_inputs(config)
    if plan is not None:
        plan_file = project_path(plan)
    else:
        plan_file = solution_path(config)
        if plan_file is None:
            raise ValueError("evaluate-plan sin --plan evalúa la solución del YAML, pero solution es null")
    evaluated, witness = plan_report(instance, load_solution(plan_file, instance.mine))
    directory = create_run_directory(project_path(config.output) / "evaluate-plan", "evaluate-plan", config.name)
    copy_config(config, directory)
    report = base_report("evaluate-plan", config, instance)
    report["plan"] = evaluated
    if evaluated["feasible"]:
        write_plan_csv(directory / "plan.csv", witness)  # ordered version of the evaluated plan
    write_json(directory / "report.json", report)
    print(f"Reporte: {directory / 'report.json'}")
    print(f"VAN recalculado: {evaluated['npv']:g}; factible={evaluated['feasible']}")
    for error in evaluated["errors"][:10]:
        print(f"  - {error}")
    return directory


def execute(args) -> Path:
    config = load_config(args.config)
    if args.mode == "train":
        return run_train(config)
    if args.mode == "experiments":
        return run_experiments(config)
    return run_evaluate_plan(config, args.plan)


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        execute(args)
    except (ValueError, OSError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
