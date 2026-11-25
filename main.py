"""Entry point. Every value lives in the YAML given by --config (and the model YAMLs it
lists); the terminal only carries the mode, boolean flags and, for evaluate-plan, the plan
to evaluate.

    python main.py train --config configs/config.yml
    python main.py evaluate-plan --config configs/config.yml [--plan RUTA]
"""

import argparse
from pathlib import Path

from configs import project_path
from configs.config_settings import DEFAULT_CONFIG, RunConfig, load_config
from src.data.dataset import Solution, load_mine, load_solution
from src.data.instance import MineInstance
from src.data.precedences import order_selection, pit_precedences
from src.evaluation.metrics import Extraction, Plan, evaluate_plan
from src.evaluation.reporter import code_provenance, create_run_directory, mine_metadata, runtime_metadata, write_json, write_plan_csv
from src.models.registry import build_estimator, load_models
from src.runner import compare_reference, fit_seeds, hyperparameters, seed_row, seeds_summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Planificación minera de una mina a cielo abierto")
    modes = result.add_subparsers(dest="mode", required=True)
    for name, text in (("train", "Entrenar el modelo del YAML una vez por semilla y exportar sus resultados"),
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
    return {"schema_version": 4, "mode": mode, "config": config.as_dict(), "code": code_provenance(),
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


def run_train(config: RunConfig) -> Path:
    settings = load_models(config)  # every model YAML, before loading data or training
    estimator = build_estimator(config.model, settings[config.model])
    instance = load_inputs(config)
    reference_file = solution_path(config)
    directory = create_run_directory(project_path(config.output), "train")
    report = base_report("train", config, instance)
    report["model"] = {"name": config.model, "file": str(project_path(config.models[config.model])),
                       "params": hyperparameters(estimator)}
    runs = fit_seeds(instance, estimator, config.seeds, directory)
    rows = [seed_row(run) for run in runs]
    summary = seeds_summary(runs)

    # The reference is read only after every seed has been trained and saved.
    reference = None
    if reference_file is not None:
        reference, _ = plan_report(instance, load_solution(reference_file, instance.mine))
        if reference["feasible"]:
            summary.update(compare_reference(runs, rows, reference["npv"]))
    report.update(reference=reference, seeds=rows, summary=summary)
    write_json(directory / "report.json", report)

    print(f"Reporte: {directory / 'report.json'}")
    for row in rows:
        print(f"Semilla {row['seed']}: VAN del plan entregado={row['score']:g} ({row['blocks']} bloques)")
    if "reference_npv" in summary:
        counts = "".join(f"; {name}={total}" for name, total in summary["reference"].items())
        print(f"Referencia {summary['reference_npv']:g}: el plan entregado la iguala en "
              f"{summary['matches_reference']}/{len(rows)} semillas{counts}")
    elif reference is not None:
        print("La solución de referencia no es factible; no se compara")
    else:
        print("Sin solución de referencia (solution: null)")
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
    directory = create_run_directory(project_path(config.output), "evaluate-plan")
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
