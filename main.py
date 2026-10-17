"""Local reproducible entry point. No reference metadata is passed to training."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
from time import perf_counter

from src.config import LearningParameters, ProblemParameters, SpatialParameters, project_path
from src.data.dataset import load_mine
from src.data.transforms import (complete_cone_candidates, grid9_z_up, grid9_z_down,
                                 load_precedences, load_selections, order_selection)
from src.evaluation.benchmark import maximum_closure
from src.evaluation.metrics import Extraction, Plan, evaluate_plan
from src.evaluation.plots import plot_history
from src.evaluation.reporter import (code_provenance, create_run_directory, export_checked_plan,
                                     load_plan, mine_metadata, plan_payload, precedence_metadata,
                                     q_table_payload, runtime_metadata, write_json)
from src.models.q_learning import QLearning, SpatialQLearning
from src.training.experiment_runner import run_experiments


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Planificación minera experimental; réplica del paper pendiente")
    result.add_argument("mode", choices=("reference", "train", "evaluate-plan", "benchmark"))
    result.add_argument("--data", required=True, help="Ruta exacta del Excel; relativa al repositorio o absoluta")
    rules = result.add_mutually_exclusive_group()
    rules.add_argument("--precedence", choices=("grid9-z-up", "grid9-z-down"), help="Regla explícita; grid9-z-up corresponde a la superficie Z=3 confirmada del caso reducido")
    rules.add_argument("--precedences-file", help="JSON explícito con source y predecessors para todos los IDs")
    result.add_argument("--periods", type=int, default=1)
    result.add_argument("--discount-rate", type=float, default=0.0)
    result.add_argument("--capacity", type=float)
    result.add_argument("--capacity-unit", choices=("blocks", "tonnes"), default="blocks")
    result.add_argument("--episodes", type=int, default=100)
    result.add_argument("--seeds", type=int, nargs="+", default=[0])
    result.add_argument("--alpha", type=float, help="Default full-state=0.2; spatial-cone=1")
    result.add_argument("--gamma", type=float, help="Factor RL, distinto de discount-rate; default full-state=1, spatial-cone=0.7")
    result.add_argument("--epsilon", type=float, help="Default full-state=0.2; spatial-cone=0.3")
    result.add_argument("--model", choices=("full-state", "spatial-cone"), default="full-state")
    result.add_argument("--surface", choices=("min-z", "max-z"), help="Spatial: por defecto coincide con la regla geométrica; obligatorio con reglas JSON")
    result.add_argument("--max-steps", type=int, help="Horizonte espacial artificial; default=500")
    result.add_argument("--construction-up", action="store_true", help="Extensión: permitir subir también al construir; no la variante de cinco direcciones del paper")
    result.add_argument("--exact-benchmark", action="store_true", help="Solo train: máximo cierre DESPUÉS de entrenar; sin capacidad ni descuento")
    result.add_argument("--plan", help="Plan interno JSON para evaluate-plan")
    result.add_argument("--output", default="artifacts/reports")
    result.add_argument("--plot", action="store_true", help="Guardar curva SVG por semilla")
    return result


def reference_report(mine, selections, parameters, precedences) -> tuple[dict, Plan]:
    economic_value = sum(mine.by_id[i].value for i in selections.reference.ids)
    ids = selections.reference.ids
    order_error = None
    if precedences is not None:
        try:
            ids = order_selection(ids, precedences)
        except ValueError as exc:
            order_error = str(exc)
    plan = Plan(tuple(Extraction(i) for i in ids))
    # The visual is only a selection. A one-period witness is constructed explicitly.
    evaluation = evaluate_plan(mine, plan, parameters, precedences)
    geometry = {}
    if precedences is not None and precedences.rule in {"grid9-z-up", "grid9-z-down"}:
        surface = "max-z" if precedences.rule == "grid9-z-up" else "min-z"
        candidates = complete_cone_candidates(mine, surface)
        geometry = {"surface": surface, "complete_cone_candidate_ids": candidates,
                    "matches_excel_candidates": set(candidates) == set(selections.candidates.ids),
                    "used_to_restrict_agent": False,
                    "interpretation": "Coincidencia geométrica diagnóstica; significado de colores pendiente de Aldo"}
    return {"candidates": asdict(selections.candidates), "reference": asdict(selections.reference),
            "declared_candidates": selections.declared_candidates,
            "candidate_count_matches_declaration": len(selections.candidates.ids) == selections.declared_candidates,
            "recalculated_raw_value": economic_value, "declared_value": selections.declared_value,
            "economic_sum_matches": math.isclose(economic_value, selections.declared_value, rel_tol=1e-12, abs_tol=1e-6),
            "evaluation": asdict(evaluation), "witness_order_error": order_error,
            "candidate_geometry_diagnostic": geometry,
            "schedule_note": "Orden testigo construido, NO secuencia del Excel; período 0 sin capacidad ni descuento",
            "witness_plan": plan_payload(plan)}, plan


def execute(args) -> Path:
    data_path = project_path(args.data)
    problem = ProblemParameters(args.periods, args.discount_rate, args.capacity, args.capacity_unit)
    defaults = (1.0, 0.7, 0.3) if args.model == "spatial-cone" else (0.2, 1.0, 0.2)
    learning = LearningParameters(args.episodes, *(default if value is None else value
                                  for value, default in zip((args.alpha, args.gamma, args.epsilon), defaults)))
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Las semillas deben ser distintas")
    if args.mode in {"train", "benchmark"} and not (args.precedence or args.precedences_file):
        raise ValueError("Entrenar o calcular benchmark requiere precedencias explícitas; no se infieren del resultado")
    if args.mode == "reference" and problem != ProblemParameters():
        raise ValueError("La selección visual no define períodos/capacidad/descuento; use evaluate-plan con un plan explícito")
    if args.mode == "evaluate-plan" and not args.plan:
        raise ValueError("evaluate-plan requiere --plan")
    if (args.mode == "benchmark" or args.exact_benchmark) and problem != ProblemParameters():
        raise ValueError("Máximo cierre requiere un período sin capacidad ni descuento")
    if args.exact_benchmark and args.mode != "train":
        raise ValueError("--exact-benchmark solo se usa en train; también existe el modo benchmark")
    spatial = None
    if args.model == "spatial-cone":
        if args.mode != "train":
            raise ValueError("--model spatial-cone solo se usa en train")
        expected_surface = {"grid9-z-up": "max-z", "grid9-z-down": "min-z"}.get(args.precedence)
        surface = args.surface or expected_surface
        if surface is None:
            raise ValueError("Spatial con precedencias JSON requiere --surface explícita")
        if expected_surface is not None and surface != expected_surface:
            raise ValueError("La orientación de surface contradice la regla de precedencias")
        spatial = SpatialParameters(surface, 500 if args.max_steps is None else args.max_steps, args.construction_up)
    elif args.surface or args.max_steps is not None or args.construction_up:
        raise ValueError("Las opciones espaciales requieren --model spatial-cone")
    mine = load_mine(data_path)
    grid_rule = {"grid9-z-up": grid9_z_up, "grid9-z-down": grid9_z_down}.get(args.precedence)
    precedences = (grid_rule(mine) if grid_rule else
                   load_precedences(project_path(args.precedences_file), mine) if args.precedences_file else None)
    report = {"schema_version": 1, "mode": args.mode, "effective_config": {
        **vars(args), "data": str(data_path), "output": str(project_path(args.output)),
        "alpha": learning.alpha, "gamma": learning.gamma, "epsilon": learning.epsilon,
        "problem": asdict(problem), "learning": asdict(learning),
        "spatial": asdict(spatial) if spatial is not None else None},
        "code": code_provenance(), "runtime": runtime_metadata(), "data": mine_metadata(mine),
        "precedences": precedence_metadata(precedences),
        "model": (SpatialQLearning.name if spatial else QLearning.name) if args.mode == "train" else None,
        "paper": {"doi": "10.17159/2411-9717/1367/2021", "source": "PDF/texto aportado por usuario; pp. 647-650"},
        "domain_confirmations": {
            "scope": "Caso reducido 105 bloques; confirmaciones del usuario en conversación",
            "surface": "Z=3 en el Excel del caso reducido",
            "reward": "B: valor incremental del conjunto nuevo; no recompensa terminal",
            "revisits": "mover sin volver a extraer ni contabilizar",
            "selection": "conservar camino observado de mayor VAN; permite costos negativos intermedios",
            "still_pending": "estado Q exacto, bordes, movimientos por fase y cierre natural de episodio"},
        "paper_replication": "partial: incremental reward, revisits and Z=3 confirmed for reduced case; exact Q state/borders/phase moves/episode ending pending",
        "timing_scope": "generation_seconds incluye entrenamiento y verificación; excluye carga, metadatos, gráficos y reportes"}

    if args.mode == "train":
        experiments = run_experiments(mine, precedences, problem, learning, tuple(args.seeds), spatial)
        if spatial is not None:
            report["spatial_interpretation"] = {
                "q_key": "position, period; aliased, NOT complete Markov state",
                "learning_directions": ["north", "south", "west", "east", "down", "up"],
                "construction_directions": ["north", "south", "west", "east", "down"] + (["up"] if spatial.construction_up else []),
                "reward": "VAN incremental de bloques nuevos en el cono; confirmado por el usuario (B)",
                "start": "máximo Valor en superficie declarada; empate por ID; acción inicial contabilizada",
                "revisits": "tránsito sin extracción/recompensa/capacidad adicional",
                "periods": "avance explícito; cono completo en un período o rechazo atómico",
                "termination": "stop, todos extraídos o max_steps; límite artificial incluido en estado físico",
                "selection": "mejor prefijo factible observado de learning o construction; no modifica el aprendizaje ni detiene el recorrido",
                "client_validation": "pending; see docs/paper_mapping.md"}
        # Only after every seed has finished do we read reference/candidate metadata.
        selections = load_selections(data_path, mine)
        reference, _ = reference_report(mine, selections, ProblemParameters(), precedences)
        report["reference"] = reference
        reference_value = reference["recalculated_raw_value"]
        reference_matches = reference["economic_sum_matches"]
        reference_feasible = reference["evaluation"]["feasible"]
        comparable = problem == ProblemParameters() and reference_matches and reference_feasible is True
        comparison = {"target_from_workbook": selections.declared_value,
                      "reference_value_recalculated": reference_value,
                      "reference_matches_declaration": reference_matches,
                      "reference_feasible": reference_feasible,
                      "same_economic_scenario": problem == ProblemParameters(),
                      "comparable": comparable,
                      "precedence_status": "conditional" if precedences.provisional else "specified",
                      "runs": len(experiments),
                      "best_episode_target_hits": sum(math.isclose(e.result.best_evaluation.npv, selections.declared_value,
                                                                   rel_tol=1e-12, abs_tol=1e-6) for e in experiments) if comparable else None,
                      "greedy_target_hits": sum(math.isclose(e.result.greedy_evaluation.npv, selections.declared_value,
                                                             rel_tol=1e-12, abs_tol=1e-6) for e in experiments) if comparable else None,
                      "selected_path_target_hits": sum(math.isclose(e.result.selected_path.evaluation.npv, selections.declared_value,
                                                                    rel_tol=1e-12, abs_tol=1e-6) for e in experiments) if comparable else None}
        report["comparison"] = comparison
        if args.exact_benchmark:
            start = perf_counter()
            report["exact_benchmark"] = asdict(maximum_closure(mine, precedences, problem))
            report["benchmark_seconds"] = perf_counter() - start
            optimum = report["exact_benchmark"]["evaluation"]["npv"]
            report["benchmark_comparison"] = {
                "optimum": optimum,
                "best_episode_gaps": [optimum - e.result.best_evaluation.npv for e in experiments],
                "greedy_gaps": [optimum - e.result.greedy_evaluation.npv for e in experiments],
                "selected_path_gaps": [optimum - e.result.selected_path.evaluation.npv for e in experiments],
                "used_by_agent": False}
        directory = create_run_directory(project_path(args.output), args.mode)
        report["experiments"] = []
        for experiment in experiments:
            seed_dir = directory / f"seed-{experiment.seed}"
            seed_dir.mkdir()
            result = experiment.result
            write_json(seed_dir / "training.json", asdict(result))
            write_json(seed_dir / "q-table.json", q_table_payload(experiment.model))
            best_dir, greedy_dir = seed_dir / "best", seed_dir / "greedy"
            best_dir.mkdir()
            greedy_dir.mkdir()
            best_export = export_checked_plan(best_dir, mine, result.best_plan, problem, precedences)
            greedy_export = export_checked_plan(greedy_dir, mine, result.greedy_plan, problem, precedences)
            selected_dir = seed_dir / "selected"
            selected_dir.mkdir()
            selected_export = export_checked_plan(selected_dir, mine, result.selected_path.plan, problem, precedences)
            if args.plot:
                plot_history(result.history, seed_dir / "training.svg")
            report["experiments"].append({"seed": experiment.seed, "generation_seconds": experiment.generation_seconds,
                "best_episode": result.best_episode, "best_evaluation": asdict(result.best_evaluation),
                "greedy_evaluation": asdict(result.greedy_evaluation), "best_export": best_export,
                "greedy_export": greedy_export, "q_entries": len(experiment.model.q),
                "greedy_termination_reason": result.greedy_termination_reason,
                "greedy_steps": result.greedy_steps,
                "selected_path": asdict(result.selected_path), "selected_export": selected_export,
                "step_limit_episodes": sum(r.termination_reason == "step_limit" for r in result.history)})
        report["generation_seconds"] = sum(e.generation_seconds for e in experiments)
        report["validation_metadata"] = {"candidates": asdict(selections.candidates),
                                         "reference": asdict(selections.reference), "used_by_agent": False}
    else:
        if args.mode == "reference":
            selections = load_selections(data_path, mine)
        elif args.mode == "evaluate-plan":
            plan = load_plan(project_path(args.plan))
        start = perf_counter()
        if args.mode == "reference":
            reference, plan = reference_report(mine, selections, problem, precedences)
            report["reference"] = reference
        elif args.mode == "benchmark":
            benchmark = maximum_closure(mine, precedences, problem)
            report["exact_benchmark"] = asdict(benchmark)
            report["model"] = "maximum-closure-mincut-diagnostic-not-rl"
            plan = benchmark.plan
        evaluation = evaluate_plan(mine, plan, problem, precedences)
        report["generation_seconds"] = perf_counter() - start
        report["evaluation"] = asdict(evaluation)
        report["plan"] = plan_payload(plan)
        directory = create_run_directory(project_path(args.output), args.mode)
        report["plan_export"] = export_checked_plan(directory, mine, plan, problem, precedences)
    write_json(directory / "report.json", report)
    print(f"Reporte: {directory / 'report.json'}")
    if args.mode == "train":
        for entry in report["experiments"]:
            print(f"Semilla {entry['seed']}: mejor episodio={entry['best_evaluation']['npv']:g}; "
                  f"política greedy final={entry['greedy_evaluation']['npv']:g}; "
                  f"camino seleccionado={entry['selected_path']['evaluation']['npv']:g}")
        print(f"Coincidencias con referencia (mejor/greedy): {comparison['best_episode_target_hits']}/{comparison['greedy_target_hits']}; "
              f"ejecuciones={comparison['runs']}; precedencias={comparison['precedence_status']}")
        print(f"Caminos seleccionados que coinciden con referencia: {comparison['selected_path_target_hits']}")
    else:
        print(f"VAN recalculado: {evaluation.npv:g}; factible={evaluation.feasible}; precedencias={evaluation.precedence_status}")
    return directory


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        execute(args)
    except (ValueError, OSError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
