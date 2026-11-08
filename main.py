"""Entry point. Every value lives in the YAML given by --config; the terminal only carries
the mode, boolean flags and, for evaluate-plan, the plan to evaluate.

    python main.py train --config configs/config.yml [--plot]
    python main.py evaluate-plan --config configs/config.yml [--plan RUTA]
"""

import argparse
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
from statistics import mean

from src.config import DEFAULT_CONFIG, RunConfig, load_config, project_path
from src.data.dataset import Mine, Solution, load_mine, load_solution
from src.data.precedences import Precedences, order_selection, pit_precedences
from src.evaluation.metrics import Extraction, Plan, evaluate_plan
from src.evaluation.plots import plot_history
from src.evaluation.reporter import (code_provenance, create_run_directory, mine_metadata, q_table_payload, runtime_metadata, write_json,
                                     write_plan_csv)
from src.training.trainer import train


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Planificación minera con Q-learning espacial")
    modes = result.add_subparsers(dest="mode", required=True)
    for name, text in (("train", "Entrenar un agente por semilla y exportar sus planes"),
                       ("evaluate-plan", "Calcular el VAN y verificar la factibilidad de un plan")):
        mode = modes.add_parser(name, help=text, description=text)
        mode.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML de configuración; default={DEFAULT_CONFIG}")
        if name == "train":
            mode.add_argument("--plot", action="store_true", help="Guardar la curva de aprendizaje SVG por semilla")
        else:
            mode.add_argument("--plan", help="CSV del plan (order,id,period); sin --plan se evalúa la solución del YAML")
    return result


def same_value(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-6)


def file_metadata(solution: Solution) -> dict:
    return {"path": solution.source, "sha256": solution.data_sha256, "ordered": solution.ordered,
            "blocks": len(solution.entries)}


def plan_report(mine: Mine, solution: Solution, config: RunConfig, precedences: Precedences) -> tuple[dict, Plan]:
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
                    ids = order_selection(ids, precedences, done)
                except ValueError as exc:
                    order_error = str(exc)
            extractions.extend(Extraction(i, period) for i in ids)
            done.update(ids)
        extractions = tuple(extractions)
    plan = Plan(extractions)
    evaluation = evaluate_plan(mine, plan, config.problem, precedences)
    return {"file": file_metadata(solution), "feasible": evaluation.feasible, "npv": evaluation.npv,
            "raw_value": evaluation.raw_value, "period_usage": evaluation.period_usage,
            "errors": evaluation.errors[:20], "order": "del archivo" if solution.ordered else
            ("testigo construido para verificar" if order_error is None else f"sin orden válido: {order_error}")}, plan


def base_report(mode: str, config: RunConfig, mine: Mine, precedences: Precedences) -> dict:
    return {"schema_version": 2, "mode": mode, "config": config.as_dict(), "code": code_provenance(),
            "runtime": runtime_metadata(), "data": mine_metadata(mine), "precedence_rule": precedences.rule}


def load_inputs(config: RunConfig) -> tuple[Mine, Precedences]:
    mine = load_mine(project_path(config.data))
    return mine, pit_precedences(mine)


def solution_path(config: RunConfig) -> Path | None:
    if config.solution is None:
        return None
    path = project_path(config.solution)
    if not path.is_file():
        raise ValueError(f"No existe la solución {path}; corrija solution en el YAML o use solution: null")
    return path


def seed_row(result, directory: Path) -> dict:
    """Write one seed's files and return its key figures for report.json."""
    folder = directory / f"seed-{result.seed}"
    folder.mkdir()
    write_plan_csv(folder / "best_observed.csv", result.best_observed.plan)
    best_rollout = max(result.policy, key=lambda r: r.evaluation.npv)  # first one among ties
    write_plan_csv(folder / "policy_best.csv", best_rollout.plan)  # already verified in its rollout
    decisions = sum(r.steps for r in result.policy)
    values = [r.evaluation.npv for r in result.policy]
    write_json(folder / "training.json", {
        "episodes": [asdict(e) for e in result.history],
        "policy_rollouts": [{k: v for k, v in asdict(r).items() if k != "plan"} for r in result.policy]})
    return {
        "seed": result.seed, "seconds": result.seconds,
        "training": {"transitions": result.transitions, "episodes": len(result.history),
                     "episode_ends": dict(sorted(Counter(e.end for e in result.history).items())),
                     "unique_states": result.unique_states, "unique_pairs": result.unique_pairs,
                     "pairs_visited_once": result.pairs_visited_once / result.unique_pairs},
        "best_observed": {"npv": result.best_observed.evaluation.npv, "episode": result.best_observed.episode,
                          "step": result.best_observed.step, "blocks": len(result.best_observed.plan.extractions),
                          "file": f"{folder.name}/best_observed.csv"},
        "policy": {"rollouts": len(values), "npv_mean": mean(values), "npv_min": min(values), "npv_max": max(values),
                   "tie_decisions": sum(r.tie_decisions for r in result.policy) / decisions,
                   "all_zero_decisions": sum(r.all_zero_decisions for r in result.policy) / decisions,
                   "unseen_state_decisions": sum(r.unseen_state_decisions for r in result.policy) / decisions,
                   "episode_cap_rollouts": sum(r.end == "episode_cap" for r in result.policy),
                   "best_file": f"{folder.name}/policy_best.csv"},
    }


def run_train(config: RunConfig, *, plot: bool = False) -> Path:
    mine, precedences = load_inputs(config)
    reference_file = solution_path(config)
    directory = create_run_directory(project_path(config.output), "train")
    report = base_report("train", config, mine, precedences)
    rows, policy_values = [], []
    for seed in config.seeds:
        result, model = train(mine, precedences, config.problem, config.agent, config.training,
                              config.evaluation, seed)
        rows.append(seed_row(result, directory))
        policy_values.append([r.evaluation.npv for r in result.policy])
        write_json(directory / f"seed-{seed}" / "q-table.json", q_table_payload(model))
        if plot:
            plot_history([e.npv for e in result.history], directory / f"seed-{seed}" / "training.svg")

    # The reference is read only after every seed has finished training.
    reference = None
    if reference_file is not None:
        reference, _ = plan_report(mine, load_solution(reference_file, mine), config, precedences)
        if reference["feasible"]:
            for row, values in zip(rows, policy_values, strict=True):
                row["best_observed"]["matches_reference"] = same_value(row["best_observed"]["npv"], reference["npv"])
                row["policy"]["matches_reference"] = sum(same_value(v, reference["npv"]) for v in values)
    report["reference"] = reference
    report["seeds"] = rows
    best = [r["best_observed"]["npv"] for r in rows]
    summary = {"seeds": len(rows), "best_observed_max": max(best), "best_observed_mean": mean(best),
               "policy_npv_mean": mean(r["policy"]["npv_mean"] for r in rows)}
    if reference is not None and reference["feasible"]:
        rollouts = config.evaluation.policy_rollouts
        summary.update({"reference_npv": reference["npv"],
                        "best_observed_matches": sum(r["best_observed"]["matches_reference"] for r in rows),
                        "policy_matches": sum(r["policy"]["matches_reference"] for r in rows),
                        "policy_rollouts": rollouts * len(rows),
                        "seeds_policy_always_matches": sum(r["policy"]["matches_reference"] == rollouts for r in rows)})
    report["summary"] = summary
    write_json(directory / "report.json", report)

    print(f"Reporte: {directory / 'report.json'}")
    for row in rows:
        policy = row["policy"]
        print(f"Semilla {row['seed']}: mejor observado={row['best_observed']['npv']:g}; "
              f"política media={policy['npv_mean']:g} (mín {policy['npv_min']:g}, máx {policy['npv_max']:g})")
    if "reference_npv" in summary:
        print(f"Referencia {summary['reference_npv']:g}: mejor observado la iguala en "
              f"{summary['best_observed_matches']}/{len(rows)} semillas; política en "
              f"{summary['policy_matches']}/{summary['policy_rollouts']} ejecuciones")
    elif reference is not None:
        print("La solución de referencia no es factible; no se compara")
    else:
        print("Sin solución de referencia (solution: null)")
    return directory


def run_evaluate_plan(config: RunConfig, plan: str | None = None) -> Path:
    mine, precedences = load_inputs(config)
    if plan is not None:
        plan_file = project_path(plan)
    else:
        plan_file = solution_path(config)
        if plan_file is None:
            raise ValueError("evaluate-plan sin --plan evalúa la solución del YAML, pero solution es null")
    evaluated, witness = plan_report(mine, load_solution(plan_file, mine), config, precedences)
    directory = create_run_directory(project_path(config.output), "evaluate-plan")
    report = base_report("evaluate-plan", config, mine, precedences)
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
        return run_train(config, plot=args.plot)
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
