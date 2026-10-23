"""Post-hoc reporting for complete, transition-matched validation seed suites."""

from dataclasses import asdict
import math
from pathlib import Path
from statistics import mean, median

from src.config import ProblemParameters
from src.evaluation.benchmark import maximum_closure
from src.evaluation.reporter import (code_provenance, create_run_directory, export_checked_plan, mine_metadata,
                                     precedence_metadata, q_table_payload, runtime_metadata, write_json)
from src.evaluation.tiny_oracle import enumerate_tiny
from src.training.validation_runner import train_transition_budget


def run_validation_suite(mine, rules, spatial, learning, seeds, budget, output: Path) -> Path:
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError("Se requieren semillas enteras distintas y una lista no vacía")
    if learning.gamma != 1 or not spatial.construction_up:
        raise ValueError("La validación requiere gamma=1 y seis direcciones en ambas fases")
    directory = create_run_directory(output, "learning-validation")
    report = {"schema_version": 1, "status": "running", "purpose": "validation-only", "is_final_model": False,
              "benchmark_used_by_agent": False, "code": code_provenance(), "runtime": runtime_metadata(),
              "data": mine_metadata(mine), "precedences": precedence_metadata(rules),
              "config": {"seeds": list(seeds), "budget": asdict(budget), "learning": asdict(learning),
                         "spatial": asdict(spatial), "episodes_parameter_used": False},
              "semantics": {"stop": "true terminal, no bootstrap", "all_extracted": "true terminal, no bootstrap",
                            "finite_horizon": "true terminal, no bootstrap; clock retained in Q",
                            "external_cap": "nonterminal truncation, bootstrap from successor before reset; stationary clock",
                            "transition_budget": "external truncation unless the same transition truly terminates",
                            "evaluation_rng": "random.Random(f'{training_seed}:evaluation:{rollout_index}')",
                            "all_zero_q": "no numerical action preference; NOT equivalent to never updated",
                            "best_observed": "training prefixes only, never evaluation",
                            "coverage": "observed keys/pairs; full reachable-space denominator only on tiny finite mines"},
              "expected_runs": len(seeds), "completed_runs": 0, "runs": []}
    write_json(directory / "report.json", report)
    try:
        for seed in seeds:
            run = train_transition_budget(mine, rules, spatial, learning, seed, budget)
            if run.seed != seed or run.training["transitions"] != budget.transitions:
                raise RuntimeError("El ejecutor cambió la semilla o el presupuesto de transiciones")
            folder = directory / f"seed-{seed}"
            folder.mkdir()
            context = {"protocol": "learning-validation", "seed": seed, "purpose": "experimental-only"}
            tag = {"is_final_model": False, "experimental_control": context}
            write_json(folder / "training.json", {**run.training, "best_observed": asdict(run.best_observed),
                                                   "best_completed": asdict(run.best_completed) if run.best_completed else None, **tag})
            write_json(folder / "policy.json", {"rollouts": [asdict(p) for p in run.policy], **tag})
            write_json(folder / "q-table.json", {**q_table_payload(run.model), **tag})
            paths = [("best-observed", run.best_observed)]
            if run.best_completed is not None:
                paths.append(("best-completed", run.best_completed))
            paths.extend((f"policy-{i}", p.path) for i, p in enumerate(run.policy))
            exports = {}
            for role, path in paths:
                destination = folder / role
                destination.mkdir()
                name = export_checked_plan(destination, mine, path.plan, parameters=ProblemParameters(), precedences=rules,
                                           experimental_context={**context, "selection": role})
                if name is None:
                    raise RuntimeError("El verificador rechazó un plan de validación")
                exports[role] = name
            values = [p.path.evaluation.npv for p in run.policy]
            report["runs"].append({"seed": seed, "directory": folder.name, "exports": exports,
                                   "best_observed_npv": run.best_observed.evaluation.npv,
                                   "best_completed_npv": run.best_completed.evaluation.npv if run.best_completed else None,
                                   "policy_values": values, "policy_mean": mean(values), "policy_median": median(values),
                                   "policy_completed_flags": [p.terminated and not p.truncated for p in run.policy],
                                   "policy_min": min(values), "policy_max": max(values),
                                   "policy_transitions": sum(p.path.steps for p in run.policy),
                                   "policy_truncations": sum(p.truncated for p in run.policy),
                                   "policy_termination_counts": {reason: sum(p.reason == reason for p in run.policy)
                                                                  for reason in sorted({p.reason for p in run.policy})},
                                   "policy_decisions": [p.decisions for p in run.policy],
                                   "training": {k: v for k, v in run.training.items() if k != "history"},
                                   "generation_seconds": run.generation_seconds})
            report["completed_runs"] += 1
            write_json(directory / "report.json", report)
            del run
        if report["completed_runs"] != report["expected_runs"]:
            raise RuntimeError("Suite incompleta: no se calcula una tasa de éxito parcial")
        # The trainer has no oracle argument. Nothing below executes before all seeds finish.
        oracle = None
        if len(mine.blocks) <= 6:
            oracle = enumerate_tiny(mine, rules, spatial)
            target = oracle.optimum
            report["benchmark"] = {"kind": "independent-exhaustive-action-oracle", **asdict(oracle), "used_by_agent": False}
        else:
            closure = maximum_closure(mine, rules)
            target = closure.evaluation.npv
            report["benchmark"] = {"kind": "economic-upper-bound", "value": target,
                                   "spatial_attainability": "not_certified_by_this_run", "used_by_agent": False}
        for row in report["runs"]:
            row["best_observed_gap"] = target - row["best_observed_npv"]
            row["best_observed_hit"] = math.isclose(row["best_observed_npv"], target, abs_tol=1e-6, rel_tol=1e-12)
            row["policy_hits"] = sum(math.isclose(v, target, abs_tol=1e-6, rel_tol=1e-12) for v in row["policy_values"])
            row["policy_completed_hits"] = sum(done and math.isclose(v, target, abs_tol=1e-6, rel_tol=1e-12)
                                                for v, done in zip(row["policy_values"], row["policy_completed_flags"], strict=True))
            row["policy_truncated_value_hits"] = row["policy_hits"] - row["policy_completed_hits"]
            row["policy_mean_gap"] = target - row["policy_mean"]
            if oracle is not None and spatial.horizon_mode == "finite":
                row["training"]["reachable_q_state_coverage"] = row["training"]["unique_q_states"] / oracle.reachable_states
                row["training"]["reachable_state_action_coverage"] = row["training"]["unique_state_action_pairs"] / oracle.transitions
        count = budget.evaluation_rollouts
        summary = {"seed_count": len(seeds), "comparison_status": "ten_seed_comparison" if len(seeds) >= 10 else "smoke_fewer_than_10_seeds",
                   "best_observed_max": max(r["best_observed_npv"] for r in report["runs"]),
                   "best_observed_hits": sum(r["best_observed_hit"] for r in report["runs"]),
                   "policy_mean": mean(r["policy_mean"] for r in report["runs"]),
                   "policy_hits": sum(r["policy_hits"] for r in report["runs"]),
                   "policy_completed_hits": sum(r["policy_completed_hits"] for r in report["runs"]),
                   "policy_truncated_value_hits": sum(r["policy_truncated_value_hits"] for r in report["runs"]),
                   "policy_rollouts": count * len(seeds),
                   "seeds_reproducing_target_in_every_rollout": sum(r["policy_hits"] == count for r in report["runs"]),
                   "seeds_completing_target_in_every_rollout": sum(r["policy_completed_hits"] == count for r in report["runs"]),
                   "training_transitions_per_seed": budget.transitions,
                   "policy_transitions": sum(r["policy_transitions"] for r in report["runs"])}
        if len(mine.blocks) <= 6:
            summary["tiny_policy_gate"] = ("not_assessed_fewer_than_10_seeds" if len(seeds) < 10 else
                                            "passed" if summary["seeds_completing_target_in_every_rollout"] == len(seeds) else "failed")
        report.update(status="completed", summary=summary)
        write_json(directory / "report.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "report.json", report)
        raise
    return directory
