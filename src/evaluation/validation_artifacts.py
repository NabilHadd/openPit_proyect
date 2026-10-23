"""Filesystem adapter for validation provenance, checkpoints and checked plans."""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import ProblemParameters
from src.evaluation.reporter import (code_provenance, create_run_directory, export_checked_plan, mine_metadata,
                                     precedence_metadata, q_table_payload, runtime_metadata, write_json)

if TYPE_CHECKING:
    from src.training.validation_runner import ValidationRun


class FileValidationArtifacts:
    def start_suite(self, mine, rules, spatial, learning, seeds, budget, output: Path) -> tuple[Path, dict]:
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
        self.write_report(directory, report)
        return directory, report

    def save_run(self, directory: Path, mine, rules, run: "ValidationRun") -> tuple[str, dict[str, str]]:
        folder = directory / f"seed-{run.seed}"
        folder.mkdir()
        context = {"protocol": "learning-validation", "seed": run.seed, "purpose": "experimental-only"}
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
        return folder.name, exports

    def write_report(self, directory: Path, report: dict) -> None:
        write_json(directory / "report.json", report)
