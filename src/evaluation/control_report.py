"""Post-hoc comparisons: target values enter here only after all controls finish."""

from dataclasses import asdict
import math
from statistics import mean, median

from src.training.controlled_experiments import CONTROL_SPECS


def matches(value: float, target: float) -> bool:
    return math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-6)


def first_benchmark_hit(training_payload: dict, target: float) -> dict | None:
    """Analyze saved history, without resuming or updating any model."""
    cumulative = 0
    for record in training_payload["history"]:
        path = record.get("best_path")
        if path is not None and path["evaluation"]["feasible"] is True and matches(path["evaluation"]["npv"], target):
            return {"phase": "learning", "episode": record["episode"], "step": path["steps"],
                    "training_transitions_before_hit": cumulative + path["steps"]}
        cumulative += record["steps"]
    path = training_payload.get("greedy_best_path")
    if path is not None and path["evaluation"]["feasible"] is True and matches(path["evaluation"]["npv"], target):
        return {"phase": "construction", "episode": None, "step": path["steps"],
                "training_transitions_before_hit": cumulative}
    return None


def attach_benchmark_outcome(row: dict, training_payload: dict, target: float):
    row["benchmark"] = {
        "target": target,
        "selected_hit": row["selected_evaluation"]["feasible"] is True and matches(row["selected_evaluation"]["npv"], target),
        "best_episode_hit": row["best_episode_evaluation"]["feasible"] is True and matches(row["best_episode_evaluation"]["npv"], target),
        "greedy_hit": row["greedy_evaluation"]["feasible"] is True and matches(row["greedy_evaluation"]["npv"], target),
        "selected_gap": target - row["selected_evaluation"]["npv"],
        "greedy_gap": target - row["greedy_evaluation"]["npv"],
        "first_observed_hit": first_benchmark_hit(training_payload, target),
        "used_by_agent": False,
    }


def summarize_controls(rows: list[dict], seeds: tuple[int, ...], *, reference_compatible: bool = False) -> dict:
    groups = {}
    for spec in CONTROL_SPECS:
        runs = [row for row in rows if row["control"]["id"] == spec.id]
        if sorted(row["seed"] for row in runs) != sorted(seeds):
            raise ValueError("La suite está incompleta o tiene semillas duplicadas")
        scores = [r["selected_evaluation"]["npv"] for r in runs]
        hits = sum(r["benchmark"]["selected_hit"] for r in runs)
        if len(seeds) < 10:
            milestone = "not_assessed_fewer_than_10_seeds"
        elif not reference_compatible:
            milestone = "not_assessed_reference_unverified_or_incompatible"
        else:
            milestone = "achieved" if hits else "not_achieved"
        groups[spec.id] = {
            "control": asdict(spec), "runs": len(runs),
            "selected_max": max(scores), "selected_mean": mean(scores), "selected_median": median(scores),
            "selected_hits": hits, "selected_success_rate": hits / len(runs),
            "selected_gap_mean": mean(r["benchmark"]["selected_gap"] for r in runs),
            "greedy_max": max(r["greedy_evaluation"]["npv"] for r in runs),
            "greedy_mean": mean(r["greedy_evaluation"]["npv"] for r in runs),
            "greedy_hits": sum(r["benchmark"]["greedy_hit"] for r in runs),
            "generation_seconds": sum(r["generation_seconds"] for r in runs),
            "learning_transitions": sum(r["diagnostics"]["learning"]["transitions"] for r in runs),
            "greedy_step_limit_runs": sum(r["diagnostics"]["greedy"]["termination_reason"] == "step_limit" for r in runs),
            "mean_q_state_reuse_fraction": mean(r["diagnostics"]["state_coverage"]["q_state_reuse_fraction"] for r in runs),
            "mean_state_action_reuse_fraction": mean(r["diagnostics"]["state_coverage"]["state_action_reuse_fraction"] for r in runs),
            "mean_q_singleton_fraction": mean(r["diagnostics"]["state_coverage"]["q_states_visited_once_fraction"] for r in runs),
            "mean_no_extraction_fraction": mean(r["diagnostics"]["learning"]["no_extraction_fraction"] for r in runs),
            "milestone_1": milestone,
        }
    paired = {}
    for label, before, after in (
        ("gamma_effect_position", "A_position_g07", "B_position_g1"),
        ("gamma_effect_physical", "C_physical_g07", "D_physical_g1"),
        ("encoding_effect_g07", "A_position_g07", "C_physical_g07"),
        ("encoding_effect_g1", "B_position_g1", "D_physical_g1"),
    ):
        a = {r["seed"]: r for r in rows if r["control"]["id"] == before}
        b = {r["seed"]: r for r in rows if r["control"]["id"] == after}
        differences = [{"seed": seed, "selected_npv_delta": b[seed]["selected_evaluation"]["npv"] - a[seed]["selected_evaluation"]["npv"]} for seed in seeds]
        paired[label] = {"before": before, "after": after, "per_seed": differences,
                         "mean_delta": mean(d["selected_npv_delta"] for d in differences),
                         "median_delta": median(d["selected_npv_delta"] for d in differences)}
    return {"controls": groups, "paired_descriptive_differences": paired,
            "interpretation": "Controles experimentales, no selección del modelo final ni prueba de convergencia. Igual presupuesto de episodios/horizonte, no de transiciones ni tiempo. Diferencias descriptivas, no significancia estadística."}


def summary_markdown(summary: dict, target: float) -> str:
    lines = ["# Controles experimentales — NO modelo final", "",
             f"Óptimo del benchmark: {target:g}. Calculado después de todas las semillas; no usado por el agente.", "",
             "| Control | Éxitos observados | Máx. seleccionado | Media | Máx. greedy | Reuso Q | Greedy por límite | Hito 1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, row in summary["controls"].items():
        lines.append(f"| {name} | {row['selected_hits']}/{row['runs']} | {row['selected_max']:g} | {row['selected_mean']:g} | {row['greedy_max']:g} | {row['mean_q_state_reuse_fraction']:.1%} | {row['greedy_step_limit_runs']}/{row['runs']} | {row['milestone_1']} |")
    lines.extend(["", summary["interpretation"], "", "Los retornos de navegación no invalidan planes; el horizonte forma parte del estado físico exacto. Ninguna variante se promueve automáticamente.", ""])
    return "\n".join(lines)
