"""Statistical projections of completed validation runs; no training or file I/O."""

import math
from statistics import mean, median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.tiny_oracle import TinyOracleResult
    from src.training.validation_runner import ValidationRun


def summarize_run(run: "ValidationRun", *, directory: str, exports: dict[str, str]) -> dict:
    values = [p.path.evaluation.npv for p in run.policy]
    return {"seed": run.seed, "directory": directory, "exports": exports,
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
            "generation_seconds": run.generation_seconds}


def attach_benchmark_metrics(row: dict, *, target: float, oracle: "TinyOracleResult | None",
                             finite_horizon: bool) -> None:
    row["best_observed_gap"] = target - row["best_observed_npv"]
    row["best_observed_hit"] = math.isclose(row["best_observed_npv"], target, abs_tol=1e-6, rel_tol=1e-12)
    row["policy_hits"] = sum(math.isclose(v, target, abs_tol=1e-6, rel_tol=1e-12) for v in row["policy_values"])
    row["policy_completed_hits"] = sum(done and math.isclose(v, target, abs_tol=1e-6, rel_tol=1e-12)
                                       for v, done in zip(row["policy_values"], row["policy_completed_flags"], strict=True))
    row["policy_truncated_value_hits"] = row["policy_hits"] - row["policy_completed_hits"]
    row["policy_mean_gap"] = target - row["policy_mean"]
    if oracle is not None and finite_horizon:
        row["training"]["reachable_q_state_coverage"] = row["training"]["unique_q_states"] / oracle.reachable_states
        row["training"]["reachable_state_action_coverage"] = row["training"]["unique_state_action_pairs"] / oracle.transitions


def summarize_suite(rows: list[dict], *, evaluation_rollouts: int, training_transitions: int, tiny: bool) -> dict:
    count = evaluation_rollouts
    summary = {"seed_count": len(rows), "comparison_status": "ten_seed_comparison" if len(rows) >= 10 else "smoke_fewer_than_10_seeds",
               "best_observed_max": max(r["best_observed_npv"] for r in rows),
               "best_observed_hits": sum(r["best_observed_hit"] for r in rows),
               "policy_mean": mean(r["policy_mean"] for r in rows),
               "policy_hits": sum(r["policy_hits"] for r in rows),
               "policy_completed_hits": sum(r["policy_completed_hits"] for r in rows),
               "policy_truncated_value_hits": sum(r["policy_truncated_value_hits"] for r in rows),
               "policy_rollouts": count * len(rows),
               "seeds_reproducing_target_in_every_rollout": sum(r["policy_hits"] == count for r in rows),
               "seeds_completing_target_in_every_rollout": sum(r["policy_completed_hits"] == count for r in rows),
               "training_transitions_per_seed": training_transitions,
               "policy_transitions": sum(r["policy_transitions"] for r in rows)}
    if tiny:
        summary["tiny_policy_gate"] = ("not_assessed_fewer_than_10_seeds" if len(rows) < 10 else
                                       "passed" if summary["seeds_completing_target_in_every_rollout"] == len(rows) else "failed")
    return summary
