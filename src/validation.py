"""Standalone four-phase validation entry point; historical main.py is unchanged."""

import argparse
import json

from src.config import LearningParameters, SpatialParameters, project_path
from src.data.dataset import load_mine
from src.data.precedences import pit_precedences
from src.evaluation.tiny_oracle import tiny_mine
from src.training.validation_runner import ValidationBudget
from src.training.validation_suite import run_validation_suite


def parser():
    result = argparse.ArgumentParser(description="Validación Q-learning experimental: gamma=1, estado exacto, presupuesto por transiciones")
    result.add_argument("--data", help="CSV estándar de bloques (id,x,y,z,value[,tonnes]); sin --data se usa la mina sintética de seis bloques")
    result.add_argument("--transitions", required=True, type=int)
    result.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    result.add_argument("--alpha", type=float, default=1)
    result.add_argument("--epsilon", type=float, default=.3)
    result.add_argument("--max-steps", type=int)
    result.add_argument("--horizon-mode", choices=("finite", "external"), default="finite")
    result.add_argument("--evaluation-rollouts", type=int, default=20)
    result.add_argument("--evaluation-ties", choices=("uniform", "ordered"), default="uniform",
                        help="ordered conserva el desempate anterior SOLO como diagnóstico")
    result.add_argument("--output", default="artifacts/reports")
    return result


def execute(args):
    if args.data:
        mine = load_mine(project_path(args.data))
        rules = pit_precedences(mine)
        cap = 500 if args.max_steps is None else args.max_steps
    else:
        mine, rules = tiny_mine()
        cap = 12 if args.max_steps is None else args.max_steps
    spatial = SpatialParameters(cap, True, args.horizon_mode)
    learning = LearningParameters(alpha=args.alpha, gamma=1, epsilon=args.epsilon)
    budget = ValidationBudget(args.transitions, args.evaluation_rollouts, args.evaluation_ties)
    directory = run_validation_suite(mine, rules, spatial, learning, tuple(args.seeds), budget, project_path(args.output))
    report = json.loads((directory / "report.json").read_text())
    summary = report["summary"]
    print(f"Validación EXPERIMENTAL, no modelo final. Reporte: {directory / 'report.json'}")
    print(f"Mejor observado={summary['best_observed_max']:g}; hits observados={summary['best_observed_hits']}/{summary['seed_count']}; "
          f"política final completada={summary['policy_completed_hits']}/{summary['policy_rollouts']}; "
          f"aciertos truncados={summary['policy_truncated_value_hits']}; media={summary['policy_mean']:g}")
    return directory


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        directory = execute(args)
    except (ValueError, OSError) as exc:
        cli.exit(2, f"Error: {exc}\n")
        return
    report = json.loads((directory / "report.json").read_text())
    if report["summary"].get("tiny_policy_gate") == "failed":
        cli.exit(1, "La política no superó la prueba diminuta; no escalar este protocolo todavía.\n")


if __name__ == "__main__":
    main()
