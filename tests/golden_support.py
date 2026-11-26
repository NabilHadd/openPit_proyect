"""Fingerprints of Q-learning fits, shared by tests/test_golden_q_learning.py and its
subprocess check. Only public fitted attributes are read (q_, history_, best_observed_,
policy_, rng_), so the fingerprint survives refactors that keep them.

Generated with Python 3.12 (the project's version): another version may give other
fingerprints. Regenerate tests/golden/q_learning.json ONLY when a change is meant to alter
results:

    PYTHONPATH=. python tests/golden_support.py --write
"""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

from configs.config_settings import EnvironmentParameters, ProblemParameters
from src.data.dataset import load_mine
from src.data.instance import MineInstance
from src.data.precedences import pit_precedences
from src.evaluation.tiny_oracle import tiny_mine
from src.models.q_learning import QLearningEstimator


GOLDEN = Path(__file__).resolve().parent / "golden" / "q_learning.json"
SEEDS = (0, 1, 2)


def synthetic_mine():
    """Fabricated 7x5x3 grid with varied values (NOT the client's data)."""
    rows = ["id,x,y,z,value"]
    for z in range(1, 4):
        for y in range(1, 6):
            for x in range(1, 8):
                value = (x * 7 + y * 3 + z * 5) % 11 - 4
                rows.append(f"{x + (y - 1) * 7 + (z - 1) * 35},{x},{y},{z},{value}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.csv"
        path.write_text("\n".join(rows) + "\n")
        mine = load_mine(path)
    return mine, pit_precedences(mine)


def cases():
    """name -> (instance, params). Every key/horizon/problem combination on two mines."""
    mines = {"tiny": (tiny_mine(), 2000, ProblemParameters(2, 0.1, 3)),
             "grid": (synthetic_mine(), 1500, ProblemParameters(2, 0.1, 10))}
    result = {}
    for mine_name, ((mine, rules), transitions, multi) in mines.items():
        for state in ("physical", "position"):
            for horizon in ("finite", "external"):
                for problem_name, problem in (("one", ProblemParameters()), ("multi", multi)):
                    instance = MineInstance(mine, rules, problem, EnvironmentParameters(30, horizon))
                    params = {"state": state, "transitions": transitions, "policy_rollouts": 3,
                              "epsilon": 0.5, "epsilon_schedule": "linear"}
                    result[f"{mine_name}-{state}-{horizon}-{problem_name}"] = instance, params
    return result


def _digest(payload) -> str:
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _key(key):
    payload = asdict(key)
    if "extracted" in payload:
        payload["extracted"] = sorted(payload["extracted"])
    return payload


def _plan(plan):
    return [[e.block_id, e.period] for e in plan.extractions]


def fingerprint(model) -> dict:
    """Hash of each part of a fit, so a mismatch says which part changed."""
    best = model.best_observed_
    return {
        "q": _digest([[_key(k), a, v] for (k, a), v in model.q_.items()]),  # insertion order matters
        "history": _digest([asdict(e) for e in model.history_]),
        "best": _digest([_plan(best.plan), best.evaluation.npv, best.episode, best.step]),
        "policy": _digest([[_plan(r.plan), r.evaluation.npv, r.steps, r.end, r.tie_decisions,
                            r.all_zero_decisions, r.unseen_state_decisions] for r in model.policy_]),
        "rng": _digest(repr(model.rng_.getstate())),
        "best_npv": best.evaluation.npv,
    }


def fit_case(name: str, seed: int):
    instance, params = cases()[name]
    return QLearningEstimator(**params, random_state=seed).fit(instance)


# Other models checked across processes (PYTHONHASHSEED): their own params on the same cases.
OTHER_MODELS = {"dyna_q": {"planning_steps": 3}, "prioritized_sweeping": {"planning_steps": 3}}


def model_fingerprint(key: str, name: str, seed: int) -> dict:
    from src.models.register import spec
    instance, params = cases()[name]
    if key == "beam_search":
        model = spec(key).estimator(width=3, random_state=seed).fit(instance)
        return {"plan": _digest(_plan(model.plan_)), "trace": _digest([[c, v] for c, v in model.trace_]),
                "counters": model.counters_}
    model = spec(key).estimator(**params, **OTHER_MODELS.get(key, {}), random_state=seed).fit(instance)
    return {**fingerprint(model), "counters": model.counters_}


def compute() -> dict:
    return {name: {str(seed): fingerprint(QLearningEstimator(**params, random_state=seed).fit(instance))
                   for seed in SEEDS}
            for name, (instance, params) in cases().items()}


if __name__ == "__main__":
    if "--write" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n")
        print(f"Escrito {GOLDEN}")
    elif "--case" in sys.argv:  # used by the PYTHONHASHSEED test: --case NAME SEED [MODEL]
        name, seed, *model = sys.argv[sys.argv.index("--case") + 1:]
        if model and model[0] != "q_learning":
            print(json.dumps(model_fingerprint(model[0], name, int(seed)), sort_keys=True))
        else:
            print(json.dumps(fingerprint(fit_case(name, int(seed))), sort_keys=True))
