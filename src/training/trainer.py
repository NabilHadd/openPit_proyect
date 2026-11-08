"""Training by transition budget, best observed plan and learned-policy rollouts.

- Training runs epsilon-greedy episodes until `transitions` steps have been taken.
- An episode ends for real (no future value) on stop, all reachable blocks extracted or,
  in finite mode, the step limit. Reaching max_steps in external mode, or running out of
  budget, only interrupts it: the last update still counts the future value.
- Best observed: after every extraction the extracted set is a feasible plan; the one of
  highest VAN seen during training is the plan delivered.
- Policy: after training, the agent builds plans without exploration (always the action
  of highest Q); ties are broken at random, so several rollouts measure consistency.
"""

from collections import Counter
from dataclasses import dataclass
import math
import random
from time import perf_counter

from src.config import AgentParameters, EvaluationParameters, ProblemParameters, TrainingParameters
from src.data.dataset import Mine
from src.data.precedences import Precedences
from src.envs.mine_env import MineEnv
from src.evaluation.metrics import Evaluation, Plan, evaluate_plan
from src.models.q_learning import QLearning


@dataclass(frozen=True)
class ObservedPlan:
    plan: Plan
    evaluation: Evaluation
    episode: int
    step: int


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    steps: int
    npv: float
    end: str  # stop | all_extracted | step_limit | episode_cap | budget


@dataclass(frozen=True)
class PolicyRollout:
    rollout: int
    plan: Plan
    evaluation: Evaluation
    steps: int
    end: str
    tie_decisions: int
    all_zero_decisions: int
    unseen_state_decisions: int


@dataclass(frozen=True)
class TrainingResult:
    seed: int
    best_observed: ObservedPlan
    policy: tuple[PolicyRollout, ...]
    history: tuple[EpisodeRecord, ...]
    transitions: int
    unique_states: int
    unique_pairs: int
    pairs_visited_once: int
    seconds: float


def checked_evaluation(env: MineEnv, plan: Plan | None = None) -> Evaluation:
    result = evaluate_plan(env.mine, env.plan if plan is None else plan, env.parameters, env.precedences)
    if result.feasible is not True:
        raise RuntimeError(f"El entorno produjo un plan inválido: {result.errors}")
    return result


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-6)


def q_bound_violations(model: QLearning, mine: Mine) -> list[dict]:
    """No future return can exceed the positive value still in the ground (gamma <= 1).

    Only for the physical key. The position key forgets what was extracted, so it can
    credit the same block again and again: its Q is an approximation, not a true value."""
    if model.parameters.state != "physical":
        return []
    positive = {b.id: b.value for b in mine.blocks if b.value > 0}
    violations = []
    for (key, action), value in model.q.items():
        extracted = getattr(key, "extracted", frozenset())
        bound = math.fsum(v for i, v in positive.items() if i not in extracted)
        if not math.isfinite(value) or value - bound > 1e-6:
            violations.append({"action": action, "q": value, "upper_bound": bound})
    return violations


def _policy_rollout(env: MineEnv, model: QLearning, visited: set, rollout: int) -> PolicyRollout:
    rng = random.Random(f"{model.seed}:policy:{rollout}")  # never touches the training RNG
    state = env.reset()
    steps = ties = zeros = unseen = 0
    while not state.terminal and steps < env.training.max_steps:
        actions = env.available_actions()
        best = model.best_actions(state, actions)
        ties += len(best) > 1
        zeros += all(model.value(state, a) == 0 for a in actions)
        unseen += model.state_key(state) not in visited
        state, _, _ = env.step(rng.choice(best))
        steps += 1
    end = env.termination_reason if state.terminal else "episode_cap"
    return PolicyRollout(rollout, env.plan, checked_evaluation(env), steps, end, ties, zeros, unseen)


def train(mine: Mine, precedences: Precedences, problem: ProblemParameters, agent: AgentParameters,
          training: TrainingParameters, evaluation: EvaluationParameters, seed: int) -> tuple[TrainingResult, QLearning]:
    """Receives the mine and its rules only; a reference solution never reaches the agent."""
    started = perf_counter()
    env = MineEnv(mine, problem, precedences, training)
    model = QLearning(agent, seed)
    visited, pairs = set(), Counter()
    history = []
    best = None  # (npv, plan, episode, step)
    transitions = 0
    while transitions < training.transitions:
        episode = len(history)
        state = env.reset()
        total = 0.0
        steps = 0
        while not state.terminal and steps < training.max_steps and transitions < training.transitions:
            actions = env.available_actions()
            action = model.select_action(state, actions)
            key = model.state_key(state)
            visited.add(key)
            pairs[key, action] += 1
            following, reward, terminal = env.step(action)
            model.update(state, action, reward, following, env.available_actions(), terminal)
            transitions += 1
            steps += 1
            if len(following.extracted) != len(state.extracted):
                total += reward
                if best is None or total > best[0] and not _same(total, best[0]):
                    best = total, env.plan, episode, steps
            state = following
        final = checked_evaluation(env)
        if not _same(total, final.npv):
            raise RuntimeError("Las recompensas del episodio no coinciden con el VAN de su plan")
        end = env.termination_reason if state.terminal else (
            "budget" if transitions == training.transitions else "episode_cap")
        history.append(EpisodeRecord(episode, steps, final.npv, end))

    violations = q_bound_violations(model, mine)
    if violations:
        raise RuntimeError(f"Q supera el valor físicamente posible: {violations[:3]}")
    best_evaluation = checked_evaluation(env, best[1])
    if not _same(best_evaluation.npv, best[0]):
        raise RuntimeError("El mejor plan observado no coincide con su VAN verificado")
    observed = ObservedPlan(best[1], best_evaluation, best[2], best[3])
    policy = tuple(_policy_rollout(env, model, visited, r) for r in range(evaluation.policy_rollouts))
    result = TrainingResult(seed, observed, policy, tuple(history), transitions, len(visited), len(pairs),
                            sum(count == 1 for count in pairs.values()), perf_counter() - started)
    return result, model
