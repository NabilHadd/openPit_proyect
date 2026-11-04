"""Transition-budget Q-learning with explicit task endings and external cutoffs."""

from collections import Counter
from dataclasses import dataclass
import math
from time import perf_counter

from src.config import LearningParameters, ProblemParameters, SpatialParameters
from src.data.dataset import Mine
from src.envs.spatial_cone_env import DIRECTIONS, SpatialConeEnv
from src.evaluation.experiment_metrics import mining_context
from src.models.validated_q_learning import ValidationQLearning
from src.training.trainer import ObservedPath, PathObserver, checked_evaluation


@dataclass(frozen=True)
class ValidationBudget:
    transitions: int = 10000
    evaluation_rollouts: int = 20
    evaluation_ties: str = "uniform"

    def __post_init__(self):
        for value in (self.transitions, self.evaluation_rollouts):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Los presupuestos de transiciones y evaluaciones deben ser enteros positivos")
        if self.evaluation_ties not in {"uniform", "ordered"}:
            raise ValueError("evaluation_ties debe ser uniform u ordered")


class DecisionCoverage:
    def __init__(self, model, visits, pair_visits, contexts):
        self.model, self.visits, self.pair_visits, self.contexts = model, visits, pair_visits, contexts
        self.counts = Counter()
        self.unique_contexts = set()
        self.unseen_contexts = set()
        self.segment_contexts = set()
        self.first_unknown_step = None
        self.first_all_zero_step = None
        self.trailing_no_extraction = 0

    def before(self, state, actions, action):
        key = self.model.state_key(state)
        values = [self.model.value(state, a) for a in actions]
        unknown = key not in self.visits
        zero = all(v == 0 for v in values)
        context = mining_context(state)
        self.unique_contexts.add(context)
        if context not in self.contexts:
            self.unseen_contexts.add(context)
        self.counts["transitions"] += 1
        self.counts["all_zero_q_decisions"] += zero
        self.counts["unseen_q_state_decisions"] += unknown
        self.counts["unseen_state_action_decisions"] += (key, action) not in self.pair_visits
        self.counts["unknown_only_because_of_clock"] += unknown and context in self.contexts
        self.counts["max_q_tie_decisions"] += values.count(max(values)) > 1
        self.counts["stop_decisions"] += action == "stop"
        if unknown and self.first_unknown_step is None:
            self.first_unknown_step = self.counts["transitions"]
        if zero and self.first_all_zero_step is None:
            self.first_all_zero_step = self.counts["transitions"]

    def after(self, state, following, action):
        context = mining_context(following)
        if following.extracted != state.extracted:
            self.trailing_no_extraction = 0
            self.segment_contexts = {context}
        else:
            self.counts["no_extraction_transitions"] += 1
            self.counts["zero_extraction_moves"] += action in DIRECTIONS
            self.trailing_no_extraction += 1
            if action in DIRECTIONS:
                self.segment_contexts.add(mining_context(state))
                self.counts["navigation_context_returns"] += context in self.segment_contexts
                self.segment_contexts.add(context)
            else:
                self.segment_contexts = {context}

    def payload(self):
        return {**dict(self.counts), "unique_mining_contexts": len(self.unique_contexts),
                "unique_unseen_mining_contexts": len(self.unseen_contexts),
                "first_unknown_step": self.first_unknown_step, "first_all_zero_step": self.first_all_zero_step,
                "trailing_no_extraction_transitions": self.trailing_no_extraction}


@dataclass(frozen=True)
class PolicyRollout:
    path: ObservedPath
    best_prefix: ObservedPath
    terminated: bool
    truncated: bool
    reason: str
    decisions: dict


@dataclass(frozen=True)
class ValidationRun:
    seed: int
    model: ValidationQLearning
    best_observed: ObservedPath
    best_completed: ObservedPath | None
    training: dict
    policy: tuple[PolicyRollout, ...]
    generation_seconds: float


def q_bound_check(model: ValidationQLearning, mine: Mine) -> dict:
    """No future return can exceed the remaining positive block values (gamma=1)."""
    violations = []
    maximum_excess = 0.0
    for (state, action), value in model.q.items():
        bound = math.fsum(b.value for b in mine.blocks if b.value > 0 and b.id not in state.extracted)
        excess = value - bound
        if not math.isfinite(value) or excess > 1e-6:
            if len(violations) < 10:
                violations.append({"position": state.position, "action": action, "q": value, "upper_bound": bound})
            maximum_excess = max(maximum_excess, excess)
    return {"passed": not violations, "examples": violations, "maximum_excess": maximum_excess}


def train_transition_budget(mine, rules, spatial: SpatialParameters, learning: LearningParameters,
                            seed: int, budget: ValidationBudget) -> ValidationRun:
    """No target/reference input. Budget cuts bootstrap; true task terminals do not.

    Finite mode retains the task clock in Q. External mode has a stationary clock;
    the collector owns the episode cap and bootstraps before resetting.
    """
    if not spatial.construction_up:
        raise ValueError("La validación requiere seis direcciones en ambas fases")
    started = perf_counter()
    env = SpatialConeEnv(mine, ProblemParameters(), rules, spatial)
    model = ValidationQLearning(learning, seed, evaluation_ties=budget.evaluation_ties)
    visits, pairs, contexts = Counter(), Counter(), set()
    history = []
    best_observed = best_completed = None
    transitions = 0
    totals = Counter()
    while transitions < budget.transitions:
        episode = len(history)
        state = env.reset(training=True)
        observer = PathObserver(env, episode, "learning")
        coverage = DecisionCoverage(model, visits, pairs, contexts)
        reward_sum = 0.0
        steps = 0
        while not state.terminal and steps < spatial.max_steps and transitions < budget.transitions:
            actions = env.available_actions()
            action = model.select_action(state, actions, explore=True)
            coverage.before(state, actions, action)
            key = model.state_key(state)
            visits[key] += 1
            pairs[key, action] += 1
            contexts.add(mining_context(state))
            following, reward, terminal = env.step(action)
            # Budget exhaustion is deliberately NOT passed as terminal to update.
            model.update(state, action, reward, following, env.available_actions(), terminal)
            coverage.after(state, following, action)
            reward_sum += reward
            transitions += 1
            steps += 1
            observer.observe(steps, action)
            state = following
        evaluation = checked_evaluation(env)
        if not math.isclose(reward_sum, evaluation.npv, rel_tol=1e-12, abs_tol=1e-6):
            raise RuntimeError("Las recompensas no coinciden con el VAN del plan")
        path = ObservedPath(env.plan, evaluation, episode, "learning", steps, env.action_history)
        if best_observed is None or observer.best.evaluation.npv > best_observed.evaluation.npv:
            best_observed = observer.best
        if state.terminal and (best_completed is None or evaluation.npv > best_completed.evaluation.npv):
            best_completed = path
        totals.update(coverage.counts)
        history.append({"episode": episode, "steps": steps, "npv": evaluation.npv,
                        "best_prefix_npv": observer.best.evaluation.npv,
                        "terminated": state.terminal, "truncated": not state.terminal,
                        "reason": (env.termination_reason if state.terminal else
                                   "transition_budget" if transitions == budget.transitions else "external_episode_limit"),
                        "cumulative_transitions": transitions, "decisions": coverage.payload()})

    learning_seconds = perf_counter() - started
    policy = []
    for rollout in range(budget.evaluation_rollouts):
        model.reset_evaluation(rollout)
        state = env.reset(training=False)
        observer = PathObserver(env, None, "construction")
        coverage = DecisionCoverage(model, visits, pairs, contexts)
        steps = 0
        while not state.terminal and steps < spatial.max_steps:
            actions = env.available_actions()
            action = model.select_action(state, actions, explore=False)
            coverage.before(state, actions, action)
            following, _, _ = env.step(action)
            coverage.after(state, following, action)
            steps += 1
            observer.observe(steps, action)
            state = following
        path = ObservedPath(env.plan, checked_evaluation(env), None, "construction", steps, env.action_history)
        policy.append(PolicyRollout(path, observer.best, state.terminal, not state.terminal,
                                    env.termination_reason if state.terminal else "external_episode_limit", coverage.payload()))
    training = {"transitions": transitions, "episodes": len(history), "decisions": dict(totals),
                "termination_counts": dict(Counter(h["reason"] for h in history)),
                "mean_episode_steps": transitions / len(history),
                "unique_q_states": len(visits), "unique_state_action_pairs": len(pairs),
                "q_state_reuse_fraction": (transitions - len(visits)) / transitions,
                "state_action_reuse_fraction": (transitions - len(pairs)) / transitions,
                "q_states_visited_once_fraction": sum(v == 1 for v in visits.values()) / len(visits),
                "state_action_pairs_visited_once_fraction": sum(v == 1 for v in pairs.values()) / len(pairs),
                "unique_mining_contexts": len(contexts), "q_entries": len(model.q),
                "q_bounds": q_bound_check(model, mine), "history": history,
                "learning_seconds": learning_seconds}
    if not training["q_bounds"]["passed"]:
        raise RuntimeError("Q supera el retorno económico físicamente posible")
    return ValidationRun(seed, model, best_observed, best_completed, training, tuple(policy), perf_counter() - started)
