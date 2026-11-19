"""Tabular Q-learning over the spatial mine environment, as a MineEstimator.

- fit runs epsilon-greedy episodes until `transitions` steps have been taken. Epsilon is
  recomputed at every transition with the schedule of configs/q_learning.yml
  (src/models/exploration.py): it starts at `epsilon` and reaches 0 when the budget runs out.
- An episode ends for real (no future value) on stop, all reachable blocks extracted or,
  in finite mode, the step limit. Reaching max_steps in external mode, or running out of
  budget, only interrupts it: the last update still counts the future value.
- Best observed: after every extraction the extracted set is a feasible plan; the one of
  highest VAN seen during training is the plan predict delivers.
- Policy: after training, the agent builds plans without exploration (always the action
  of highest Q); ties are broken at random, so several rollouts measure consistency. They
  are diagnostics, not the delivered plan.
"""

from collections import Counter
from collections.abc import Hashable
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from statistics import mean

from sklearn.utils.validation import check_is_fitted

from configs import integer
from configs.q_learning_settings import QLearningParams
from src.data.dataset import Mine
from src.data.instance import MineInstance
from src.envs.mine_env import STOP, MineEnv, State
from src.evaluation.metrics import Evaluation, Plan, checked_evaluation, same_value
from src.evaluation.plots import plot_history
from src.evaluation.reporter import write_json, write_plan_csv
from src.models.base import MineEstimator
from src.models.exploration import epsilon_at


@dataclass(frozen=True)
class PositionKey:
    """Compact key: forgets what was extracted. steps_left keeps the task clock in finite mode
    (it is always 0 in external mode), so a step limit never contradicts the key."""
    position: int | None
    period: int
    steps_left: int


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


class QLearningEstimator(MineEstimator):
    """Hyperparameters: the fields of QLearningParams (same names and defaults) plus
    random_state, the seed of training. `state` is what the Q table remembers of a state."""
    name = "spatial-cone-q-learning"

    def __init__(self, state="physical", alpha=1.0, gamma=1.0, epsilon=0.3, epsilon_schedule="constant",
                 transitions=100_000, policy_rollouts=20, random_state=0):
        self.state = state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.transitions = transitions
        self.policy_rollouts = policy_rollouts
        self.random_state = random_state

    # --- contract -------------------------------------------------------------------------

    def fit(self, instance: MineInstance) -> "QLearningEstimator":
        """Receives the instance only; a reference solution never reaches the model."""
        self._check_params()
        env = MineEnv(instance.mine, instance.problem, instance.precedences, instance.environment)
        self._reset_learning()
        visited, pairs = set(), Counter()
        history = []
        best = None  # (npv, plan, episode, step)
        transitions = 0
        while transitions < self.transitions:
            episode = len(history)
            state = env.reset()
            total = 0.0
            steps = 0
            while not state.terminal and steps < instance.environment.max_steps and transitions < self.transitions:
                actions = env.available_actions()
                epsilon = epsilon_at(self.epsilon_schedule, self.epsilon, transitions, self.transitions)
                action = self.select_action(state, actions, epsilon)
                key = self.state_key(state)
                visited.add(key)
                pairs[key, action] += 1
                following, reward, terminal = env.step(action)
                self.update(state, action, reward, following, env.available_actions(), terminal)
                transitions += 1
                steps += 1
                if len(following.extracted) != len(state.extracted):
                    total += reward
                    if best is None or total > best[0] and not same_value(total, best[0]):
                        best = total, env.plan, episode, steps
                state = following
            final = checked_evaluation(instance, env.plan)
            if not same_value(total, final.npv):
                raise RuntimeError("Las recompensas del episodio no coinciden con el VAN de su plan")
            end = env.termination_reason if state.terminal else (
                "budget" if transitions == self.transitions else "episode_cap")
            history.append(EpisodeRecord(episode, steps, final.npv, end))

        violations = q_bound_violations(self, instance.mine)
        if violations:
            raise RuntimeError(f"Q supera el valor físicamente posible: {violations[:3]}")
        best_evaluation = checked_evaluation(instance, best[1])
        if not same_value(best_evaluation.npv, best[0]):
            raise RuntimeError("El mejor plan observado no coincide con su VAN verificado")
        self.instance_ = instance
        self.history_ = tuple(history)
        self.best_observed_ = ObservedPlan(best[1], best_evaluation, best[2], best[3])
        self.policy_ = tuple(self._policy_rollout(env, instance, visited, r) for r in range(self.policy_rollouts))
        self.n_transitions_ = transitions
        self.unique_states_ = len(visited)
        self.unique_pairs_ = len(pairs)
        self.pairs_visited_once_ = sum(count == 1 for count in pairs.values())
        return self

    def predict(self, instance: MineInstance) -> Plan:
        """The best plan observed while training. It belongs to the instance of fit, so
        any other instance is rejected."""
        check_is_fitted(self, "best_observed_")
        if instance != self.instance_:
            raise ValueError("Q-learning entrega el mejor plan observado al entrenar: "
                             "predict requiere la misma instancia que fit")
        return self.best_observed_.plan

    def diagnostics(self) -> dict:
        check_is_fitted(self, "best_observed_")
        decisions = sum(r.steps for r in self.policy_)
        values = [r.evaluation.npv for r in self.policy_]
        best = self.best_observed_
        return {
            "training": {"transitions": self.n_transitions_, "episodes": len(self.history_),
                         "episode_ends": dict(sorted(Counter(e.end for e in self.history_).items())),
                         "unique_states": self.unique_states_, "unique_pairs": self.unique_pairs_,
                         "pairs_visited_once": self.pairs_visited_once_ / self.unique_pairs_},
            "best_observed": {"npv": best.evaluation.npv, "episode": best.episode, "step": best.step,
                              "blocks": len(best.plan.extractions)},
            "policy": {"rollouts": len(values), "npv_mean": mean(values), "npv_min": min(values),
                       "npv_max": max(values),
                       "tie_decisions": sum(r.tie_decisions for r in self.policy_) / decisions,
                       "all_zero_decisions": sum(r.all_zero_decisions for r in self.policy_) / decisions,
                       "unseen_state_decisions": sum(r.unseen_state_decisions for r in self.policy_) / decisions,
                       "episode_cap_rollouts": sum(r.end == "episode_cap" for r in self.policy_)},
        }

    def save_artifacts(self, folder: Path) -> None:
        """best_observed.csv (the delivered plan), policy_best.csv (best policy rollout),
        training.json (episodes and rollouts), q-table.json and training.svg (VAN of every
        training episode and best so far)."""
        check_is_fitted(self, "best_observed_")
        folder = Path(folder)
        write_plan_csv(folder / "best_observed.csv", self.best_observed_.plan)
        best_rollout = max(self.policy_, key=lambda r: r.evaluation.npv)  # first one among ties
        write_plan_csv(folder / "policy_best.csv", best_rollout.plan)  # already verified in its rollout
        write_json(folder / "training.json", {
            "episodes": [asdict(e) for e in self.history_],
            "policy_rollouts": [{k: v for k, v in asdict(r).items() if k != "plan"} for r in self.policy_]})
        write_json(folder / "q-table.json", q_table_payload(self))
        plot_history([e.npv for e in self.history_], folder / "training.svg")

    def compare_reference(self, reference_npv: float) -> dict[str, int]:
        """Policy rollouts whose VAN matches the reference; policy_always_matches is 1 when
        all of them do, so its total over seeds counts the seeds whose policy always matches."""
        check_is_fitted(self, "best_observed_")
        matches = sum(same_value(r.evaluation.npv, reference_npv) for r in self.policy_)
        return {"policy_matches": matches, "policy_rollouts": len(self.policy_),
                "policy_always_matches": int(matches == len(self.policy_))}

    # --- Q-learning ------------------------------------------------------------------------

    def _check_params(self):
        """QLearningParams holds the rules of the hyperparameters; they are not repeated here."""
        params = self.get_params()
        integer(params.pop("random_state"), "random_state", 0)
        QLearningParams(**params)

    def _reset_learning(self):
        """Empty Q table and training RNG; fit starts with them. Tests also use it to drive
        select_action and update without training."""
        self.rng_ = random.Random(self.random_state)
        self.q_: dict[tuple[Hashable, str], float] = {}

    def state_key(self, state: State) -> Hashable:
        if self.state == "position":
            return PositionKey(state.position, state.period, state.steps_left)
        return state

    def value(self, state: State, action: str) -> float:
        return self.q_.get((self.state_key(state), action), 0.0)

    def best_actions(self, state: State, actions: tuple[str, ...]) -> tuple[str, ...]:
        """Every action tied at the maximum Q value."""
        if not actions:
            raise ValueError("No hay acciones disponibles")
        values = [self.value(state, a) for a in actions]
        best = max(values)
        return tuple(a for a, v in zip(actions, values) if v == best)

    def select_action(self, state: State, actions: tuple[str, ...], epsilon: float) -> str:
        """Training choice: epsilon-greedy, ties broken at random with the training RNG.

        epsilon is recomputed by fit at every transition (decreasing schedule). A random
        exploration move never chooses STOP: stopping at random cut episodes short (~13
        steps) and deep blocks were rarely reached. STOP is still chosen when it is the
        action of highest Q."""
        if not actions:
            raise ValueError("No hay acciones disponibles")
        if self.rng_.random() < epsilon:
            moves = tuple(a for a in actions if a != STOP)
            return self.rng_.choice(moves or actions)
        return self.rng_.choice(self.best_actions(state, actions))

    def update(self, state: State, action: str, reward: float, next_state: State,
               next_actions: tuple[str, ...], terminal: bool) -> float:
        """terminal=True only when the task really ends; an interruption still bootstraps."""
        if not terminal and not next_actions:
            raise ValueError("Transición no terminal sin acciones siguientes")
        bootstrap = 0.0 if terminal else max(self.value(next_state, a) for a in next_actions)
        previous = self.value(state, action)
        updated = previous + self.alpha * (reward + self.gamma * bootstrap - previous)
        self.q_[self.state_key(state), action] = updated
        return updated

    def _policy_rollout(self, env: MineEnv, instance: MineInstance, visited: set, rollout: int) -> PolicyRollout:
        rng = random.Random(f"{self.random_state}:policy:{rollout}")  # never touches the training RNG
        state = env.reset()
        steps = ties = zeros = unseen = 0
        while not state.terminal and steps < instance.environment.max_steps:
            actions = env.available_actions()
            best = self.best_actions(state, actions)
            ties += len(best) > 1
            zeros += all(self.value(state, a) == 0 for a in actions)
            unseen += self.state_key(state) not in visited
            state, _, _ = env.step(rng.choice(best))
            steps += 1
        end = env.termination_reason if state.terminal else "episode_cap"
        return PolicyRollout(rollout, env.plan, checked_evaluation(instance, env.plan), steps, end, ties, zeros, unseen)


def q_bound_violations(model: QLearningEstimator, mine: Mine) -> list[dict]:
    """No future return can exceed the positive value still in the ground (gamma <= 1).

    Only for the physical key. The position key forgets what was extracted, so it can
    credit the same block again and again: its Q is an approximation, not a true value."""
    if model.state != "physical":
        return []
    positive = {b.id: b.value for b in mine.blocks if b.value > 0}
    violations = []
    for (key, action), value in model.q_.items():
        extracted = getattr(key, "extracted", frozenset())
        bound = math.fsum(v for i, v in positive.items() if i not in extracted)
        if not math.isfinite(value) or value - bound > 1e-6:
            violations.append({"action": action, "q": value, "upper_bound": bound})
    return violations


def q_table_payload(model: QLearningEstimator) -> dict:
    def key_payload(key):
        payload = asdict(key)
        if "extracted" in payload:
            payload["extracted"] = sorted(payload["extracted"])
        return payload
    return {"model": model.name, "entries": [
        {"state": key_payload(state), "action": action, "q": value}
        for (state, action), value in model.q_.items()
    ]}
