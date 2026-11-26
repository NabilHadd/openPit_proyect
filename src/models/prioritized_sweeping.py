"""Prioritized sweeping (Sutton y Barto, sección 8.4) over the spatial mine environment.

Each real transition is stored in a learned model, as in Dyna-Q, plus its predecessor:
predecessors[key(next_state)][(key(state), action)] = state. Instead of updating the real
pair directly, its priority P = |td_error| (how much its update would change Q) is computed
and, if P > theta, it is queued. Then up to `planning_steps` pairs leave the queue, highest
priority first; each one is updated from the model and every predecessor of its state whose
priority exceeds theta is queued. Values thus propagate backwards from where they changed.

- planning_steps = 0: there is no planning loop, so the real pair is updated directly and
  the fit is identical to Q-learning.
- planning_updates counts the updates made from the queue (the real pair's included, since
  with n > 0 it is updated there); queue_pushes counts the insertions in the queue.
- Reproducibility: the heap holds (-priority, counter, key, action), the counter breaking
  ties by order of arrival, so two states are never compared; predecessors are dicts
  (insertion order), never sets, whose order of str changes between processes. A dict of
  the queued priority of each pair discards stale heap entries: a pair is queued again only
  with a higher priority, and an entry whose priority is no longer the queued one is skipped
  without consuming a planning step.
- Like Dyna-Q it requires state: physical (the position key is not Markov).
"""

from dataclasses import dataclass
import heapq
import math
from typing import ClassVar

from configs import integer
from src.models.register import register
from src.models.tabular import TabularParams, TabularQ


@dataclass(frozen=True)
class PrioritizedSweepingParams(TabularParams):
    """TabularParams plus planning_steps (n >= 0: updates from the queue after each real
    transition) and theta (>= 0: minimum priority, in VAN units, to enter the queue)."""
    planning_steps: int = 10
    theta: float = 0.001

    SMOKE: ClassVar[dict] = {"transitions": 500, "policy_rollouts": 2, "planning_steps": 3}

    def __post_init__(self):
        super().__post_init__()
        if self.state != "physical":
            raise ValueError("prioritized_sweeping requiere state: physical (la clave position no es markoviana)")
        integer(self.planning_steps, "planning_steps", 0)
        if isinstance(self.theta, bool) or not isinstance(self.theta, (int, float)) or not math.isfinite(self.theta) \
                or self.theta < 0:
            raise ValueError("theta debe ser un número finito >= 0")


@register("prioritized_sweeping", params=PrioritizedSweepingParams)
class PrioritizedSweepingEstimator(TabularQ):
    name = "spatial-cone-prioritized-sweeping"

    def __init__(self, state="physical", alpha=1.0, gamma=1.0, epsilon=0.3, epsilon_schedule="constant",
                 transitions=100_000, policy_rollouts=20, planning_steps=10, theta=0.001, random_state=0):
        self.state = state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.transitions = transitions
        self.policy_rollouts = policy_rollouts
        self.planning_steps = planning_steps
        self.theta = theta
        self.random_state = random_state

    def _reset_learning(self):
        super()._reset_learning()
        self.model_: dict = {}         # (key, action) -> (state, reward, next_state, next_actions, terminal)
        self.predecessors_: dict = {}  # key(next_state) -> {(key(state), action): state}, insertion order
        self._heap: list = []          # (-priority, counter, key, action)
        self._queued: dict = {}        # (key, action) -> priority of its live heap entry
        self._arrivals = 0
        self.planning_updates_ = 0
        self.queue_pushes_ = 0

    def _push(self, key, action, priority: float):
        if priority <= self.theta or priority <= self._queued.get((key, action), -1.0):
            return
        self._queued[key, action] = priority
        heapq.heappush(self._heap, (-priority, self._arrivals, key, action))
        self._arrivals += 1
        self.queue_pushes_ += 1

    def _priority(self, key, action) -> float:
        state, reward, next_state, next_actions, terminal = self.model_[key, action]
        return abs(self.td_error(state, action, reward, next_state, next_actions, terminal))

    def observe(self, state, action, reward, next_state, next_actions, terminal):
        key = self.state_key(state)
        self.model_[key, action] = (state, reward, next_state, next_actions, terminal)
        self.predecessors_.setdefault(self.state_key(next_state), {})[key, action] = state
        if self.planning_steps == 0:
            self.update(state, action, reward, next_state, next_actions, terminal)
            return
        self._push(key, action, self._priority(key, action))
        done = 0
        while done < self.planning_steps and self._heap:
            negative, _, pair_key, pair_action = heapq.heappop(self._heap)
            if self._queued.get((pair_key, pair_action)) != -negative:
                continue  # stale: the pair was queued again with a higher priority
            del self._queued[pair_key, pair_action]
            lived, lived_reward, lived_next, lived_actions, lived_terminal = self.model_[pair_key, pair_action]
            self.update(lived, pair_action, lived_reward, lived_next, lived_actions, lived_terminal)
            self.planning_updates_ += 1
            done += 1
            for (before_key, before_action), _ in self.predecessors_.get(pair_key, {}).items():
                self._push(before_key, before_action, self._priority(before_key, before_action))

    def _extra_counters(self) -> dict[str, int]:
        return {"planning_updates": self.planning_updates_, "queue_pushes": self.queue_pushes_}

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["planning"] = {"planning_steps": self.planning_steps, "theta": self.theta,
                              "planning_updates": self.planning_updates_, "queue_pushes": self.queue_pushes_,
                              "queue_left": len(self._queued), "model_pairs": len(self.model_),
                              "states_with_predecessors": len(self.predecessors_)}
        return result
