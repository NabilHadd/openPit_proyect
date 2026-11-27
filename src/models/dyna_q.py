"""Dyna-Q (Sutton y Barto, sección 8.2) over the spatial mine environment.

Each real transition does what Q-learning does (one update) and is stored in a LEARNED
model: a dict (state_key, action) -> (state, reward, next_state, next_actions, terminal) of
the transitions actually lived. Then `planning_steps` simulated updates replay pairs drawn
uniformly from that model. Planning never calls the simulator, so the budget stays
`transitions` (real steps) and the simulated updates are counted as planning_updates.

The environment is deterministic, so the last outcome of a pair is its only outcome when
the key is the physical state. The position key is not Markov (different real states share
it), so a model indexed by it would replay outcomes of other states: Dyna-Q requires
state: physical.

The draws use their own RNG, random.Random(f"{seed}:planning"), so the exploration RNG is
the one Q-learning would use; with planning_steps = 0 the fit is identical to Q-learning.
"""

from dataclasses import dataclass
import random
from typing import ClassVar

from configs import integer
from src.models.register import register
from src.models.tabular import TabularParams, TabularQ


@dataclass(frozen=True)
class DynaQParams(TabularParams):
    """TabularParams plus planning_steps: simulated updates after each real transition (n >= 0)."""
    planning_steps: int = 10

    SMOKE: ClassVar[dict] = {"transitions": 500, "policy_rollouts": 2, "planning_steps": 3}

    def __post_init__(self):
        super().__post_init__()
        if self.state != "physical":
            raise ValueError("dyna_q requiere state: physical (la clave position no es markoviana: el modelo "
                             "aprendido repetiría resultados de otros estados)")
        integer(self.planning_steps, "planning_steps", 0)


@register("dyna_q", params=DynaQParams)
class DynaQEstimator(TabularQ):
    name = "spatial-cone-dyna-q"

    def __init__(self, state="physical", alpha=1.0, gamma=1.0, epsilon=0.3, epsilon_schedule="constant",
                 transitions=100_000, policy_rollouts=20, planning_steps=10, initial_q="zero",
                 random_state=0):
        self.state = state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.transitions = transitions
        self.policy_rollouts = policy_rollouts
        self.initial_q = initial_q
        self.planning_steps = planning_steps
        self.random_state = random_state

    def _reset_learning(self):
        super()._reset_learning()
        self.model_: dict = {}        # (state_key, action) -> (state, reward, next_state, next_actions, terminal)
        self._model_keys: list = []   # same keys in insertion order, for uniform draws
        self._planning_rng = random.Random(f"{self.random_state}:planning")
        self.planning_updates_ = 0

    def observe(self, state, action, reward, next_state, next_actions, terminal):
        self.update(state, action, reward, next_state, next_actions, terminal)
        key = self.state_key(state), action
        if key not in self.model_:
            self._model_keys.append(key)
        self.model_[key] = (state, reward, next_state, next_actions, terminal)
        for _ in range(self.planning_steps):
            chosen = self._model_keys[self._planning_rng.randrange(len(self._model_keys))]
            lived, lived_reward, lived_next, lived_actions, lived_terminal = self.model_[chosen]
            self.update(lived, chosen[1], lived_reward, lived_next, lived_actions, lived_terminal)
            self.planning_updates_ += 1

    def _extra_counters(self) -> dict[str, int]:
        return {"planning_updates": self.planning_updates_}

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["planning"] = {"planning_steps": self.planning_steps, "planning_updates": self.planning_updates_,
                              "model_pairs": len(self.model_)}
        return result
