"""Tabular Q-learning over the spatial mine environment: the baseline.

Everything it does is the default of TabularQ (src/models/tabular.py): one Q-learning
update per real transition, no planning. Its hyperparameters are those of TabularParams.
"""

from dataclasses import dataclass

from src.models.register import register
from src.models.tabular import PositionKey, TabularParams, TabularQ  # noqa: F401  (PositionKey: public name)


@dataclass(frozen=True)
class QLearningParams(TabularParams):
    """Hyperparameters of tabular Q-learning: exactly those of TabularParams."""


@register("q_learning", params=QLearningParams)
class QLearningEstimator(TabularQ):
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
