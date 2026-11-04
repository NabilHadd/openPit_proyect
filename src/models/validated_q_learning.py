"""Opt-in validation policy; preserves the historical controls and their RNGs."""

import random

from src.models.experimental_q_learning import ControlQLearning


class ValidationQLearning(ControlQLearning):
    name = "validation-exact-cone-q-learning"

    def __init__(self, parameters, seed: int, *, evaluation_ties: str = "uniform"):
        if parameters.gamma != 1:
            raise ValueError("La validación VAN requiere gamma=1, sin descuento artificial por acción")
        if evaluation_ties not in {"uniform", "ordered"}:
            raise ValueError("evaluation_ties debe ser uniform u ordered")
        super().__init__(parameters, seed, "physical")
        self.name = type(self).name
        self.evaluation_ties = evaluation_ties
        self.reset_evaluation(0)

    def reset_evaluation(self, rollout: int):
        self.evaluation_rng = random.Random(f"{self.seed}:evaluation:{rollout}")

    def select_action(self, state, actions, *, explore: bool = True):
        if explore:
            return super().select_action(state, actions, explore=True)
        if not actions:
            raise ValueError("No hay acciones disponibles")
        values = {a: self.value(state, a) for a in actions}
        ties = tuple(a for a in actions if values[a] == max(values.values()))
        return self.evaluation_rng.choice(ties) if self.evaluation_ties == "uniform" else ties[0]
