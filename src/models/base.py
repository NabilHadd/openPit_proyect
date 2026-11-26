"""Common contract of every model: a scikit-learn estimator that receives a MineInstance and
delivers a Plan. How it learns or searches is each model's own business.

Rules every model follows (scikit-learn conventions):
- __init__ takes every hyperparameter, plus random_state, as a named argument and only
  stores it under the same name: no validation or conversion there, or clone and
  set_params break. Validation happens in fit (`_check_params`, with the params
  dataclass that @register attaches as `Params`).
- What is learned is created in fit and its name ends in "_" (check_is_fitted relies on it).
  fit stores the instance as `instance_`.
- A data transformation the model needs happens inside its own fit/predict; predict
  returns a Plan with the original block IDs, and score evaluates it on the original instance.

Cost of training, the same way for every model, so they can be compared:
- `counters_`: dict name -> int of what fit spent (e.g. transitions, simulator_calls,
  updates). `cost_unit` (class attribute) names the counter the model's budget is measured in.
- `trace_`: tuple of (counters, npv), one entry each time the best plan found so far
  improved, with a copy of every counter at that moment. Its last npv is the score. The
  runner reads it after training to report when the best plan (or the reference) was first
  reached, in cost_unit, and keeps the other counters available.

`deterministic` (class attribute, default False): True for a model whose result does not
depend on random_state (e.g. a search without random tie-breaking). The experiments mode
runs it with the first seed only, so its summary has n=1 and no confidence interval.

ABC makes the abstract methods mandatory: a model that leaves one unimplemented cannot be
instantiated. save_artifacts and compare_reference have defaults.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from configs import integer
from src.data.instance import MineInstance
from src.evaluation.metrics import Plan, checked_evaluation, same_value
from src.evaluation.reporter import write_plan_csv


class MineEstimator(BaseEstimator, ABC):
    cost_unit: ClassVar[str] = "transitions"
    deterministic: ClassVar[bool] = False
    Params: ClassVar[type | None] = None  # set by @register (src/models/register.py)

    @abstractmethod
    def fit(self, instance: MineInstance) -> "MineEstimator":
        """Learn on the instance and return self."""

    @abstractmethod
    def predict(self, instance: MineInstance) -> Plan:
        """The plan the model delivers for the instance."""

    @abstractmethod
    def diagnostics(self) -> dict:
        """Model-specific figures of the last fit for report.json (JSON-serializable)."""

    def save_artifacts(self, folder: Path) -> None:
        """Write the model's own files of the last fit into folder. Default: plan.csv, the
        delivered plan."""
        check_is_fitted(self, "instance_")
        write_plan_csv(Path(folder) / "plan.csv", self.predict(self.instance_))

    def compare_reference(self, reference_npv: float) -> dict[str, int]:
        """Counts of the model's own results that match the reference VAN ({} if it has none).

        Called only after every seed has been trained and saved, so the reference never
        reaches training. Every value is a count, so totals over seeds are plain sums; it
        must not change what the model learned. The delivered plan itself is compared by
        the runner through score."""
        check_is_fitted(self, "instance_")
        return {}

    def score(self, instance: MineInstance) -> float:
        """Verified VAN of the plan of predict; higher is better, as in scikit-learn."""
        return checked_evaluation(instance, self.predict(instance)).npv

    # --- helpers for subclasses -----------------------------------------------------------

    def _check_params(self):
        """The params dataclass holds the rules of the hyperparameters; they are not repeated."""
        if self.Params is None:
            raise TypeError(f"{type(self).__name__} no es un modelo registrado: use una subclase "
                            "registrada con @register (src/models/register.py), p. ej. QLearningEstimator")
        params = self.get_params()
        if "random_state" not in params:
            raise TypeError(f"{type(self).__name__}.__init__ debe recibir random_state")
        integer(params.pop("random_state"), "random_state", 0)
        return self.Params(**params)

    def _check_instance(self, instance: MineInstance):
        """For models whose plan belongs to the instance of fit."""
        check_is_fitted(self, "instance_")
        if instance != self.instance_:
            raise ValueError(f"{type(self).__name__} entrega el plan encontrado al entrenar: "
                             "predict requiere la misma instancia que fit")

    def _start_trace(self):
        self._trace = []

    def _offer(self, counters: dict[str, int], npv: float) -> bool:
        """Record a new best VAN with a copy of the counters at that moment; returns whether
        it improved (ties do not)."""
        if self._trace and (npv <= self._trace[-1][1] or same_value(npv, self._trace[-1][1])):
            return False
        self._trace.append((dict(counters), npv))
        return True
