"""Common contract of every model: a scikit-learn estimator that receives a MineInstance and
delivers a Plan. How it learns is each model's own business.

Rules every model follows (scikit-learn conventions):
- __init__ takes every hyperparameter, plus random_state, as a named argument and only
  stores it under the same name: no validation or conversion there, or clone and
  set_params break. Validation happens in fit.
- What is learned is created in fit and its name ends in "_" (check_is_fitted relies on it).
- A data transformation the model needs happens inside its own fit/predict; predict
  returns a Plan with the original block IDs, and score evaluates it on the original instance.

ABC makes the abstract methods mandatory: a model that leaves one unimplemented cannot be
instantiated.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.base import BaseEstimator

from src.data.instance import MineInstance
from src.evaluation.metrics import Plan, checked_evaluation


class MineEstimator(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, instance: MineInstance) -> "MineEstimator":
        """Learn on the instance and return self."""

    @abstractmethod
    def predict(self, instance: MineInstance) -> Plan:
        """The plan the model delivers for the instance."""

    @abstractmethod
    def diagnostics(self) -> dict:
        """Model-specific figures of the last fit for report.json (JSON-serializable)."""

    @abstractmethod
    def save_artifacts(self, folder: Path) -> None:
        """Write the model's own files of the last fit (learned state, history, plots...) into folder."""

    @abstractmethod
    def compare_reference(self, reference_npv: float) -> dict[str, int]:
        """Counts of the model's own results that match the reference VAN ({} if it has none).

        Called only after every seed has been trained and saved, so the reference never
        reaches training. Every value is a count, so totals over seeds are plain sums; it
        must not change what the model learned. The delivered plan itself is compared by
        the runner through score."""

    def score(self, instance: MineInstance) -> float:
        """Verified VAN of the plan of predict; higher is better, as in scikit-learn."""
        return checked_evaluation(instance, self.predict(instance)).npv
