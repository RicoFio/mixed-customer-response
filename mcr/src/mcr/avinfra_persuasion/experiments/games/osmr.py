from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...datastructures import MetricName
from ..helpers import sorted_metrics
from .base import BernoulliMaskGameBase, FiniteDifferenceAdamMixin


@dataclass
class OSMRGame(FiniteDifferenceAdamMixin, BernoulliMaskGameBase):
    """
    - OS: One scalar Sender
    - MR: Multiple Receivers with multi-measure preferences
    - State-independent (general) policy
    - Single public signal
    - Finite public prior
    """
    
    _validation_name = "OSMRGame"
    _receiver_count_error = "OSMRGame requires at least one receiver."
    _finite_prior_error = "OSMRGame currently requires a FinitePrior."
    _signal_policy_error = "OSMRGame currently requires a MaskSignalPolicy sender."

    def evaluate_policy(
        self,
        probabilities: Mapping[MetricName, float] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        expected_sender_metric = 0.0
        breakdown_rows: list[dict[str, Any]] = []

        for scenario_name, scenario in self.finite_prior.support.items():
            scenario_probability = self.finite_prior.probabilities[scenario_name]
            for mask in self._all_masks:
                mask_probability = self._mask_probability(mask, probabilities)
                if np.isclose(mask_probability, 0.0):
                    continue

                signal = self.sender.materialize_signal(
                    mask=mask,
                    realized_scenario=scenario,
                )
                evaluation = self._evaluate_signal(signal, scenario)
                weighted_contribution = (
                    scenario_probability
                    * mask_probability
                    * evaluation["sender_metric_value"]
                )
                expected_sender_metric += weighted_contribution
                breakdown_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_probability": scenario_probability,
                        "mask": sorted_metrics(mask),
                        "mask_probability": mask_probability,
                        "sender_metric_value": evaluation["sender_metric_value"],
                        "weighted_contribution": weighted_contribution,
                        "path_counts": dict(evaluation["path_counts"]),
                        "realized_scenario_name": evaluation["realized_scenario"].name,
                    }
                )

        return {
            "expected_sender_utility": expected_sender_metric,
            "expected_sender_metric": expected_sender_metric,
            "breakdown_rows": tuple(breakdown_rows),
        }

    def solve(
        self,
        max_iter: int = 100,
        step_size: float = 0.15,
        finite_diff_epsilon: float = 1e-4,
        convergence_tol: float = 1e-8,
        convergence_patience: int = 15,
    ) -> dict[str, Any]:
        return self._solve_with_finite_difference_adam(
            max_iter=max_iter,
            step_size=step_size,
            finite_diff_epsilon=finite_diff_epsilon,
            convergence_tol=convergence_tol,
            convergence_patience=convergence_patience,
            progress=True,
        )
