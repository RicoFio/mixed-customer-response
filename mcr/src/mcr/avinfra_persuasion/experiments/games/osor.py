from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...bp.receivers import Receiver
from ...bp.signals import Signal
from ...datastructures import FinitePrior, MetricName, Scenario
from ..helpers import (
    SignalObservationKey,
    canonical_signal_key,
    format_signal_key,
    sorted_metrics,
)
from .base import BernoulliMaskGameBase, FiniteDifferenceAdamMixin


@dataclass
class OSORGame(FiniteDifferenceAdamMixin, BernoulliMaskGameBase):
    """
    In this game:
        - Scalar sender
        - Single receiver with multi-measure preferences
        - State-independent (general) policy
        - Single public signal
        - Finite public prior
    """

    _validation_name = "OSORGame"
    _exact_receivers = 1
    _receiver_count_error = "OSORGame currently supports exactly one receiver."
    _finite_prior_error = "OSORGame currently requires a FinitePrior."
    _signal_policy_error = "OSORGame currently requires a MaskSignalPolicy sender."

    @property
    def receiver(self) -> Receiver:
        return self.receivers[0]

    def posterior_from_signal(self, signal: Signal) -> FinitePrior:
        if not signal.value:
            return self.finite_prior

        posterior_support: dict[str, Scenario] = {}
        posterior_probabilities: dict[str, float] = {}
        for scenario_name, scenario in self.finite_prior.support.items():
            for metric, observed_value in signal.value.items():
                realized_value = getattr(scenario, metric.value)
                if isinstance(observed_value, Mapping):
                    if not isinstance(realized_value, Mapping) or any(
                        key not in realized_value or realized_value[key] != value
                        for key, value in observed_value.items()
                    ):
                        break
                elif realized_value != observed_value:
                    break
            else:
                posterior_support[scenario_name] = scenario
                posterior_probabilities[scenario_name] = (
                    self.finite_prior.probabilities[scenario_name]
                )

        if not posterior_support:
            raise ValueError("Signal is inconsistent with the finite prior support.")

        return FinitePrior(
            name=f"{self.finite_prior.name}_posterior",
            support=posterior_support,
            probabilities=posterior_probabilities,
        )

    def _receiver_after_signal(self, receiver: Receiver, signal: Signal) -> Receiver:
        self.receiver.reset_for_evaluation()
        self.receiver.update_internal_belief(signal)
        return self.receiver

    def _evaluate_signal(
        self,
        signal: Signal,
        realized_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receiver = self._receiver_after_signal(self.receiver, signal)
        posterior = updated_receiver.belief
        if not isinstance(posterior, FinitePrior):
            raise NotImplementedError("OSORGame currently requires a FinitePrior.")
        chosen_route = updated_receiver.get_path_choice()
        realized_metrics = updated_receiver.compute_realized_metrics(realized_scenario)
        return {
            "posterior": posterior,
            "chosen_route": chosen_route,
            "realized_metrics": realized_metrics,
        }

    def _scenario_mask_rows(
        self,
        probabilities: Mapping[MetricName, float],
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for scenario_name, scenario in self.finite_prior.support.items():
            scenario_probability = self.finite_prior.probabilities[scenario_name]
            for mask in self._all_masks:
                signal = self.sender.materialize_signal(
                    mask=mask,
                    realized_scenario=scenario,
                )
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario": scenario,
                        "scenario_probability": scenario_probability,
                        "mask": sorted_metrics(mask),
                        "mask_probability": self._mask_probability(mask, probabilities),
                        "signal": signal,
                        "signal_key": canonical_signal_key(signal),
                    }
                )
        return tuple(rows)

    def bayes_plausibility_report(
        self,
        probabilities: Mapping[MetricName, float] | None = None,
        rows: tuple[dict[str, Any], ...] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()
        if rows is None:
            rows = self._scenario_mask_rows(probabilities)

        joint_by_signal: dict[SignalObservationKey, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        signal_probabilities: dict[SignalObservationKey, float] = defaultdict(float)
        for row in rows:
            joint_probability = row["scenario_probability"] * row["mask_probability"]
            signal_probabilities[row["signal_key"]] += joint_probability
            joint_by_signal[row["signal_key"]][row["scenario_name"]] += (
                joint_probability
            )

        reconstructed_prior = {
            scenario_name: 0.0 for scenario_name in self.finite_prior.support
        }
        report_rows: list[dict[str, Any]] = []
        for signal_key in sorted(signal_probabilities, key=format_signal_key):
            signal_probability = signal_probabilities[signal_key]
            if signal_probability <= 0.0:
                continue
            posterior_probabilities = {
                scenario_name: (
                    joint_by_signal[signal_key].get(scenario_name, 0.0)
                    / signal_probability
                )
                for scenario_name in self.finite_prior.support
            }
            for scenario_name, posterior_probability in posterior_probabilities.items():
                reconstructed_prior[scenario_name] += (
                    signal_probability * posterior_probability
                )
            report_rows.append(
                {
                    "signal_summary": format_signal_key(signal_key),
                    "signal_probability": signal_probability,
                    "posterior_probabilities": posterior_probabilities,
                }
            )

        max_error = max(
            abs(
                reconstructed_prior[scenario_name]
                - self.finite_prior.probabilities[scenario_name]
            )
            for scenario_name in self.finite_prior.support
        )
        return {
            "rows": tuple(report_rows),
            "reconstructed_prior": reconstructed_prior,
            "max_error": max_error,
        }

    def _mask_verification_rows(
        self,
        rows: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        nominal_scenario = Scenario.from_world("nominal", self.world)
        verification_rows: list[dict[str, Any]] = []
        for row in rows:
            posterior = self.posterior_from_signal(row["signal"])
            observed_values: list[dict[str, Any]] = []
            for metric in sorted_metrics(frozenset(row["signal"].value)):
                nominal_values = getattr(nominal_scenario, metric.value)
                for key, realized_value in sorted(
                    row["signal"].value[metric].items(),
                    key=lambda item: str(item[0]),
                ):
                    observed_values.append(
                        {
                            "metric": metric,
                            "key": key,
                            "nominal_value": nominal_values[key],
                            "realized_value": realized_value,
                        }
                    )

            posterior_changed = any(
                abs(
                    posterior.probabilities.get(scenario_name, 0.0)
                    - self.finite_prior.probabilities[scenario_name]
                )
                > 1e-12
                for scenario_name in self.finite_prior.support
            )
            verification_rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "mask": row["mask"],
                    "hidden_metrics": tuple(
                        metric
                        for metric in self._metric_order
                        if metric not in row["mask"]
                    ),
                    "signal_summary": format_signal_key(row["signal_key"]),
                    "observed_values": tuple(observed_values),
                    "posterior_probabilities": dict(posterior.probabilities),
                    "posterior_changed": posterior_changed,
                }
            )
        return tuple(verification_rows)

    def evaluate_policy(
        self,
        probabilities: Mapping[MetricName, float] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        rows = self._scenario_mask_rows(probabilities)
        sender_metric = self._sender_metric()
        expected_sender_metric = 0.0
        breakdown_rows: list[dict[str, Any]] = []

        for row in rows:
            evaluation = self._evaluate_signal(
                signal=row["signal"],
                realized_scenario=row["scenario"],
            )
            sender_metric_value = evaluation["realized_metrics"][sender_metric]
            weighted_contribution = (
                row["scenario_probability"]
                * row["mask_probability"]
                * sender_metric_value
            )
            expected_sender_metric += weighted_contribution
            breakdown_rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "scenario_probability": row["scenario_probability"],
                    "mask": row["mask"],
                    "mask_probability": row["mask_probability"],
                    "signal_summary": format_signal_key(row["signal_key"]),
                    "posterior_probabilities": dict(
                        evaluation["posterior"].probabilities
                    ),
                    "chosen_route_label": evaluation["chosen_route"].label,
                    "chosen_path": evaluation["chosen_route"].path,
                    "realized_metrics": dict(evaluation["realized_metrics"]),
                    "sender_metric_value": sender_metric_value,
                    "weighted_contribution": weighted_contribution,
                }
            )

        return {
            "expected_sender_utility": expected_sender_metric,
            "expected_sender_metric": expected_sender_metric,
            "breakdown_rows": tuple(breakdown_rows),
        }

    def diagnostics(
        self,
        probabilities: Mapping[MetricName, float] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        rows = self._scenario_mask_rows(probabilities)
        return {
            "bayes_report": self.bayes_plausibility_report(
                probabilities=probabilities,
                rows=rows,
            ),
            "mask_verification_rows": self._mask_verification_rows(rows),
        }

    def solve(
        self,
        max_iter: int = 100,
        step_size: float = 0.15,
        finite_diff_epsilon: float = 1e-2,
        convergence_tol: float = 1e-6,
        convergence_patience: int = 15,
        progress: bool = False,
    ) -> dict[str, Any]:
        return self._solve_with_finite_difference_adam(
            max_iter=max_iter,
            step_size=step_size,
            finite_diff_epsilon=finite_diff_epsilon,
            convergence_tol=convergence_tol,
            convergence_patience=convergence_patience,
            progress=progress
        )
