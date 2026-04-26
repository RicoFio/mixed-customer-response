from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...bp.game import ConvergenceGame
from ...bp.receivers import Receiver
from ...bp.senders import Sender, Objective
from ...bp.signals import MaskSignalPolicy, Signal
from ...datastructures import (
    FinitePrior,
    MetricName,
    Prior,
    Scenario,
    World,
)

from ..helpers import (
    SignalObservationKey,
    sorted_metrics,
    canonical_signal_key,
    format_signal_key,
)



@dataclass
class GameOne(ConvergenceGame):
    """
    Simple one sender / one receiver game that allows us to 

    Args:
        ConvergenceGame (_type_): _description_

    Returns:
        _type_: _description_
    """
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: Prior
    seed: int

    _metric_order: tuple[MetricName, ...] = field(init=False, repr=False)
    _all_masks: tuple[frozenset[MetricName], ...] = field(init=False, repr=False)
    _logits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.receivers) != 1:
            raise ValueError("GameOne currently supports exactly one receiver.")
        if not isinstance(self.public_prior, FinitePrior):
            raise NotImplementedError("GameOne currently requires a FinitePrior.")
        if not isinstance(self.sender.signal_policy, MaskSignalPolicy):
            raise NotImplementedError(
                "GameOne currently requires a MaskSignalPolicy sender."
            )
        if self.sender.prior != self.public_prior:
            raise ValueError("Sender prior must match the public prior.")
        if any(receiver.prior != self.public_prior for receiver in self.receivers):
            raise ValueError("Receiver priors must match the public prior.")
        if self.sender.world != self.world:
            raise ValueError("Sender world must match the game world.")
        if any(receiver.world != self.world for receiver in self.receivers):
            raise ValueError("Receiver worlds must match the game world.")

        self._metric_order = sorted_metrics(
            self.sender.signal_policy.considered_metrics
        )
        self._all_masks = tuple(
            frozenset(
                metric
                for idx, metric in enumerate(self._metric_order)
                if bitmask & (1 << idx)
            )
            for bitmask in range(1 << len(self._metric_order))
        )

        probabilities = np.array(
            [
                self.sender.signal_policy.probability(metric)
                for metric in self._metric_order
            ],
            dtype=float,
        )
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        self._logits = np.log(clipped / (1.0 - clipped))

    @property
    def receiver(self) -> Receiver:
        return self.receivers[0]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def signaling_scheme(
        self,
        logits: np.ndarray | None = None,
    ) -> dict[MetricName, float]:
        logits = self._logits if logits is None else np.asarray(logits, dtype=float)
        probabilities = self._sigmoid(logits)
        return {
            metric: float(probability)
            for metric, probability in zip(self._metric_order, probabilities)
        }

    def _mask_probability(
        self,
        mask: frozenset[MetricName],
        probabilities: Mapping[MetricName, float],
    ) -> float:
        mass = 1.0
        for metric in self._metric_order:
            probability = probabilities[metric]
            mass *= probability if metric in mask else (1.0 - probability)
        return mass

    def _sender_metric(self) -> MetricName:
        if len(self.sender.preference.elements) != 1:
            raise ValueError("Sender preference must be degenerate.")
        return next(iter(self.sender.preference.elements))

    def posterior_from_signal(self, signal: Signal) -> FinitePrior:
        if not signal.value:
            return self.public_prior

        posterior_support: dict[str, Scenario] = {}
        posterior_probabilities: dict[str, float] = {}
        for scenario_name, scenario in self.public_prior.support.items():
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
                    self.public_prior.probabilities[scenario_name]
                )

        if not posterior_support:
            raise ValueError("Signal is inconsistent with the finite prior support.")

        return FinitePrior(
            name=f"{self.public_prior.name}_posterior",
            support=posterior_support,
            probabilities=posterior_probabilities,
        )

    def _receiver_after_signal(self, signal: Signal) -> Receiver:
        self.receiver.reset_for_evaluation()
        self.receiver.update_internal_belief(signal)
        return self.receiver

    def _evaluate_signal(
        self,
        signal: Signal,
        realized_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receiver = self._receiver_after_signal(signal)
        posterior = updated_receiver.belief
        if not isinstance(posterior, FinitePrior):
            raise NotImplementedError("GameOne currently requires a FinitePrior.")
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
        for scenario_name, scenario in self.public_prior.support.items():
            scenario_probability = self.public_prior.probabilities[scenario_name]
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
            joint_by_signal[row["signal_key"]][row["scenario_name"]] += joint_probability

        reconstructed_prior = {
            scenario_name: 0.0 for scenario_name in self.public_prior.support
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
                for scenario_name in self.public_prior.support
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
                - self.public_prior.probabilities[scenario_name]
            )
            for scenario_name in self.public_prior.support
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
                    - self.public_prior.probabilities[scenario_name]
                )
                > 1e-12
                for scenario_name in self.public_prior.support
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
            sender_metric_value = -evaluation["realized_metrics"][sender_metric]
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
            "expected_sender_utility": -expected_sender_metric,
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

    def _objective_from_flat_logits(self, flat_logits: np.ndarray) -> float:
        probabilities = self.signaling_scheme(logits=flat_logits)
        return float(self.evaluate_policy(probabilities)["expected_sender_utility"])

    def _finite_difference_gradient(
        self,
        flat_logits: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        gradient = np.zeros_like(flat_logits)
        for idx in range(flat_logits.size):
            direction = np.zeros_like(flat_logits)
            direction[idx] = epsilon
            plus = self._objective_from_flat_logits(flat_logits + direction)
            minus = self._objective_from_flat_logits(flat_logits - direction)
            gradient[idx] = (plus - minus) / (2.0 * epsilon)
        return gradient

    def solve(
        self,
        max_iter: int = 100,
        step_size: float = 0.15,
        finite_diff_epsilon: float = 1e-2,
        convergence_tol: float = 1e-6,
        convergence_patience: int = 15,
    ) -> dict[str, Any]:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if finite_diff_epsilon <= 0:
            raise ValueError("finite_diff_epsilon must be positive.")
        if convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive.")

        flat_logits = self._logits.copy()
        m = np.zeros_like(flat_logits)
        v = np.zeros_like(flat_logits)
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-4

        utility_history: list[float] = []
        grad_norm_history: list[float] = []
        policy_history: list[dict[MetricName, float]] = []
        stagnant_steps = 0
        converged = False

        for step in range(1, max_iter + 1):
            policy_history.append(self.signaling_scheme(logits=flat_logits))
            utility = self._objective_from_flat_logits(flat_logits)
            gradient = self._finite_difference_gradient(
                flat_logits=flat_logits,
                epsilon=finite_diff_epsilon,
            )
            gradient = -gradient if self.sender.objective == Objective.MINIMIZE else gradient
            grad_norm = float(np.linalg.norm(gradient))

            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            flat_logits = flat_logits + step_size * m_hat / (np.sqrt(v_hat) + adam_eps)

            utility_history.append(float(utility))
            grad_norm_history.append(grad_norm)

            if len(utility_history) >= 2:
                utility_delta = abs(utility_history[-1] - utility_history[-2])
                if utility_delta < convergence_tol:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0

            if stagnant_steps >= convergence_patience:
                converged = True
                break

        self._logits = flat_logits
        final_probabilities = self.signaling_scheme()
        self.sender.signal_policy.update_probabilities(final_probabilities)
        evaluation = self.evaluate_policy(final_probabilities)
        return {
            "iterations": len(utility_history),
            "converged": converged,
            "utility_history": utility_history,
            "grad_norm_history": grad_norm_history,
            "policy_history": policy_history + [final_probabilities],
            "final_probabilities": final_probabilities,
            **evaluation,
        }
