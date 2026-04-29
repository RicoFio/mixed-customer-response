from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from ...bp.receivers import ExperiencedRouteChoiceReceiver, Receiver
from ...bp.signals import Signal
from ...datastructures import MetricName, Scenario
from ..helpers import sorted_metrics
from .base import StateDependentMaskGameBase, EnumerationMixin


@dataclass
class OSMRSPMRGame(StateDependentMaskGameBase, EnumerationMixin):
    """
    - OS: One scalar Sender
    - MR: Multiple Receivers with multi-measure preferences
    - SP: State-dependent policy
    - MR: Receivers with memory over previous realizations
    - Single public signal
    - Finite public prior
    """
    horizon: int = 20

    _rollout_scenarios: tuple[Scenario, ...] = field(init=False, repr=False)

    _validation_name = "OSMRSPMRGame"
    _receiver_count_error = "OSMRSPMRGame requires at least one receiver."
    _finite_prior_error = "OSMRSPMRGame currently requires a FinitePrior."
    _signal_policy_error = (
        "OSMRSPMRGame currently requires a StateDependentMaskSignalPolicy sender."
    )

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        super().__post_init__()
        if any(
            not isinstance(receiver, ExperiencedRouteChoiceReceiver)
            for receiver in self.receivers
        ):
            raise ValueError(
                "OSMRSPMRGame currently requires ExperiencedRouteChoiceReceiver instances."
            )
        self._rollout_scenarios = tuple(
            self.finite_prior.sample(self.horizon, seed=self.seed)
        )

    def _receiver_after_signal(
        self,
        receiver: ExperiencedRouteChoiceReceiver,
        signal: Signal,
    ) -> ExperiencedRouteChoiceReceiver:
        receiver.reset_public_belief()
        receiver.update_internal_belief(signal)
        return receiver

    def _receiver_metrics_after_realization(
        self,
        updated_receivers: list[Receiver],
        realized_scenario: Scenario,
    ) -> dict[str, Mapping[MetricName, float]]:
        receiver_metrics: dict[str, Mapping[MetricName, float]] = {}
        for receiver in updated_receivers:
            if not isinstance(receiver, ExperiencedRouteChoiceReceiver):
                raise ValueError(
                    "OSMRSPMRGame currently requires ExperiencedRouteChoiceReceiver instances."
                )
            realized_metrics = receiver.compute_realized_metrics(realized_scenario)
            receiver.update_private_route_belief(realized_metrics)
            receiver_metrics[receiver.id] = realized_metrics
        return receiver_metrics

    def _evaluate_round(
        self,
        round_index: int,
        signal: Signal,
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        evaluation = self._evaluate_multi_receiver_signal(
            signal=signal,
            believed_scenario=believed_scenario,
        )
        path_choices = evaluation["path_choices"]
        updated_paths = tuple(
            sorted(
                {choice.path for choice in path_choices.values()},
                key=str,
            )
        )

        return {
            "round": round_index,
            "scenario_name": believed_scenario.name,
            "mask": sorted_metrics(signal.metrics),
            **evaluation,
            "updated_paths": updated_paths,
        }

    def evaluate_policy(
        self,
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        total_sender_metric = 0.0
        breakdown_rows: list[dict[str, Any]] = []
        rollout_rng = np.random.default_rng(self.seed + 1)

        with self._temporary_state_distributions(probabilities):
            for receiver in self.receivers:
                receiver.reset_for_rollout()
            for round_index, scenario in enumerate(self._rollout_scenarios, start=1):
                signal = self.sender.emit_signal(
                    realized_scenario=scenario,
                    rng=rollout_rng,
                )
                evaluation = self._evaluate_round(
                    round_index=round_index,
                    signal=signal,
                    believed_scenario=scenario,
                )
                total_sender_metric += evaluation["sender_metric_value"]
                breakdown_rows.append(
                    {
                        "round": evaluation["round"],
                        "scenario_name": evaluation["scenario_name"],
                        "mask": evaluation["mask"],
                        "path_counts": dict(evaluation["path_counts"]),
                        "sender_metric_value": evaluation["sender_metric_value"],
                        "updated_paths": evaluation["updated_paths"],
                    }
                )

        average_sender_metric = total_sender_metric / self.horizon
        return {
            "expected_sender_utility": average_sender_metric,
            "average_sender_metric": average_sender_metric,
            "breakdown_rows": tuple(breakdown_rows),
            "final_private_route_beliefs": {
                receiver.id: receiver.private_route_beliefs
                for receiver in self.receivers
            },
        }

    def solve_exact(self):
        self._solve_by_enumeration()
