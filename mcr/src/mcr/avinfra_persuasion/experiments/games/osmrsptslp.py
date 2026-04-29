from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import factorial
from typing import Any

import numpy as np

from ...bp.receivers import Receiver
from ...bp.signals import Signal
from ...datastructures import MetricName, Scenario
from ..helpers import sorted_metrics
from .base import FiniteDifferenceAdamMixin, TypedStateDependentMaskGameBase, EnumerationMixin

MaskCountProfile = tuple[int, ...]


@dataclass
class OSMRSPTSLPGame(FiniteDifferenceAdamMixin, TypedStateDependentMaskGameBase, EnumerationMixin):
    """
    - OS: One scalar Sender
    - MR: Multiple Receivers with multi-measure preferences
    - SP: State-dependent policy
    - TS: Type-private (randomized) signal
    - LP: Lottery Policy
    - Finite public prior

    The policy is anonymous within a receiver type: receivers with the same
    ``rtype`` use the same state/type mask distribution. Lottery evaluation
    still draws a private mask per receiver, so same-type receivers can receive
    different realized private signals without giving the policy receiver IDs.
    """
    _validation_name = "OSMRSPTSLPGame"
    _receiver_count_error = "OSMRSPTSLPGame requires at least one receiver."
    _finite_prior_error = "OSMRSPTSLPGame currently requires a FinitePrior."
    _signal_policy_error = (
        "OSMRSPTSLPGame currently requires a TypedStateDependentMaskSignalPolicy sender."
    )

    def _path_choices_after_receiver_signals(
        self,
        signals_by_receiver_id: Mapping[str, Signal],
    ) -> tuple[list[Receiver], dict[Any, Any]]:
        updated_receivers: list[Receiver] = []
        path_choices = {}

        for receiver in self.receivers:
            updated_receiver = self._receiver_after_signal(
                receiver,
                signals_by_receiver_id[receiver.id],
            )
            updated_receivers.append(updated_receiver)
            path_choices[updated_receiver.individual] = (
                updated_receiver.get_path_choice()
            )

        return updated_receivers, path_choices

    def _evaluate_lottery_signals(
        self,
        signals_by_receiver_id: Mapping[str, Signal],
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receivers, path_choices = self._path_choices_after_receiver_signals(
            signals_by_receiver_id
        )
        realized_scenario = self.world.get_realized_metrics(
            path_choices=path_choices,
            name=f"realized_{believed_scenario.name}",
            base_scenario=believed_scenario,
        )
        receiver_metrics = self._receiver_metrics_after_realization(
            updated_receivers=updated_receivers,
            realized_scenario=realized_scenario,
        )
        sender_metric = self._sender_metric()
        sender_metric_value = sum(
            metrics[sender_metric] for metrics in receiver_metrics.values()
        )

        return {
            "realized_scenario": realized_scenario,
            "path_choices": path_choices,
            "receiver_metrics": receiver_metrics,
            "sender_metric_value": sender_metric_value,
            "path_counts": Counter(choice.path for choice in path_choices.values()),
        }

    def _receivers_by_type(self) -> dict[str, tuple[Receiver, ...]]:
        return {
            type_name: tuple(
                receiver
                for receiver in self.receivers
                if receiver.rtype == type_name
            )
            for type_name in self._type_order
        }

    def _mask_count_profiles(
        self,
        n_receivers: int,
        n_masks: int | None = None,
    ) -> tuple[MaskCountProfile, ...]:
        n_masks = len(self._all_masks) if n_masks is None else n_masks
        if n_masks == 1:
            return ((n_receivers,),)

        profiles: list[MaskCountProfile] = []
        for count in range(n_receivers + 1):
            for tail in self._mask_count_profiles(n_receivers - count, n_masks - 1):
                profiles.append((count, *tail))
        return tuple(profiles)

    def _type_count_profile_probability(
        self,
        *,
        scenario_name: str,
        type_name: str,
        count_profile: MaskCountProfile,
        probabilities: Mapping[str, Mapping[str, Mapping[frozenset[MetricName], float]]],
    ) -> float:
        n_receivers = sum(count_profile)
        mass = float(factorial(n_receivers))
        for count, mask in zip(count_profile, self._all_masks):
            if count == 0:
                continue
            mask_probability = self._mask_probability(
                scenario_name,
                type_name,
                mask,
                probabilities,
            )
            mass *= (mask_probability ** count) / factorial(count)
        return mass

    def _assign_count_profiles_to_receivers(
        self,
        count_profiles_by_type: Mapping[str, MaskCountProfile],
        receivers_by_type: Mapping[str, tuple[Receiver, ...]],
    ) -> dict[str, frozenset[MetricName]]:
        masks_by_receiver_id: dict[str, frozenset[MetricName]] = {}
        for type_name, count_profile in count_profiles_by_type.items():
            receivers = receivers_by_type[type_name]
            receiver_idx = 0
            for mask, count in zip(self._all_masks, count_profile):
                for receiver in receivers[receiver_idx : receiver_idx + count]:
                    masks_by_receiver_id[receiver.id] = mask
                receiver_idx += count
        return masks_by_receiver_id

    def _format_mask_count_profile(
        self,
        count_profile: MaskCountProfile,
    ) -> dict[tuple[MetricName, ...], int]:
        return {
            sorted_metrics(mask): count
            for mask, count in zip(self._all_masks, count_profile)
            if count
        }

    def evaluate_policy(
        self,
        probabilities: (
            Mapping[str, Mapping[str, Mapping[frozenset[MetricName], float]]]
            | None
        ) = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        expected_sender_metric = 0.0
        breakdown_rows: list[dict[str, Any]] = []
        receivers_by_type = self._receivers_by_type()
        count_profiles_by_type_options = {
            type_name: self._mask_count_profiles(len(receivers))
            for type_name, receivers in receivers_by_type.items()
        }

        with self._temporary_state_distributions(probabilities):
            for scenario_name, scenario in self.finite_prior.support.items():
                scenario_probability = self.finite_prior.probabilities[scenario_name]
                for count_profile_tuple in product(
                    *(
                        count_profiles_by_type_options[type_name]
                        for type_name in self._type_order
                    )
                ):
                    count_profiles_by_type = {
                        type_name: count_profile
                        for type_name, count_profile in zip(
                            self._type_order,
                            count_profile_tuple,
                        )
                    }
                    mask_probability = 1.0
                    for type_name, count_profile in count_profiles_by_type.items():
                        mask_probability *= self._type_count_profile_probability(
                            scenario_name=scenario_name,
                            type_name=type_name,
                            count_profile=count_profile,
                            probabilities=probabilities,
                        )
                    if np.isclose(mask_probability, 0.0):
                        continue

                    masks_by_receiver_id = self._assign_count_profiles_to_receivers(
                        count_profiles_by_type,
                        receivers_by_type,
                    )
                    signals_by_receiver_id = {
                        receiver.id: self.sender.materialize_signal(
                            mask=mask,
                            realized_scenario=scenario,
                        )
                        for receiver in self.receivers
                        for mask in (masks_by_receiver_id[receiver.id],)
                    }
                    evaluation = self._evaluate_lottery_signals(
                        signals_by_receiver_id,
                        scenario,
                    )
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
                            "mask_counts_by_type": {
                                type_name: self._format_mask_count_profile(
                                    count_profile
                                )
                                for type_name, count_profile
                                in count_profiles_by_type.items()
                            },
                            "mask_probability": mask_probability,
                            "sender_metric_value": evaluation["sender_metric_value"],
                            "weighted_contribution": weighted_contribution,
                            "path_counts": dict(evaluation["path_counts"]),
                            "realized_scenario_name": evaluation[
                                "realized_scenario"
                            ].name,
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
        step_size: float = 0.1,
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

    def solve_exact(self) -> dict[str, Any]:
        return self._solve_by_enumeration()
