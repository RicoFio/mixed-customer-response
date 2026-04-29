from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from enum import Enum

from ..datastructures import MetricName, Prior, Scenario, World
from .game import Preference
from .signals import (
    MaskSignal,
    MaskSignalPolicy,
    Signal,
    SignalPolicy,
    StateDependentMaskSignalPolicy,
    TypedStateDependentMaskSignalPolicy,
)


ARC_SIGNAL_METRICS = frozenset(
    {
        MetricName.TRAVEL_TIME,
        MetricName.DISCOMFORT,
        MetricName.HAZARD,
        MetricName.COST,
        MetricName.EMISSIONS,
    }
)
NODE_SIGNAL_METRICS = frozenset({MetricName.POLICING})


class Objective(Enum):
    MAXIMIZE = 0
    MINIMIZE = 1


@dataclass
class Sender:
    prior: Prior
    world: World
    preference: Preference
    signal_policy: SignalPolicy
    objective: Objective = Objective.MINIMIZE

    def emit_signal(
        self,
        realized_scenario: Scenario,
        rng: np.random.Generator | None = None,
        *,
        receiver_type: str | None = None,
    ) -> Signal:
        sampled_signal = self.signal_policy.sample(
            realized_scenario=realized_scenario,
            rng=rng,
            receiver_type=receiver_type,
        )
        if isinstance(sampled_signal, MaskSignal):
            return self.materialize_signal(
                mask=sampled_signal.metrics,
                realized_scenario=realized_scenario,
            )
        raise NotImplementedError(
            f"Unsupported signal type: {type(sampled_signal).__name__!r}."
        )

    def materialize_signal(
        self,
        mask: frozenset[MetricName],
        realized_scenario: Scenario,
    ) -> MaskSignal:
        signal = MaskSignal(metrics=mask)
        observed_values: dict[MetricName, Mapping[object, float]] = {}
        instrumented_nodes = {
            node
            for arc in self.world.I
            for node in arc
        }

        for metric in signal.metrics:
            if metric in ARC_SIGNAL_METRICS:
                metric_values = getattr(realized_scenario, metric.value)
                observed_values[metric] = {
                    arc: metric_values[arc]
                    for arc in self.world.I
                }
                continue

            if metric in NODE_SIGNAL_METRICS:
                metric_values = getattr(realized_scenario, metric.value)
                observed_values[metric] = {
                    node: metric_values[node]
                    for node in instrumented_nodes
                }
                continue

            raise NotImplementedError(
                "MaskSignal currently supports only realized arc and node metrics, "
                f"got {metric.value!r}."
            )

        return MaskSignal(metrics=signal.metrics, value=observed_values)

    def _materialize_mask_signal(
        self,
        *,
        signal: MaskSignal,
        realized_scenario: Scenario,
    ) -> MaskSignal:
        return self.materialize_signal(
            mask=signal.metrics,
            realized_scenario=realized_scenario,
        )

    def signal_likelihood(
        self,
        signal: Signal,
        scenario: Scenario,
        *,
        receiver_type: str | None = None,
    ) -> float:
        truthful_signal = self.materialize_signal(
            mask=signal.metrics,
            realized_scenario=scenario,
        )
        if (
            truthful_signal.metrics != signal.metrics
            or truthful_signal.value != signal.value
        ):
            return 0.0

        if isinstance(self.signal_policy, MaskSignalPolicy):
            return self.signal_policy.mask_probability(signal.metrics)
        if isinstance(self.signal_policy, StateDependentMaskSignalPolicy):
            return self.signal_policy.mask_probability(scenario.name, signal.metrics)
        if isinstance(self.signal_policy, TypedStateDependentMaskSignalPolicy):
            if receiver_type is None:
                raise ValueError(
                    "TypedStateDependentMaskSignalPolicy requires a receiver type "
                    "to compute signal likelihood."
                )
            return self.signal_policy.mask_probability(
                scenario.name,
                receiver_type,
                signal.metrics,
            )
        raise NotImplementedError(
            f"Unsupported signal policy: {type(self.signal_policy).__name__!r}."
        )

    def expected_metrics(self, realized_world):
        # Given the realized world, obtain the total metrics across the entire network
        raise NotImplementedError()

@dataclass
class ScalarSender(Sender):

    def __post_init__(self) -> None:
        if not self.preference.is_degenerate():
            raise ValueError(
                "A ScalarSender can only consider one metric. Preference needs "
                "to be `degenerate`."
            )


@dataclass
class OnlineSender(Sender):
    pass
