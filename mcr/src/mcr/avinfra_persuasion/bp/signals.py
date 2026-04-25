from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..datastructures import MetricName


@dataclass(frozen=True)
class Signal:
    metrics: frozenset[MetricName]
    value: dict[MetricName, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            frozenset(MetricName.coerce(metric) for metric in self.metrics),
        )
        object.__setattr__(
            self,
            "value",
            {
                MetricName.coerce(metric): value
                for metric, value in dict(self.value).items()
            },
        )

    def reveals(self, metric: MetricName | str) -> bool:
        return MetricName.coerce(metric) in self.metrics


@dataclass(frozen=True)
class MaskSignal(Signal):
    """Signal represented by the set of revealed metrics."""


@dataclass
class SignalPolicy:
    seed: int = field(default=1, kw_only=True)

    def sample(self, rng: np.random.Generator | None = None) -> Signal:
        raise NotImplementedError


@dataclass
class MaskSignalPolicy(SignalPolicy):
    """
    Independent Bernoulli mask over a fixed set of considered metrics.

    Each metric has one probability ``p`` of being revealed, with ``1 - p``
    the probability of being hidden.
    """

    considered_metrics: frozenset[MetricName]
    probabilities: dict[MetricName, float] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        metrics = frozenset(
            MetricName.coerce(metric) for metric in self.considered_metrics
        )
        if not metrics:
            raise ValueError(
                "MaskSignalPolicy requires at least one considered metric."
            )

        normalized_probabilities = {
            MetricName.coerce(metric): float(probability)
            for metric, probability in dict(self.probabilities).items()
        }
        extra_metrics = set(normalized_probabilities) - set(metrics)
        if extra_metrics:
            raise ValueError(
                "MaskSignalPolicy probabilities reference metrics outside the "
                f"considered set: {extra_metrics!r}"
            )

        for metric in metrics:
            normalized_probabilities.setdefault(metric, 0.5)
        self._validate_probabilities(normalized_probabilities)

        self.considered_metrics = metrics
        self.probabilities = normalized_probabilities

    def probability(self, metric: MetricName | str) -> float:
        metric_name = self._coerce_considered_metric(metric)
        return self.probabilities[metric_name]

    def update_probability(
        self,
        metric: MetricName | str,
        probability: float,
    ) -> None:
        metric_name = self._coerce_considered_metric(metric)
        probability_value = float(probability)
        self._validate_probability(metric_name, probability_value)
        self.probabilities[metric_name] = probability_value

    def update_probabilities(
        self,
        updates: Mapping[MetricName | str, float],
    ) -> None:
        normalized_updates = {
            self._coerce_considered_metric(metric): float(probability)
            for metric, probability in updates.items()
        }
        for metric, probability in normalized_updates.items():
            self._validate_probability(metric, probability)
        self.probabilities.update(normalized_updates)

    def sample(self, rng: np.random.Generator | None = None) -> MaskSignal:
        draw_rng = self.rng if rng is None else rng
        metric_order = tuple(
            sorted(self.considered_metrics, key=lambda metric: metric.value)
        )
        mask_values = {
            metric: int(draw_rng.binomial(1, self.probabilities[metric]))
            for metric in metric_order
        }
        revealed_metrics = {
            metric for metric, keep in mask_values.items() if keep == 1
        }
        return MaskSignal(metrics=frozenset(revealed_metrics))

    def _coerce_considered_metric(self, metric: MetricName | str) -> MetricName:
        metric_name = MetricName.coerce(metric)
        if metric_name not in self.considered_metrics:
            raise ValueError(
                f"Metric {metric_name.value!r} is not part of this mask policy."
            )
        return metric_name

    @staticmethod
    def _validate_probability(metric: MetricName, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability for {metric.value!r} must lie in [0, 1], "
                f"got {probability!r}."
            )

    @classmethod
    def _validate_probabilities(
        cls,
        probabilities: Mapping[MetricName, float],
    ) -> None:
        for metric, probability in probabilities.items():
            cls._validate_probability(metric, probability)
