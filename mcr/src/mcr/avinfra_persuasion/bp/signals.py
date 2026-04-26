from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..datastructures import MetricName, Scenario


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

    def sample(
        self,
        realized_scenario: Scenario | None = None,
        rng: np.random.Generator | None = None,
    ) -> Signal:
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

    def sample(
        self,
        realized_scenario: Scenario | None = None,
        rng: np.random.Generator | None = None,
    ) -> MaskSignal:
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

    def mask_probability(
        self,
        mask: frozenset[MetricName] | set[MetricName],
    ) -> float:
        normalized_mask = self._coerce_mask(mask)
        probability = 1.0
        for metric in self.considered_metrics:
            keep_probability = self.probabilities[metric]
            probability *= keep_probability if metric in normalized_mask else (1.0 - keep_probability)
        return probability

    def _coerce_mask(
        self,
        mask: frozenset[MetricName] | set[MetricName],
    ) -> frozenset[MetricName]:
        normalized_mask = frozenset(MetricName.coerce(metric) for metric in mask)
        extra_metrics = set(normalized_mask) - set(self.considered_metrics)
        if extra_metrics:
            raise ValueError(
                "Mask contains metrics outside the considered set: "
                f"{extra_metrics!r}"
            )
        return normalized_mask

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


@dataclass
class StateDependentMaskSignalPolicy(SignalPolicy):
    state_names: frozenset[str]
    considered_metrics: frozenset[MetricName]
    state_probabilities: dict[str, dict[frozenset[MetricName], float]] = field(
        default_factory=dict
    )
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        normalized_state_names = frozenset(str(name) for name in self.state_names)
        if not normalized_state_names:
            raise ValueError(
                "StateDependentMaskSignalPolicy requires at least one state."
            )

        metrics = frozenset(
            MetricName.coerce(metric) for metric in self.considered_metrics
        )
        if not metrics:
            raise ValueError(
                "StateDependentMaskSignalPolicy requires at least one considered metric."
            )

        all_masks = self._all_masks(metrics)
        explicit_distributions = dict(self.state_probabilities)
        extra_states = set(explicit_distributions) - set(normalized_state_names)
        if extra_states:
            raise ValueError(
                "StateDependentMaskSignalPolicy probabilities reference unknown "
                f"states: {extra_states!r}"
            )

        normalized_distributions: dict[str, dict[frozenset[MetricName], float]] = {}
        uniform_probability = 1.0 / len(all_masks)
        for state_name in sorted(normalized_state_names):
            if state_name in explicit_distributions:
                normalized_distributions[state_name] = self._normalize_distribution(
                    state_name=state_name,
                    distribution=explicit_distributions[state_name],
                    all_masks=all_masks,
                )
            else:
                normalized_distributions[state_name] = {
                    mask: uniform_probability for mask in all_masks
                }

        self.state_names = normalized_state_names
        self.considered_metrics = metrics
        self.state_probabilities = normalized_distributions

    def distribution_for_state(
        self,
        state_name: str,
    ) -> Mapping[frozenset[MetricName], float]:
        normalized_state_name = self._coerce_state_name(state_name)
        return dict(self.state_probabilities[normalized_state_name])

    def mask_probability(
        self,
        state_name: str,
        mask: frozenset[MetricName] | set[MetricName],
    ) -> float:
        normalized_state_name = self._coerce_state_name(state_name)
        normalized_mask = self._coerce_mask(mask)
        return self.state_probabilities[normalized_state_name][normalized_mask]

    def update_state_distribution(
        self,
        state_name: str,
        distribution: Mapping[frozenset[MetricName] | set[MetricName], float],
    ) -> None:
        normalized_state_name = self._coerce_state_name(state_name)
        all_masks = self._all_masks(self.considered_metrics)
        self.state_probabilities[normalized_state_name] = self._normalize_distribution(
            state_name=normalized_state_name,
            distribution=distribution,
            all_masks=all_masks,
        )

    def update_state_distributions(
        self,
        distributions: Mapping[
            str,
            Mapping[frozenset[MetricName] | set[MetricName], float],
        ],
    ) -> None:
        for state_name, distribution in distributions.items():
            self.update_state_distribution(state_name, distribution)

    def sample(
        self,
        realized_scenario: Scenario | None = None,
        rng: np.random.Generator | None = None,
    ) -> MaskSignal:
        if realized_scenario is None:
            raise ValueError(
                "StateDependentMaskSignalPolicy requires a realized scenario to sample."
            )
        state_name = self._coerce_state_name(realized_scenario.name)
        draw_rng = self.rng if rng is None else rng
        masks = tuple(
            sorted(
                self.state_probabilities[state_name],
                key=lambda mask: tuple(metric.value for metric in sorted(mask, key=lambda metric: metric.value)),
            )
        )
        probabilities = np.array(
            [self.state_probabilities[state_name][mask] for mask in masks],
            dtype=float,
        )
        mask_idx = int(draw_rng.choice(len(masks), p=probabilities))
        return MaskSignal(metrics=masks[mask_idx])

    def _coerce_state_name(self, state_name: str) -> str:
        normalized_state_name = str(state_name)
        if normalized_state_name not in self.state_names:
            raise ValueError(f"Unknown state name: {normalized_state_name!r}.")
        return normalized_state_name

    def _coerce_mask(
        self,
        mask: frozenset[MetricName] | set[MetricName],
    ) -> frozenset[MetricName]:
        normalized_mask = frozenset(MetricName.coerce(metric) for metric in mask)
        extra_metrics = set(normalized_mask) - set(self.considered_metrics)
        if extra_metrics:
            raise ValueError(
                "Mask contains metrics outside the considered set: "
                f"{extra_metrics!r}"
            )
        return normalized_mask

    @staticmethod
    def _all_masks(
        metrics: frozenset[MetricName],
    ) -> tuple[frozenset[MetricName], ...]:
        ordered_metrics = tuple(sorted(metrics, key=lambda metric: metric.value))
        return tuple(
            frozenset(
                metric
                for idx, metric in enumerate(ordered_metrics)
                if bitmask & (1 << idx)
            )
            for bitmask in range(1 << len(ordered_metrics))
        )

    def _normalize_distribution(
        self,
        *,
        state_name: str,
        distribution: Mapping[frozenset[MetricName] | set[MetricName], float],
        all_masks: tuple[frozenset[MetricName], ...],
    ) -> dict[frozenset[MetricName], float]:
        normalized_distribution = {mask: 0.0 for mask in all_masks}
        specified_masks: set[frozenset[MetricName]] = set()
        for mask, probability in distribution.items():
            normalized_mask = self._coerce_mask(mask)
            probability_value = float(probability)
            if probability_value < 0.0:
                raise ValueError(
                    "StateDependentMaskSignalPolicy probabilities cannot be negative "
                    f"for {state_name!r}."
                )
            normalized_distribution[normalized_mask] = probability_value
            specified_masks.add(normalized_mask)

        total_probability = sum(normalized_distribution.values())
        if total_probability <= 0.0:
            raise ValueError(
                "StateDependentMaskSignalPolicy probabilities must sum to a positive "
                f"value for {state_name!r}."
            )
        unspecified_masks = [
            mask for mask in all_masks if mask not in specified_masks
        ]
        if unspecified_masks and total_probability < 1.0:
            remaining_probability = 1.0 - total_probability
            fill_probability = remaining_probability / len(unspecified_masks)
            for mask in unspecified_masks:
                normalized_distribution[mask] = fill_probability
            return normalized_distribution

        return {
            mask: probability / total_probability
            for mask, probability in normalized_distribution.items()
        }
