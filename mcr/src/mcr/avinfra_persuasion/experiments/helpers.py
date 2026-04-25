from __future__ import annotations

from collections.abc import Mapping

from ..bp.signals import Signal
from ..datastructures import (
    Arc,
    MetricName,
    Scenario,
    World,
)

SignalObservationKey = tuple[
    tuple[MetricName, tuple[tuple[object, float], ...]],
    ...,
]


def scenario_matches_signal(
    *,
    scenario: Scenario,
    signal: Signal,
) -> bool:
    for metric, observed_value in signal.value.items():
        realized_value = getattr(scenario, metric.value)
        if isinstance(observed_value, Mapping):
            if not isinstance(realized_value, Mapping):
                return False
            for key, value in observed_value.items():
                if key not in realized_value or realized_value[key] != value:
                    return False
            continue
        if realized_value != observed_value:
            return False
    return True


def path_metric_totals(
    *,
    world: World,
    scenario: Scenario,
    path: tuple[Arc, ...],
) -> Mapping[MetricName, float]:
    left_turns = 0.0
    for previous_arc, next_arc in zip(path, path[1:]):
        turn = previous_arc[0], previous_arc[1], next_arc[1]
        if turn in world.L:
            left_turns += 1.0

    return {
        MetricName.TRAVEL_TIME: sum(scenario.travel_time[arc] for arc in path),
        MetricName.LEFT_TURNS: left_turns,
        MetricName.DISCOMFORT: sum(scenario.discomfort[arc] for arc in path),
        MetricName.HAZARD: sum(scenario.hazard[arc] for arc in path),
        MetricName.COST: sum(scenario.cost[arc] for arc in path),
        MetricName.EMISSIONS: sum(scenario.emissions[arc] for arc in path),
        MetricName.POLICING: sum(scenario.policing[arc[1]] for arc in path),
    }


def expected_path_metric_totals(
    *,
    world: World,
    path: tuple[Arc, ...],
    weighted_scenarios: tuple[tuple[float, Scenario], ...],
) -> Mapping[MetricName, float]:
    totals = {metric: 0.0 for metric in MetricName}
    for scenario_weight, scenario in weighted_scenarios:
        path_totals = path_metric_totals(
            world=world,
            scenario=scenario,
            path=path,
        )
        for metric, value in path_totals.items():
            totals[metric] += scenario_weight * value
    return totals


def sorted_metrics(
    metrics: frozenset[MetricName] | set[MetricName],
) -> tuple[MetricName, ...]:
    return tuple(sorted(metrics, key=lambda metric: metric.value))


def canonical_signal_key(signal: Signal) -> SignalObservationKey:
    key: list[tuple[MetricName, tuple[tuple[object, float], ...]]] = []
    for metric in sorted_metrics(frozenset(signal.metrics)):
        observed_values = signal.value.get(metric, {})
        ordered_items = tuple(
            sorted(observed_values.items(), key=lambda item: str(item[0]))
        )
        key.append((metric, ordered_items))
    return tuple(key)


def format_signal_key(signal_key: SignalObservationKey) -> str:
    if not signal_key:
        return "<empty>"

    pieces: list[str] = []
    for metric, entries in signal_key:
        if not entries:
            pieces.append(metric.value)
            continue
        entry_text = ", ".join(
            f"{location!r}={value:.3f}" for location, value in entries
        )
        pieces.append(f"{metric.value}: {entry_text}")
    return " | ".join(pieces)


def format_mask(mask: tuple[MetricName, ...] | frozenset[MetricName]) -> str:
    if not mask:
        return "<empty>"
    return ", ".join(metric.value for metric in sorted_metrics(frozenset(mask)))


def format_posterior(probabilities: Mapping[str, float]) -> str:
    return ", ".join(
        f"{name}={probability:.3f}" for name, probability in probabilities.items()
    )

