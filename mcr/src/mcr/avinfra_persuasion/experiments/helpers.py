from __future__ import annotations

from collections.abc import Mapping

from ..bp.signals import Signal
from ..datastructures import (
    MetricName,
)

SignalObservationKey = tuple[
    tuple[MetricName, tuple[tuple[object, float], ...]],
    ...,
]

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
