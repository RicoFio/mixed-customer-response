from __future__ import annotations

import numpy as np
import pytest

from mcr.avinfra_persuasion.bp.signals import MaskSignal, MaskSignalPolicy
from mcr.avinfra_persuasion.datastructures import MetricName


def test_mask_signal_coerces_metric_names() -> None:
    signal = MaskSignal(metrics=frozenset({"travel_time", MetricName.HAZARD}))

    assert signal.metrics == frozenset(
        {MetricName.TRAVEL_TIME, MetricName.HAZARD}
    )
    assert signal.reveals("travel_time")
    assert not signal.reveals(MetricName.COST)


def test_mask_signal_policy_defaults_to_half_probability() -> None:
    policy = MaskSignalPolicy(
        considered_metrics={MetricName.TRAVEL_TIME, MetricName.HAZARD},
    )

    assert policy.probability(MetricName.TRAVEL_TIME) == 0.5
    assert policy.probability(MetricName.HAZARD) == 0.5


def test_mask_signal_policy_updates_probabilities() -> None:
    policy = MaskSignalPolicy(
        considered_metrics={MetricName.TRAVEL_TIME, MetricName.HAZARD},
        probabilities={MetricName.TRAVEL_TIME: 0.2},
    )

    policy.update_probability(MetricName.TRAVEL_TIME, 0.9)
    policy.update_probabilities({MetricName.HAZARD: 0.1})

    assert policy.probability(MetricName.TRAVEL_TIME) == 0.9
    assert policy.probability(MetricName.HAZARD) == 0.1


def test_mask_signal_policy_rejects_invalid_updates() -> None:
    policy = MaskSignalPolicy(considered_metrics={MetricName.TRAVEL_TIME})

    with pytest.raises(ValueError, match="not part of this mask policy"):
        policy.update_probability(MetricName.HAZARD, 0.3)

    with pytest.raises(ValueError, match="must lie in \\[0, 1\\]"):
        policy.update_probability(MetricName.TRAVEL_TIME, 1.2)


def test_mask_signal_policy_samples_mask_with_seeded_rng() -> None:
    policy = MaskSignalPolicy(
        considered_metrics={
            MetricName.TRAVEL_TIME,
            MetricName.HAZARD,
            MetricName.COST,
        },
        probabilities={
            MetricName.TRAVEL_TIME: 1.0,
            MetricName.HAZARD: 0.0,
            MetricName.COST: 1.0,
        },
    )

    signal = policy.sample(rng=np.random.default_rng(3))

    assert signal.metrics == frozenset(
        {MetricName.TRAVEL_TIME, MetricName.COST}
    )
