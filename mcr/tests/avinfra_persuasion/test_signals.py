from __future__ import annotations

import numpy as np
import pytest

from mcr.avinfra_persuasion.bp.signals import (
    MaskSignal,
    MaskSignalPolicy,
    StateDependentMaskSignalPolicy,
)
from mcr.avinfra_persuasion.datastructures import Scenario
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
    assert signal.value == {}


def test_state_dependent_mask_signal_policy_defaults_to_uniform_per_state() -> None:
    policy = StateDependentMaskSignalPolicy(
        state_names=frozenset({"fast", "slow"}),
        considered_metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
    )

    assert policy.mask_probability("fast", frozenset()) == pytest.approx(0.25)
    assert policy.mask_probability(
        "slow",
        frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
    ) == pytest.approx(0.25)


def test_state_dependent_mask_signal_policy_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError, match="unknown states"):
        StateDependentMaskSignalPolicy(
            state_names=frozenset({"fast"}),
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            state_probabilities={"slow": {frozenset(): 1.0}},
        )

    policy = StateDependentMaskSignalPolicy(
        state_names=frozenset({"fast"}),
        considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
    )
    with pytest.raises(ValueError, match="outside the considered set"):
        policy.update_state_distribution(
            "fast",
            {frozenset({MetricName.HAZARD}): 1.0},
        )


def test_state_dependent_mask_signal_policy_normalizes_per_state_distribution() -> None:
    policy = StateDependentMaskSignalPolicy(
        state_names=frozenset({"fast"}),
        considered_metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
        state_probabilities={
            "fast": {
                frozenset({MetricName.TRAVEL_TIME}): 2.0,
                frozenset({MetricName.HAZARD}): 1.0,
            }
        },
    )

    distribution = policy.distribution_for_state("fast")
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution[frozenset({MetricName.TRAVEL_TIME})] == pytest.approx(2.0 / 3.0)
    assert distribution[frozenset({MetricName.HAZARD})] == pytest.approx(1.0 / 3.0)
    assert distribution[frozenset()] == pytest.approx(0.0)


def test_state_dependent_mask_signal_policy_samples_state_conditionally() -> None:
    policy = StateDependentMaskSignalPolicy(
        state_names=frozenset({"fast", "slow"}),
        considered_metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
        state_probabilities={
            "fast": {
                frozenset({MetricName.TRAVEL_TIME}): 1.0,
            },
            "slow": {
                frozenset({MetricName.HAZARD}): 1.0,
            },
        },
    )
    scenario = Scenario(
        name="slow",
        travel_time={("s", "t"): 1.0},
        discomfort={("s", "t"): 0.0},
        hazard={("s", "t"): 0.0},
        cost={("s", "t"): 0.0},
        emissions={("s", "t"): 0.0},
        policing={"s": 0.0, "t": 0.0},
    )

    signal = policy.sample(
        realized_scenario=scenario,
        rng=np.random.default_rng(3),
    )

    assert signal.metrics == frozenset({MetricName.HAZARD})
