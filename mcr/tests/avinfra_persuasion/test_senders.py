from __future__ import annotations

import pytest

from mcr.avinfra_persuasion.bp.senders import ScalarSender
from mcr.avinfra_persuasion.bp.signals import (
    MaskSignal,
    MaskSignalPolicy,
    StateDependentMaskSignalPolicy,
    TypedStateDependentMaskSignalPolicy,
)
from mcr.avinfra_persuasion.datastructures import (
    Demand,
    FinitePrior,
    Individual,
    InfrastructureGraph,
    MetricName,
    Scenario,
    World,
)
from mcr.avinfra_persuasion.orders import total_order_from_list


def test_sender_emits_instrumented_partial_mask_signal() -> None:
    instrumented_arc = ("s", "m")
    other_arc = ("m", "t")
    network = InfrastructureGraph(
        V={"s", "m", "t"},
        A={instrumented_arc, other_arc},
        I={instrumented_arc},
        nominal_travel_time={instrumented_arc: 1.0, other_arc: 2.0},
        nominal_discomfort={instrumented_arc: 0.5, other_arc: 1.0},
        nominal_hazards={instrumented_arc: 0.1, other_arc: 0.2},
        nominal_cost={instrumented_arc: 3.0, other_arc: 4.0},
        nominal_policing={"s": 0.0, "m": 1.0, "t": 2.0},
    )
    world = World(
        network=network,
        individuals=frozenset(
            {
                Individual(
                    id="receiver",
                    demand=Demand(origin="s", destination="t"),
                )
            }
        ),
    )
    scenario = Scenario(
        name="rho",
        travel_time={instrumented_arc: 5.0, other_arc: 7.0},
        discomfort={instrumented_arc: 0.4, other_arc: 1.1},
        hazard={instrumented_arc: 0.7, other_arc: 0.9},
        cost={instrumented_arc: 8.0, other_arc: 9.0},
        emissions={instrumented_arc: 2.0, other_arc: 3.0},
        policing={"s": 10.0, "m": 20.0, "t": 30.0},
    )
    prior = FinitePrior(
        name="prior",
        support={scenario.name: scenario},
        probabilities={scenario.name: 1.0},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=MaskSignalPolicy(
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    MetricName.POLICING,
                    MetricName.HAZARD,
                }
            ),
            probabilities={
                MetricName.TRAVEL_TIME: 1.0,
                MetricName.POLICING: 1.0,
                MetricName.HAZARD: 0.0,
            },
        ),
    )

    signal = sender.materialize_signal(
        mask=frozenset({MetricName.TRAVEL_TIME, MetricName.POLICING}),
        realized_scenario=scenario,
    )

    assert signal.metrics == frozenset(
        {MetricName.TRAVEL_TIME, MetricName.POLICING}
    )
    assert signal.value[MetricName.TRAVEL_TIME] == {instrumented_arc: 5.0}
    assert signal.value[MetricName.POLICING] == {"s": 10.0, "m": 20.0}
    assert MetricName.HAZARD not in signal.value

    sampled_signal = sender.emit_signal(scenario)
    assert sampled_signal.metrics == frozenset(
        {MetricName.TRAVEL_TIME, MetricName.POLICING}
    )


def test_sender_signal_likelihood_uses_mask_probability_for_matching_signal() -> None:
    instrumented_arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={instrumented_arc},
        I={instrumented_arc},
        nominal_travel_time={instrumented_arc: 1.0},
        nominal_discomfort={instrumented_arc: 0.5},
        nominal_hazards={instrumented_arc: 0.1},
        nominal_cost={instrumented_arc: 2.0},
        nominal_policing={"s": 0.0, "t": 1.0},
    )
    world = World(
        network=network,
        individuals=frozenset(
            {
                Individual(
                    id="receiver",
                    demand=Demand(origin="s", destination="t"),
                )
            }
        ),
    )
    scenario = Scenario.from_world("rho", world)
    prior = FinitePrior(
        name="prior",
        support={scenario.name: scenario},
        probabilities={scenario.name: 1.0},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=MaskSignalPolicy(
            considered_metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
            probabilities={
                MetricName.TRAVEL_TIME: 0.8,
                MetricName.HAZARD: 0.25,
            },
        ),
    )
    signal = sender.materialize_signal(
        mask=frozenset({MetricName.TRAVEL_TIME}),
        realized_scenario=scenario,
    )

    assert sender.signal_likelihood(signal, scenario) == pytest.approx(0.8 * 0.75)


def test_sender_signal_likelihood_uses_state_dependent_probability() -> None:
    instrumented_arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={instrumented_arc},
        I={instrumented_arc},
        nominal_travel_time={instrumented_arc: 1.0},
        nominal_discomfort={instrumented_arc: 0.5},
        nominal_hazards={instrumented_arc: 0.1},
        nominal_cost={instrumented_arc: 2.0},
        nominal_policing={"s": 0.0, "t": 1.0},
    )
    world = World(
        network=network,
        individuals=frozenset(
            {
                Individual(
                    id="receiver",
                    demand=Demand(origin="s", destination="t"),
                )
            }
        ),
    )
    fast = Scenario.from_world("fast", world)
    slow = Scenario.from_world("slow", world)
    prior = FinitePrior(
        name="prior",
        support={fast.name: fast, slow.name: slow},
        probabilities={fast.name: 0.5, slow.name: 0.5},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=StateDependentMaskSignalPolicy(
            state_names=frozenset(prior.support),
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            state_probabilities={
                "fast": {frozenset({MetricName.TRAVEL_TIME}): 0.9},
                "slow": {frozenset({MetricName.TRAVEL_TIME}): 0.1},
            },
        ),
    )
    fast_signal = sender.materialize_signal(
        mask=frozenset({MetricName.TRAVEL_TIME}),
        realized_scenario=fast,
    )

    assert sender.signal_likelihood(fast_signal, fast) == pytest.approx(0.9)
    assert sender.signal_likelihood(fast_signal, slow) == pytest.approx(0.1)


def test_sender_signal_likelihood_uses_typed_state_dependent_probability() -> None:
    instrumented_arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={instrumented_arc},
        I={instrumented_arc},
        nominal_travel_time={instrumented_arc: 1.0},
        nominal_discomfort={instrumented_arc: 0.5},
        nominal_hazards={instrumented_arc: 0.1},
        nominal_cost={instrumented_arc: 2.0},
        nominal_policing={"s": 0.0, "t": 1.0},
    )
    world = World(
        network=network,
        individuals=frozenset(
            {
                Individual(
                    id="receiver",
                    demand=Demand(origin="s", destination="t"),
                )
            }
        ),
    )
    scenario = Scenario.from_world("fast", world)
    prior = FinitePrior(
        name="prior",
        support={scenario.name: scenario},
        probabilities={scenario.name: 1.0},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=TypedStateDependentMaskSignalPolicy(
            type_names=frozenset({"human", "av"}),
            state_names=frozenset(prior.support),
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            state_type_probabilities={
                "fast": {
                    "human": {frozenset({MetricName.TRAVEL_TIME}): 0.9},
                    "av": {frozenset({MetricName.TRAVEL_TIME}): 0.2},
                },
            },
        ),
    )
    signal = sender.materialize_signal(
        mask=frozenset({MetricName.TRAVEL_TIME}),
        realized_scenario=scenario,
    )

    assert sender.signal_likelihood(
        signal,
        scenario,
        receiver_type="human",
    ) == pytest.approx(0.9)
    assert sender.signal_likelihood(
        signal,
        scenario,
        receiver_type="av",
    ) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="receiver type"):
        sender.signal_likelihood(signal, scenario)


def test_sender_signal_likelihood_rejects_mismatched_truthful_values() -> None:
    instrumented_arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={instrumented_arc},
        I={instrumented_arc},
        nominal_travel_time={instrumented_arc: 1.0},
        nominal_discomfort={instrumented_arc: 0.5},
        nominal_hazards={instrumented_arc: 0.1},
        nominal_cost={instrumented_arc: 2.0},
        nominal_policing={"s": 0.0, "t": 1.0},
    )
    world = World(
        network=network,
        individuals=frozenset(
            {
                Individual(
                    id="receiver",
                    demand=Demand(origin="s", destination="t"),
                )
            }
        ),
    )
    scenario = Scenario.from_world("rho", world)
    prior = FinitePrior(
        name="prior",
        support={scenario.name: scenario},
        probabilities={scenario.name: 1.0},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=MaskSignalPolicy(
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            probabilities={MetricName.TRAVEL_TIME: 1.0},
        ),
    )
    mismatched_signal = MaskSignal(
        metrics=frozenset({MetricName.TRAVEL_TIME}),
        value={MetricName.TRAVEL_TIME: {instrumented_arc: 99.0}},
    )

    assert sender.signal_likelihood(mismatched_signal, scenario) == 0.0
