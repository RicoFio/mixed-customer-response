from __future__ import annotations

from mcr.avinfra_persuasion.bp.senders import ScalarSender
from mcr.avinfra_persuasion.bp.signals import MaskSignalPolicy
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
