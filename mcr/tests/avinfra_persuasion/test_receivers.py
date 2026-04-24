from __future__ import annotations

from mcr.avinfra_persuasion.bp.receivers import Receiver
from mcr.avinfra_persuasion.bp.signals import Signal
from mcr.avinfra_persuasion.datastructures import (
    Arc,
    Demand,
    FinitePrior,
    Individual,
    InfrastructureGraph,
    MetricName,
    Scenario,
    World,
)
from mcr.avinfra_persuasion.orders import total_order_from_list


def _receiver_world() -> tuple[World, Arc]:
    arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={arc},
        nominal_travel_time={arc: 1.0},
        nominal_discomfort={arc: 1.0},
        nominal_hazards={arc: 0.0},
        nominal_cost={arc: 0.0},
        nominal_policing={"s": 0.0, "t": 0.0},
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
    return world, arc


def _receiver_with_prior(prior: FinitePrior, world: World) -> Receiver:
    return Receiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        prior=prior,
        world=world,
        n_scenarios=1,
    )


def test_update_internal_belief_filters_finite_prior_support() -> None:
    world, arc = _receiver_world()
    scenario_fast = Scenario(
        name="fast",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    scenario_slow = Scenario(
        name="slow",
        travel_time={arc: 3.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    prior = FinitePrior(
        name="prior",
        support={
            scenario_fast.name: scenario_fast,
            scenario_slow.name: scenario_slow,
        },
        probabilities={
            scenario_fast.name: 0.25,
            scenario_slow.name: 0.75,
        },
    )
    receiver = _receiver_with_prior(prior, world)

    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.TRAVEL_TIME}),
            value={MetricName.TRAVEL_TIME: {arc: 1.0}},
        )
    )

    posterior = receiver.belief
    assert isinstance(posterior, FinitePrior)
    assert set(posterior.support) == {"fast"}
    assert posterior.probabilities["fast"] == 1.0


def test_update_internal_belief_is_noop_for_mask_only_signal() -> None:
    world, arc = _receiver_world()
    scenario = Scenario(
        name="rho0",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    prior = FinitePrior(
        name="prior",
        support={scenario.name: scenario},
        probabilities={scenario.name: 1.0},
    )
    receiver = _receiver_with_prior(prior, world)

    receiver.update_internal_belief(
        Signal(metrics=frozenset({MetricName.TRAVEL_TIME}))
    )

    assert receiver.belief == prior
