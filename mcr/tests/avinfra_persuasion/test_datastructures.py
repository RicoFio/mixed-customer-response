from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcr.avinfra_persuasion.datastructures import (
    Demand,
    Individual,
    InfrastructureGraph,
    Scenario,
    World,
)


def _single_arc_world() -> tuple[World, tuple[str, str], Individual]:
    arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={arc},
        nominal_travel_time={arc: 2.0},
        nominal_link_capacity={arc: 1.0},
        arc_distances={arc: 1.0},
        nominal_discomfort={arc: 1.0},
        nominal_hazards={arc: 0.0},
        nominal_cost={arc: 0.0},
        nominal_policing={"s": 0.0, "t": 0.0},
    )
    individual = Individual(id="r0", demand=Demand(origin="s", destination="t"))
    world = World(network=network, individuals=frozenset({individual}))
    return world, arc, individual


def test_scenario_from_world_initializes_unit_travel_time_multipliers() -> None:
    world, arc, _ = _single_arc_world()

    scenario = Scenario.from_world("nominal", world)

    assert scenario.travel_time[arc] == pytest.approx(2.0)
    assert scenario.travel_time_multiplier[arc] == pytest.approx(1.0)


def test_scenario_multiplier_overrides_recompute_precongestion_travel_time() -> None:
    world, arc, _ = _single_arc_world()

    scenario = Scenario.from_world(
        "shifted",
        world,
        travel_time_multiplier_overrides={arc: 3.0},
    )

    assert scenario.travel_time_multiplier[arc] == pytest.approx(3.0)
    assert scenario.travel_time[arc] == pytest.approx(6.0)


def test_world_realization_uses_travel_time_multiplier() -> None:
    world, arc, individual = _single_arc_world()
    path_choice = {individual: SimpleNamespace(path=(arc,))}

    realized_nominal = world.get_realized_metrics(
        path_choices=path_choice,
        name="realized_nominal",
        base_scenario=Scenario.from_world("nominal", world),
    )
    realized_shifted = world.get_realized_metrics(
        path_choices=path_choice,
        name="realized_shifted",
        base_scenario=Scenario.from_world(
            "shifted",
            world,
            travel_time_multiplier_overrides={arc: 2.0},
        ),
    )

    assert realized_nominal.travel_time[arc] == pytest.approx(2.3)
    assert realized_shifted.travel_time[arc] == pytest.approx(4.6)
    assert realized_shifted.travel_time_multiplier[arc] == pytest.approx(2.0)
