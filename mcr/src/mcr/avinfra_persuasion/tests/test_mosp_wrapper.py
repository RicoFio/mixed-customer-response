from __future__ import annotations

import pytest

from mcr.avinfra_persuasion.datastructures import (
    Arc,
    Demand,
    Individual,
    InfrastructureGraph,
    MetricName,
    Scenario,
    World,
)
from mcr.avinfra_persuasion.mosp import UINT32_MAX, solve_mosp_routes
from mcr.avinfra_persuasion.orders import total_order_from_list


def _world_and_scenario(
    *,
    arcs: list[Arc],
    source: str,
    target: str,
    travel_time: dict[Arc, float],
    discomfort: dict[Arc, float] | None = None,
    left_turns: set[tuple[str, str, str]] | None = None,
) -> tuple[World, Scenario]:
    nodes = {node for arc in arcs for node in arc}
    zeros = {arc: 0.0 for arc in arcs}
    discomfort = discomfort or zeros
    policing = {node: 0.0 for node in nodes}
    network = InfrastructureGraph(
        V=nodes,
        A=set(arcs),
        L=left_turns or set(),
        nominal_travel_time=travel_time,
        arc_distances={arc: abs(value) for arc, value in travel_time.items()},
        nominal_discomfort=discomfort,
        nominal_hazards=zeros,
        nominal_cost=zeros,
        nominal_policing=policing,
    )
    world = World(
        network=network,
        individuals=frozenset(
            {
                Individual(
                    id="receiver",
                    demand=Demand(origin=source, destination=target),
                )
            }
        ),
    )
    scenario = Scenario(
        name="rho0",
        travel_time=travel_time,
        discomfort=discomfort,
        hazard=zeros,
        cost=zeros,
        emissions=zeros,
        policing=policing,
    )
    return world, scenario


def test_mosp_wrapper_returns_mcr_routing_solution_points() -> None:
    arcs = [
        ("s", "a"),
        ("a", "t"),
        ("s", "b"),
        ("b", "t"),
        ("s", "t"),
    ]
    world, scenario = _world_and_scenario(
        arcs=arcs,
        source="s",
        target="t",
        travel_time={
            ("s", "a"): 1.0,
            ("a", "t"): 1.0,
            ("s", "b"): 3.0,
            ("b", "t"): 3.0,
            ("s", "t"): 4.0,
        },
        discomfort={
            ("s", "a"): 5.0,
            ("a", "t"): 5.0,
            ("s", "b"): 1.0,
            ("b", "t"): 1.0,
            ("s", "t"): 4.0,
        },
    )

    solution = solve_mosp_routes(world, "s", "t", [scenario])

    assert solution.status == "optimal"
    assert set(solution.paths.values()) == {
        (("s", "a"), ("a", "t")),
        (("s", "b"), ("b", "t")),
        (("s", "t"),),
    }
    assert solution.objective_names == (
        MetricName.TRAVEL_TIME,
        MetricName.LEFT_TURNS,
        MetricName.DISCOMFORT,
        MetricName.HAZARD,
        MetricName.COST,
        MetricName.EMISSIONS,
        MetricName.POLICING,
    )
    assert {
        (
            point.objective_values[MetricName.TRAVEL_TIME],
            point.objective_values[MetricName.DISCOMFORT],
        )
        for point in solution.points
    } == {(2.0, 10.0), (6.0, 2.0), (4.0, 4.0)}


def test_mosp_wrapper_counts_turn_state_left_turns() -> None:
    arcs = [
        ("s", "a"),
        ("a", "b"),
        ("b", "t"),
        ("a", "t"),
    ]
    world, scenario = _world_and_scenario(
        arcs=arcs,
        source="s",
        target="t",
        left_turns={("s", "a", "b")},
        travel_time={
            ("s", "a"): 1.0,
            ("a", "b"): 1.0,
            ("b", "t"): 1.0,
            ("a", "t"): 4.0,
        },
    )

    solution = solve_mosp_routes(world, "s", "t", [scenario])

    assert set(solution.paths.values()) == {
        (("s", "a"), ("a", "b"), ("b", "t")),
        (("s", "a"), ("a", "t")),
    }
    assert {
        point.objective_values[MetricName.LEFT_TURNS]
        for point in solution.points
    } == {0.0, 1.0}


def test_mosp_wrapper_rejects_negative_and_overflowing_scaled_costs() -> None:
    arcs = [("s", "t")]
    world, scenario = _world_and_scenario(
        arcs=arcs,
        source="s",
        target="t",
        travel_time={("s", "t"): -1.0},
    )

    with pytest.raises(ValueError, match="non-negative"):
        solve_mosp_routes(world, "s", "t", [scenario])

    world, scenario = _world_and_scenario(
        arcs=arcs,
        source="s",
        target="t",
        travel_time={("s", "t"): float(UINT32_MAX)},
    )

    with pytest.raises(OverflowError, match="uint32_t"):
        solve_mosp_routes(world, "s", "t", [scenario], scale=2.0)


def test_mosp_solution_works_with_existing_helpers() -> None:
    arcs = [
        ("s", "a"),
        ("a", "t"),
        ("s", "b"),
        ("b", "t"),
    ]
    world, scenario = _world_and_scenario(
        arcs=arcs,
        source="s",
        target="t",
        travel_time={
            ("s", "a"): 1.0,
            ("a", "t"): 1.0,
            ("s", "b"): 3.0,
            ("b", "t"): 3.0,
        },
        discomfort={
            ("s", "a"): 5.0,
            ("a", "t"): 5.0,
            ("s", "b"): 1.0,
            ("b", "t"): 1.0,
        },
    )
    solution = solve_mosp_routes(world, "s", "t", [scenario])

    assert solution.best_by_metric(MetricName.TRAVEL_TIME).path == (
        ("s", "a"),
        ("a", "t"),
    )
    preference = total_order_from_list(
        [MetricName.DISCOMFORT, MetricName.TRAVEL_TIME]
    )
    assert solution.induced_preorder(preference).maximal_elements()
    assert solution.maximal_results(preference)

    individual = next(iter(world.individuals))
    realized = world.get_realized_metrics(
        {individual: solution.best_by_metric(MetricName.TRAVEL_TIME)}
    )
    assert set(realized.travel_time).issuperset(solution.best_by_metric(MetricName.TRAVEL_TIME).path)
