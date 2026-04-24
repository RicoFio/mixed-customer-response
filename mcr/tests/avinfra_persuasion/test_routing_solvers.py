from __future__ import annotations

import pytest

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
from mcr.avinfra_persuasion.bp.receivers import Receiver
from mcr.avinfra_persuasion.orders import total_order_from_list
from mcr.avinfra_persuasion.routing.routing_solvers import (
    RoutingSolverConfig,
    solve_routes,
)
from mcr.avinfra_persuasion.toy_setup import solve_toy_network


def _world_and_scenario(
    *,
    arcs: list[Arc],
    source: str,
    target: str,
    travel_time: dict[Arc, float],
    discomfort: dict[Arc, float],
) -> tuple[World, Scenario]:
    nodes = {node for arc in arcs for node in arc}
    zeros = {arc: 0.0 for arc in arcs}
    policing = {node: 0.0 for node in nodes}
    network = InfrastructureGraph(
        V=nodes,
        A=set(arcs),
        nominal_travel_time=travel_time,
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


def _tiny_world() -> tuple[World, Scenario]:
    arcs = [
        ("s", "a"),
        ("a", "t"),
        ("s", "b"),
        ("b", "t"),
    ]
    return _world_and_scenario(
        arcs=arcs,
        source="s",
        target="t",
        travel_time={
            ("s", "a"): 1.0,
            ("a", "t"): 1.0,
            ("s", "b"): 4.0,
            ("b", "t"): 4.0,
        },
        discomfort={
            ("s", "a"): 4.0,
            ("a", "t"): 4.0,
            ("s", "b"): 1.0,
            ("b", "t"): 1.0,
        },
    )


def test_solve_routes_defaults_to_benpy() -> None:
    world, scenario = _tiny_world()

    solution = solve_routes(
        world=world,
        source="s",
        target="t",
        scenarios=[scenario],
    )

    assert solution.model.meta["formulation"] == "turn_state"
    assert solution.paths


def test_benpy_and_mosp_backends_match_on_tiny_graph() -> None:
    world, scenario = _tiny_world()

    benpy_solution = solve_routes(
        world=world,
        source="s",
        target="t",
        scenarios=[scenario],
        config="benpy",
    )
    mosp_solution = solve_routes(
        world=world,
        source="s",
        target="t",
        scenarios=[scenario],
        config=RoutingSolverConfig(backend="mosp"),
    )

    assert set(benpy_solution.paths.values()) == set(mosp_solution.paths.values())

    benpy_by_path = {
        point.path: point.objective_values
        for point in benpy_solution.points
    }
    for point in mosp_solution.points:
        assert benpy_by_path[point.path][MetricName.TRAVEL_TIME] == pytest.approx(
            point.objective_values[MetricName.TRAVEL_TIME]
        )
        assert benpy_by_path[point.path][MetricName.DISCOMFORT] == pytest.approx(
            point.objective_values[MetricName.DISCOMFORT]
        )


def test_receiver_can_use_mosp_backend_for_path_choice_and_metrics() -> None:
    world, scenario = _tiny_world()
    prior = FinitePrior(
        name="single",
        support={scenario.name: scenario},
        probabilities={scenario.name: 1.0},
    )
    receiver = Receiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list(
            [MetricName.DISCOMFORT, MetricName.TRAVEL_TIME]
        ),
        prior=prior,
        world=world,
        n_scenarios=1,
        routing_solver_config=RoutingSolverConfig(backend="mosp"),
    )

    choice = receiver.get_path_choice()
    realized_metrics = receiver.compute_realized_metrics(scenario)

    assert choice.path in {
        (("s", "a"), ("a", "t")),
        (("s", "b"), ("b", "t")),
    }
    assert realized_metrics[MetricName.TRAVEL_TIME] >= 0.0


def test_mosp_rejects_non_turn_state_requests() -> None:
    world, scenario = _tiny_world()

    with pytest.raises(ValueError, match="use_turn_state=True"):
        solve_routes(
            world=world,
            source="s",
            target="t",
            scenarios=[scenario],
            config=RoutingSolverConfig(backend="mosp", use_turn_state=False),
        )


def test_solve_toy_network_accepts_solver_backend_string() -> None:
    world, _ = _tiny_world()

    _, solution = solve_toy_network(
        world=world,
        n_samples=1,
        rel_noise=0.0,
        solver="mosp",
    )

    assert solution.model.meta["solver"] == "mda"
    assert solution.paths
