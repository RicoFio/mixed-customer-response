from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from mcr.avinfra_persuasion.bp import receivers as receivers_module
from mcr.avinfra_persuasion.bp.receivers import Receiver
from mcr.avinfra_persuasion.bp.senders import ScalarSender
from mcr.avinfra_persuasion.bp.signals import (
    MaskSignalPolicy,
    Signal,
    StateDependentMaskSignalPolicy,
)
from mcr.avinfra_persuasion.datastructures import (
    Arc,
    Demand,
    FinitePrior,
    Individual,
    InfrastructureGraph,
    MetricName,
    SampledPrior,
    Scenario,
    World,
)
from mcr.avinfra_persuasion.opt import RoutingSolution, RoutingSolutionPoint
from mcr.avinfra_persuasion.orders import total_order_from_list


def _receiver_world() -> tuple[World, Arc]:
    arc = ("s", "t")
    network = InfrastructureGraph(
        V={"s", "t"},
        A={arc},
        I={arc},
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


def _two_route_world() -> tuple[World, Arc, Arc, Arc]:
    direct = ("s", "t")
    via_left = ("s", "m")
    via_right = ("m", "t")
    network = InfrastructureGraph(
        V={"s", "m", "t"},
        A={direct, via_left, via_right},
        I={direct},
        nominal_travel_time={direct: 1.0, via_left: 1.0, via_right: 1.0},
        nominal_discomfort={direct: 1.0, via_left: 1.0, via_right: 1.0},
        nominal_hazards={direct: 0.0, via_left: 0.0, via_right: 0.0},
        nominal_cost={direct: 0.0, via_left: 0.0, via_right: 0.0},
        nominal_policing={"s": 0.0, "m": 0.0, "t": 0.0},
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
    return world, direct, via_left, via_right


def _solution_for_points(
    points: tuple[RoutingSolutionPoint, ...],
) -> RoutingSolution:
    model = SimpleNamespace(meta={"objective_names": tuple(MetricName)})
    raw_solution = SimpleNamespace(status="optimal")
    return RoutingSolution(
        raw_solution=raw_solution,
        model=cast(Any, model),
        points=points,
    )


def _point(
    label: str,
    path: tuple[Arc, ...],
    *,
    travel_time: float,
    cost: float = 0.0,
) -> RoutingSolutionPoint:
    objective_values = {metric: 0.0 for metric in MetricName}
    objective_values[MetricName.TRAVEL_TIME] = travel_time
    objective_values[MetricName.COST] = cost
    return RoutingSolutionPoint(
        label=label,
        index=0,
        objective_values=objective_values,
        path=path,
        arc_flows={arc: 1.0 for arc in path},
        variable_values=(1.0,),
        vertex_type=1,
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
        support={"fast": scenario_fast, "slow": scenario_slow},
        probabilities={"fast": 0.25, "slow": 0.75},
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


def test_update_internal_belief_filters_node_observations() -> None:
    world, arc = _receiver_world()
    scenario_low = Scenario(
        name="low",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    scenario_high = Scenario(
        name="high",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 1.0},
    )
    prior = FinitePrior(
        name="prior",
        support={"low": scenario_low, "high": scenario_high},
        probabilities={"low": 0.5, "high": 0.5},
    )
    receiver = _receiver_with_prior(prior, world)

    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.POLICING}),
            value={MetricName.POLICING: {"t": 1.0}},
        )
    )

    posterior = receiver.belief
    assert isinstance(posterior, FinitePrior)
    assert set(posterior.support) == {"high"}


def test_update_internal_belief_filters_mixed_partial_observations() -> None:
    world, arc = _receiver_world()
    scenario_a = Scenario(
        name="a",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    scenario_b = Scenario(
        name="b",
        travel_time={arc: 2.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 1.0},
    )
    prior = FinitePrior(
        name="prior",
        support={"a": scenario_a, "b": scenario_b},
        probabilities={"a": 0.5, "b": 0.5},
    )
    receiver = _receiver_with_prior(prior, world)

    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.POLICING}),
            value={
                MetricName.TRAVEL_TIME: {arc: 2.0},
                MetricName.POLICING: {"t": 1.0},
            },
        )
    )

    posterior = receiver.belief
    assert isinstance(posterior, FinitePrior)
    assert set(posterior.support) == {"b"}


def test_update_internal_belief_rejects_inconsistent_signal() -> None:
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

    with pytest.raises(ValueError, match="inconsistent"):
        receiver.update_internal_belief(
            Signal(
                metrics=frozenset({MetricName.TRAVEL_TIME}),
                value={MetricName.TRAVEL_TIME: {arc: 9.0}},
            )
        )


def test_update_internal_belief_rejects_non_finite_belief() -> None:
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
    prior = SampledPrior(
        name="sampled",
        sampler=lambda _rng, n_samples: [scenario] * n_samples,
    )
    receiver = Receiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        prior=prior,
        world=world,
        n_scenarios=1,
    )

    with pytest.raises(NotImplementedError, match="FinitePrior"):
        receiver.update_internal_belief(
            Signal(
                metrics=frozenset({MetricName.TRAVEL_TIME}),
                value={MetricName.TRAVEL_TIME: {arc: 1.0}},
            )
        )


def test_update_internal_belief_uses_state_dependent_mask_likelihood() -> None:
    world, arc = _receiver_world()
    scenario_a = Scenario(
        name="a",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    scenario_b = Scenario(
        name="b",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    prior = FinitePrior(
        name="prior",
        support={"a": scenario_a, "b": scenario_b},
        probabilities={"a": 0.5, "b": 0.5},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=StateDependentMaskSignalPolicy(
            state_names=frozenset(prior.support),
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            state_probabilities={
                "a": {frozenset({MetricName.TRAVEL_TIME}): 0.9},
                "b": {frozenset({MetricName.TRAVEL_TIME}): 0.1},
            },
        ),
    )
    receiver = Receiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        prior=prior,
        world=world,
        sender=sender,
        n_scenarios=1,
    )

    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.TRAVEL_TIME}),
            value={MetricName.TRAVEL_TIME: {arc: 1.0}},
        )
    )

    posterior = receiver.belief
    assert isinstance(posterior, FinitePrior)
    assert posterior.probabilities["a"] == pytest.approx(0.9)
    assert posterior.probabilities["b"] == pytest.approx(0.1)


def test_update_internal_belief_keeps_old_behavior_for_state_independent_mask() -> None:
    world, arc = _receiver_world()
    scenario_a = Scenario(
        name="a",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    scenario_b = Scenario(
        name="b",
        travel_time={arc: 1.0},
        discomfort={arc: 1.0},
        hazard={arc: 0.0},
        cost={arc: 0.0},
        emissions={arc: 0.0},
        policing={"s": 0.0, "t": 0.0},
    )
    prior = FinitePrior(
        name="prior",
        support={"a": scenario_a, "b": scenario_b},
        probabilities={"a": 0.5, "b": 0.5},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        signal_policy=MaskSignalPolicy(
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            probabilities={MetricName.TRAVEL_TIME: 0.25},
        ),
    )
    receiver = Receiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        prior=prior,
        world=world,
        sender=sender,
        n_scenarios=1,
    )

    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.TRAVEL_TIME}),
            value={MetricName.TRAVEL_TIME: {arc: 1.0}},
        )
    )

    posterior = receiver.belief
    assert isinstance(posterior, FinitePrior)
    assert posterior.probabilities == {"a": 0.5, "b": 0.5}


def test_receiver_compute_paths_uses_posterior_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, direct, via_left, via_right = _two_route_world()
    scenario_direct = Scenario(
        name="direct",
        travel_time={direct: 1.0, via_left: 3.0, via_right: 3.0},
        discomfort={direct: 1.0, via_left: 1.0, via_right: 1.0},
        hazard={direct: 0.0, via_left: 0.0, via_right: 0.0},
        cost={direct: 0.0, via_left: 0.0, via_right: 0.0},
        emissions={direct: 0.0, via_left: 0.0, via_right: 0.0},
        policing={"s": 0.0, "m": 0.0, "t": 0.0},
    )
    scenario_detour = Scenario(
        name="detour",
        travel_time={direct: 9.0, via_left: 1.0, via_right: 1.0},
        discomfort={direct: 1.0, via_left: 1.0, via_right: 1.0},
        hazard={direct: 0.0, via_left: 0.0, via_right: 0.0},
        cost={direct: 0.0, via_left: 0.0, via_right: 0.0},
        emissions={direct: 0.0, via_left: 0.0, via_right: 0.0},
        policing={"s": 0.0, "m": 0.0, "t": 0.0},
    )
    prior = FinitePrior(
        name="prior",
        support={"direct": scenario_direct, "detour": scenario_detour},
        probabilities={"direct": 0.8, "detour": 0.2},
    )
    receiver = Receiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        prior=prior,
        world=world,
        n_scenarios=3,
    )
    recorded_scenarios: list[tuple[Scenario, ...]] = []

    def fake_solve_routes(**kwargs: Any) -> RoutingSolution:
        recorded_scenarios.append(tuple(kwargs["scenarios"]))
        return _solution_for_points((_point("A", (direct,), travel_time=1.0),))

    monkeypatch.setattr(receivers_module, "solve_routes", fake_solve_routes)

    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.TRAVEL_TIME}),
            value={MetricName.TRAVEL_TIME: {direct: 9.0}},
        )
    )
    receiver.get_path_choice()

    assert recorded_scenarios == [
        (scenario_detour, scenario_detour, scenario_detour)
    ]


def test_route_choice_receiver_caches_prior_paths_and_rescores_posterior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, direct, via_left, via_right = _two_route_world()
    scenario_direct = Scenario(
        name="direct",
        travel_time={direct: 1.0, via_left: 3.0, via_right: 3.0},
        discomfort={direct: 1.0, via_left: 1.0, via_right: 1.0},
        hazard={direct: 0.0, via_left: 0.0, via_right: 0.0},
        cost={direct: 1.0, via_left: 5.0, via_right: 5.0},
        emissions={direct: 0.0, via_left: 0.0, via_right: 0.0},
        policing={"s": 0.0, "m": 0.0, "t": 0.0},
    )
    scenario_detour = Scenario(
        name="detour",
        travel_time={direct: 9.0, via_left: 1.0, via_right: 1.0},
        discomfort={direct: 1.0, via_left: 1.0, via_right: 1.0},
        hazard={direct: 0.0, via_left: 0.0, via_right: 0.0},
        cost={direct: 5.0, via_left: 1.0, via_right: 1.0},
        emissions={direct: 0.0, via_left: 0.0, via_right: 0.0},
        policing={"s": 0.0, "m": 0.0, "t": 0.0},
    )
    prior = FinitePrior(
        name="prior",
        support={"direct": scenario_direct, "detour": scenario_detour},
        probabilities={"direct": 0.8, "detour": 0.2},
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=total_order_from_list([MetricName.COST]),
        signal_policy=MaskSignalPolicy(
            considered_metrics=frozenset({MetricName.TRAVEL_TIME}),
            probabilities={MetricName.TRAVEL_TIME: 1.0},
        ),
    )
    receiver = receivers_module.RouteChoiceReceiver(
        individual=next(iter(world.individuals)),
        rtype="test",
        preference=total_order_from_list([MetricName.TRAVEL_TIME]),
        prior=prior,
        world=world,
        sender=sender,
        n_scenarios=2,
    )
    call_count = 0
    cached_solution = _solution_for_points(
        (
            _point("A", (direct,), travel_time=0.0, cost=0.0),
            _point("B", (via_left, via_right), travel_time=0.0, cost=0.0),
        )
    )

    def fake_solve_routes(**_: Any) -> RoutingSolution:
        nonlocal call_count
        call_count += 1
        return cached_solution

    monkeypatch.setattr(receivers_module, "solve_routes", fake_solve_routes)

    first_choice = receiver.get_path_choice()
    receiver.update_internal_belief(
        Signal(
            metrics=frozenset({MetricName.TRAVEL_TIME}),
            value={MetricName.TRAVEL_TIME: {direct: 9.0}},
        )
    )
    second_choice = receiver.get_path_choice()

    assert first_choice.label == "A"
    assert second_choice.label == "B"
    assert call_count == 1
