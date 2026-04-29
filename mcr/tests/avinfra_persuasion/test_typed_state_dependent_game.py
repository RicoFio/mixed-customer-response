from __future__ import annotations

import pytest

from mcr.avinfra_persuasion.bp.receivers import Receiver
from mcr.avinfra_persuasion.bp.senders import ScalarSender
from mcr.avinfra_persuasion.bp.signals import TypedStateDependentMaskSignalPolicy
from mcr.avinfra_persuasion.datastructures import (
    Demand,
    FinitePrior,
    Individual,
    InfrastructureGraph,
    MetricName,
    Scenario,
    World,
)
from mcr.avinfra_persuasion.experiments.games.osmrspts import OSMRSPTSGame
from mcr.avinfra_persuasion.experiments.games.osmrsptslp import OSMRSPTSLPGame
from mcr.avinfra_persuasion.orders import total_order_from_list


def test_typed_state_dependent_game_weights_independent_type_mask_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arc = ("s", "t")
    human = Individual(id="human", demand=Demand(origin="s", destination="t"))
    av = Individual(id="av", demand=Demand(origin="s", destination="t"))
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
    world = World(network=network, individuals=frozenset({human, av}))
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
        signal_policy=TypedStateDependentMaskSignalPolicy(
            type_names=frozenset({"human", "av"}),
            state_names=frozenset(prior.support),
            considered_metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
        ),
    )
    receivers = [
        Receiver(
            individual=human,
            rtype="human",
            preference=total_order_from_list([MetricName.TRAVEL_TIME]),
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=1,
        ),
        Receiver(
            individual=av,
            rtype="av",
            preference=total_order_from_list([MetricName.TRAVEL_TIME]),
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=1,
        ),
    ]
    game = OSMRSPTSGame(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=1,
    )
    empty_mask = frozenset()
    travel_time_mask = frozenset({MetricName.TRAVEL_TIME})
    hazard_mask = frozenset({MetricName.HAZARD})
    both_mask = frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD})
    probabilities = {
        "rho": {
            "human": {
                empty_mask: 0.5,
                travel_time_mask: 0.5,
                hazard_mask: 0.0,
                both_mask: 0.0,
            },
            "av": {
                empty_mask: 0.8,
                travel_time_mask: 0.0,
                hazard_mask: 0.2,
                both_mask: 0.0,
            },
        },
    }
    recorded_profiles: list[dict[str, frozenset[MetricName]]] = []

    def fake_evaluate_typed_signals(signals_by_type, believed_scenario):
        profile = {
            type_name: signal.metrics
            for type_name, signal in signals_by_type.items()
        }
        recorded_profiles.append(profile)
        sender_metric_value = (
            10.0
            if profile == {"human": travel_time_mask, "av": hazard_mask}
            else 0.0
        )
        return {
            "realized_scenario": believed_scenario,
            "path_counts": {},
            "sender_metric_value": sender_metric_value,
        }

    monkeypatch.setattr(game, "_evaluate_typed_signals", fake_evaluate_typed_signals)

    evaluation = game.evaluate_policy(probabilities)

    assert evaluation["expected_sender_metric"] == pytest.approx(1.0)
    assert {"human": travel_time_mask, "av": hazard_mask} in recorded_profiles


def test_typed_state_dependent_lottery_game_compresses_same_type_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arc = ("s", "t")
    human_0 = Individual(id="human_0", demand=Demand(origin="s", destination="t"))
    human_1 = Individual(id="human_1", demand=Demand(origin="s", destination="t"))
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
    world = World(network=network, individuals=frozenset({human_0, human_1}))
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
        signal_policy=TypedStateDependentMaskSignalPolicy(
            type_names=frozenset({"human"}),
            state_names=frozenset(prior.support),
            considered_metrics=frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD}),
        ),
    )
    receivers = [
        Receiver(
            individual=human_0,
            rtype="human",
            preference=total_order_from_list([MetricName.TRAVEL_TIME]),
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=1,
        ),
        Receiver(
            individual=human_1,
            rtype="human",
            preference=total_order_from_list([MetricName.TRAVEL_TIME]),
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=1,
        ),
    ]
    game = OSMRSPTSLPGame(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=1,
    )
    empty_mask = frozenset()
    travel_time_mask = frozenset({MetricName.TRAVEL_TIME})
    hazard_mask = frozenset({MetricName.HAZARD})
    both_mask = frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD})
    probabilities = {
        "rho": {
            "human": {
                empty_mask: 0.5,
                travel_time_mask: 0.5,
                hazard_mask: 0.0,
                both_mask: 0.0,
            },
        },
    }
    response_calls: list[frozenset[MetricName]] = []
    original_receiver_after_signal = game._receiver_after_signal

    def counting_receiver_after_signal(receiver, signal):
        response_calls.append(signal.metrics)
        return original_receiver_after_signal(receiver, signal)

    monkeypatch.setattr(game, "_receiver_after_signal", counting_receiver_after_signal)

    evaluation = game.evaluate_policy(probabilities)

    assert len(evaluation["breakdown_rows"]) == 3
    assert response_calls == [travel_time_mask, empty_mask]
    assert {
        "human": {
            tuple(): 1,
            (MetricName.TRAVEL_TIME,): 1,
        }
    } in [
        row["mask_counts_by_type"]
        for row in evaluation["breakdown_rows"]
    ]
