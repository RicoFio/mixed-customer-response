from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from mcr.avinfra_persuasion.datastructures import Demand
from mcr.avinfra_persuasion.experiments.experiment_4_0 import (
    build_lottery_policy_game_four,
    learn_lottery_policy_game_four,
    plot_lottery_signal_policy,
    receiver_type,
)


def test_experiment_4_0_lottery_policy_uses_mode_and_od_rtypes() -> None:
    game = build_lottery_policy_game_four(seed=3, n_humans=2, n_avs=2)

    expected_types = {
        "human_0_0_1_1",
        "av_0_0_1_1",
    }

    assert {receiver.rtype for receiver in game.receivers} == expected_types
    assert game.sender.signal_policy.type_names == expected_types


def test_experiment_4_0_lottery_policy_evaluation_uses_count_profiles() -> None:
    game = build_lottery_policy_game_four(seed=3, n_humans=2, n_avs=2)

    evaluation = game.evaluate_policy()

    assert len(evaluation["breakdown_rows"]) == 400
    assert "mask_counts_by_type" in evaluation["breakdown_rows"][0]
    assert evaluation["expected_sender_metric"] == pytest.approx(
        evaluation["expected_sender_utility"]
    )


def test_experiment_4_0_learns_lottery_policy() -> None:
    game, result = learn_lottery_policy_game_four(
        seed=3,
        n_humans=1,
        n_avs=1,
        max_iter=1,
    )

    assert result["iterations"] == 1
    assert "initial_expected_sender_metric" in result
    assert set(result["final_probabilities"]) == set(game.public_prior.support)
    for state_name, type_distributions in result["final_probabilities"].items():
        assert set(type_distributions) == set(game.sender.signal_policy.type_names)
        for type_name, distribution in type_distributions.items():
            assert sum(distribution.values()) == pytest.approx(1.0)
            stored_distribution = (
                game.sender.signal_policy.distribution_for_state_type(
                    state_name,
                    type_name,
                )
            )
            for mask, probability in distribution.items():
                assert probability == pytest.approx(stored_distribution[mask])


def test_experiment_4_0_plots_policy_and_gradients() -> None:
    _, result = learn_lottery_policy_game_four(
        seed=3,
        n_humans=1,
        n_avs=1,
        max_iter=1,
    )
    _, axes = plt.subplots(1, 2)

    policy_ax = plot_lottery_signal_policy(result, ax=axes[0])

    assert policy_ax is axes[0]
    plt.close(axes[0].figure)


def test_receiver_type_includes_mode_and_demand_nodes() -> None:
    demand = Demand(origin=(0, 0), destination=(1, 1))

    assert receiver_type("human", demand) == "human_0_0_1_1"
    assert receiver_type("av", demand) == "av_0_0_1_1"
