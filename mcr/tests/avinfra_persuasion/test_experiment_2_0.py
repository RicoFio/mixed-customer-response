from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from mcr.avinfra_persuasion.datastructures import MetricName
from mcr.avinfra_persuasion.experiments.experiment_2_0 import (
    build_informative_game_two,
)
from mcr.avinfra_persuasion.experiments.plotting import plot_policy_gradient_field


def test_experiment_2_0_solve_updates_sender_policy() -> None:
    game = build_informative_game_two(seed=3, n_humans=1, n_avs=1)

    result = game.solve(max_iter=1)

    for metric, probability in result["final_probabilities"].items():
        assert probability == pytest.approx(game.sender.signal_policy.probability(metric))


def test_experiment_2_0_plot_policy_gradient_field_returns_axis() -> None:
    game = build_informative_game_two(seed=3, n_humans=1, n_avs=1)
    result = game.solve(max_iter=2, convergence_patience=10)
    _, ax = plt.subplots()

    returned = plot_policy_gradient_field(
        MetricName.HAZARD,
        MetricName.TRAVEL_TIME,
        game,
        result=result,
        ax=ax,
        grid_size=5,
        show_colorbar=False,
    )

    assert returned is ax
    plt.close(ax.figure)
