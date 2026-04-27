from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from mcr.avinfra_persuasion.datastructures import MetricName
from mcr.avinfra_persuasion.experiments.experiment_2_1 import (
    build_informative_game_two,
)
from mcr.avinfra_persuasion.experiments.plotting import (
    plot_state_policy_gradient_fields,
)


def test_experiment_2_1_evaluate_policy_runs() -> None:
    game = build_informative_game_two(seed=3, n_humans=1, n_avs=1)

    evaluation = game.evaluate_policy()

    assert len(evaluation["breakdown_rows"]) == 16


def test_experiment_2_1_solve_returns_valid_state_distributions() -> None:
    game = build_informative_game_two(seed=3, n_humans=1, n_avs=1)

    result = game.solve(max_iter=2)

    assert set(result["final_probabilities"]) == set(game.public_prior.support)
    for state_name, distribution in result["final_probabilities"].items():
        assert sum(distribution.values()) == pytest.approx(1.0)
        stored_distribution = game.sender.signal_policy.distribution_for_state(state_name)
        for mask, probability in distribution.items():
            assert probability == pytest.approx(stored_distribution[mask])


def test_experiment_2_1_plot_state_policy_gradient_fields_returns_axes() -> None:
    game = build_informative_game_two(seed=3, n_humans=1, n_avs=1)
    result = game.solve(max_iter=1)

    axes = plot_state_policy_gradient_fields(
        MetricName.HAZARD,
        MetricName.TRAVEL_TIME,
        game,
        result=result,
        grid_size=2,
        show_colorbar=False,
    )

    assert axes.size >= len(game.public_prior.support)
    plt.close(axes.reshape(-1)[0].figure)
