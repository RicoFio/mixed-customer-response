from __future__ import annotations

import pytest

from mcr.avinfra_persuasion.experiments.games.osmrspmr import OSMRThree
from mcr.avinfra_persuasion.experiments.experiment_2_2 import (
    build_informative_game_two,
)


def test_experiment_2_2_builder_returns_game_two_two() -> None:
    game = build_informative_game_two(seed=2, n_humans=1, n_avs=1, horizon=4)

    assert isinstance(game, OSMRThree)


def test_experiment_2_2_evaluate_policy_is_deterministic_for_fixed_seed() -> None:
    game = build_informative_game_two(seed=2, n_humans=1, n_avs=1, horizon=4)

    evaluation_a = game.evaluate_policy()
    evaluation_b = game.evaluate_policy()

    assert evaluation_a["average_sender_metric"] == pytest.approx(
        evaluation_b["average_sender_metric"]
    )
    assert evaluation_a["breakdown_rows"] == evaluation_b["breakdown_rows"]


def test_experiment_2_2_solve_returns_rollout_breakdown_and_private_beliefs() -> None:
    game = build_informative_game_two(seed=2, n_humans=1, n_avs=1, horizon=4)

    result = game.solve(max_iter=1)

    assert len(result["breakdown_rows"]) == 4
    assert isinstance(result["average_sender_metric"], float)
    assert set(result["final_private_route_beliefs"]) == {"a0", "h0"}
    for state_name, distribution in result["final_probabilities"].items():
        assert sum(distribution.values()) == pytest.approx(1.0)
        stored_distribution = game.sender.signal_policy.distribution_for_state(state_name)
        for mask, probability in distribution.items():
            assert probability == pytest.approx(stored_distribution[mask])
