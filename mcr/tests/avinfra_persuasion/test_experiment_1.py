from __future__ import annotations

import numpy as np
import pytest

from mcr.avinfra_persuasion.datastructures import Arc, MetricName
from mcr.avinfra_persuasion.experiments.experiment_1_0 import build_informative_game_one


def _top_path() -> tuple[Arc, ...]:
    return (
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
    )


def _bottom_path() -> tuple[Arc, ...]:
    return (
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
    )


def test_bayes_plausibility_report_reconstructs_prior() -> None:
    game = build_informative_game_one(seed=3)

    report = game.bayes_plausibility_report()

    assert report["max_error"] < 1e-10
    assert sum(row["signal_probability"] for row in report["rows"]) == pytest.approx(1.0)
    assert report["reconstructed_prior"] == {
        "instrumented_good": pytest.approx(0.5),
        "instrumented_bad": pytest.approx(0.5),
    }


def test_hidden_mask_keeps_prior_and_revealing_mask_updates_posterior() -> None:
    game = build_informative_game_one(seed=3)
    scenario = game.public_prior.support["instrumented_good"]

    empty_signal = game.sender.materialize_signal(
        mask=frozenset(),
        realized_scenario=scenario,
    )
    cost_signal = game.sender.materialize_signal(
        mask=frozenset({MetricName.COST}),
        realized_scenario=scenario,
    )
    travel_time_signal = game.sender.materialize_signal(
        mask=frozenset({MetricName.TRAVEL_TIME}),
        realized_scenario=scenario,
    )
    full_signal = game.sender.materialize_signal(
        mask=frozenset({MetricName.COST, MetricName.TRAVEL_TIME}),
        realized_scenario=scenario,
    )

    empty_posterior = game.posterior_from_signal(empty_signal)
    cost_posterior = game.posterior_from_signal(cost_signal)
    travel_time_posterior = game.posterior_from_signal(travel_time_signal)
    full_posterior = game.posterior_from_signal(full_signal)

    assert empty_signal.value == {}
    assert empty_posterior.probabilities == game.public_prior.probabilities
    assert set(cost_signal.value[MetricName.COST]) == game.world.I
    assert cost_signal.value[MetricName.COST][((0, 0), (1, 0))] == 0.0
    assert cost_signal.value[MetricName.COST][((1, 0), (1, 1))] == 0.4
    assert cost_posterior.probabilities == {
        "instrumented_good": pytest.approx(1.0),
    }
    assert travel_time_posterior.probabilities == {
        "instrumented_good": pytest.approx(1.0),
    }
    assert full_posterior.probabilities == {
        "instrumented_good": pytest.approx(1.0),
    }


def test_stateless_evaluator_repeatability_and_route_flip() -> None:
    game = build_informative_game_one(seed=3)
    good_scenario = game.public_prior.support["instrumented_good"]
    bad_scenario = game.public_prior.support["instrumented_bad"]

    empty_signal = game.sender.materialize_signal(
        mask=frozenset(),
        realized_scenario=good_scenario,
    )
    reveal_signal_good = game.sender.materialize_signal(
        mask=frozenset({MetricName.COST}),
        realized_scenario=good_scenario,
    )
    reveal_signal_bad = game.sender.materialize_signal(
        mask=frozenset({MetricName.COST}),
        realized_scenario=bad_scenario,
    )

    first_choice = game._evaluate_signal(empty_signal, good_scenario)["chosen_route"]
    second_choice = game._evaluate_signal(empty_signal, good_scenario)["chosen_route"]
    good_revealed_choice = game._evaluate_signal(
        reveal_signal_good,
        good_scenario,
    )["chosen_route"]
    bad_revealed_choice = game._evaluate_signal(
        reveal_signal_bad,
        bad_scenario,
    )["chosen_route"]

    assert first_choice.path == second_choice.path == _bottom_path()
    assert good_revealed_choice.path == _top_path()
    assert bad_revealed_choice.path == _bottom_path()


def test_evaluate_policy_returns_breakdown_and_nonzero_gradient() -> None:
    game = build_informative_game_one(seed=3)

    evaluation = game.evaluate_policy()
    gradient = game._finite_difference_gradient(game._logits.copy(), 1e-4)

    assert len(evaluation["breakdown_rows"]) == 8
    assert any(
        row["mask"] == (MetricName.COST,) and row["signal_summary"] != "<empty>"
        for row in evaluation["breakdown_rows"]
    )
    assert all(component > 0.0 for component in gradient)
    assert np.linalg.norm(gradient) > 1e-6


def test_solve_updates_sender_policy() -> None:
    game = build_informative_game_one(seed=3)

    result = game.solve(max_iter=1)

    for metric, probability in result["final_probabilities"].items():
        assert probability == pytest.approx(game.sender.signal_policy.probability(metric))
