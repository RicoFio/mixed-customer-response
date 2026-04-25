from __future__ import annotations

import numpy as np
import pytest

from mcr.avinfra_persuasion.datastructures import Arc, MetricName
from mcr.avinfra_persuasion.experiments.experiment_1 import build_informative_game_one


def _top_path() -> tuple[Arc, ...]:
    return (
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
    )


def _instrumented_path() -> tuple[Arc, ...]:
    return (
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
    )


def test_bayes_plausibility_report_reconstructs_prior() -> None:
    game = build_informative_game_one(seed=3)

    report = game.bayes_plausibility_report()

    assert report["max_error"] < 1e-10
    assert sum(row["signal_probability"] for row in report["rows"]) == pytest.approx(1.0)
    assert report["reconstructed_prior"]["instrumented_good"] == pytest.approx(0.5)
    assert report["reconstructed_prior"]["instrumented_bad"] == pytest.approx(0.5)


def test_hidden_mask_keeps_prior_and_revealing_mask_updates_posterior() -> None:
    game = build_informative_game_one(seed=3)
    good_scenario = game.public_prior.support["instrumented_good"]

    empty_signal = game.sender.materialize_signal(
        mask=frozenset(),
        realized_scenario=good_scenario,
    )
    reveal_signal = game.sender.materialize_signal(
        mask=frozenset({MetricName.HAZARD}),
        realized_scenario=good_scenario,
    )

    empty_posterior = game.posterior_from_signal(empty_signal)
    reveal_posterior = game.posterior_from_signal(reveal_signal)

    assert empty_signal.value == {}
    assert empty_posterior.probabilities == game.public_prior.probabilities
    assert reveal_signal.value[MetricName.HAZARD][((0, 0), (1, 0))] == 0.0
    assert reveal_posterior.probabilities == {
        "instrumented_good": 1.0,
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
        mask=frozenset({MetricName.POLICING}),
        realized_scenario=good_scenario,
    )
    reveal_signal_bad = game.sender.materialize_signal(
        mask=frozenset({MetricName.POLICING}),
        realized_scenario=bad_scenario,
    )

    first_choice = game.choose_route(empty_signal)
    second_choice = game.choose_route(empty_signal)
    good_revealed_choice = game.choose_route(reveal_signal_good)
    bad_revealed_choice = game.choose_route(reveal_signal_bad)

    assert first_choice.path == second_choice.path == _top_path()
    assert good_revealed_choice.path == _instrumented_path()
    assert bad_revealed_choice.path == _top_path()


def test_evaluate_policy_returns_breakdown_and_nonzero_gradient() -> None:
    game = build_informative_game_one(seed=3)

    evaluation = game.evaluate_policy()
    gradient = game._finite_difference_gradient(game._logits.copy(), 1e-4)

    assert len(evaluation["breakdown_rows"]) == 16
    assert any(
        row["mask"] == (MetricName.HAZARD,) and row["signal_summary"] != "<empty>"
        for row in evaluation["breakdown_rows"]
    )
    assert np.linalg.norm(gradient) > 1e-6
