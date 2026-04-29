from __future__ import annotations

import pytest

from mcr.avinfra_persuasion.bp.signals import StateDependentMaskSignalPolicy
from mcr.avinfra_persuasion.datastructures import MetricName
from mcr.avinfra_persuasion.experiments.experiment_0 import (
    ACCIDENT_SIGNAL,
    ACCIDENT_STATE,
    EMPTY_MASK,
    HAZARD_MASK,
    NO_ACCIDENT_SIGNAL,
    NO_ACCIDENT_STATE,
    GameZero,
    WuAminParameters,
    build_game_zero,
    solve_wu_amin_case,
)


def test_game_zero_builder_returns_hazard_only_state_dependent_game() -> None:
    game = build_game_zero(pop_lambda=0.2, seed=2)

    assert isinstance(game, GameZero)
    assert isinstance(game.sender.signal_policy, StateDependentMaskSignalPolicy)
    assert game.sender.signal_policy.considered_metrics == frozenset(
        {MetricName.HAZARD}
    )
    assert set(game.public_prior.support) == {ACCIDENT_STATE, NO_ACCIDENT_STATE}
    assert set(game.sender.signal_policy.state_names) == {
        ACCIDENT_STATE,
        NO_ACCIDENT_STATE,
    }
    for distribution in game.sender.signal_policy.state_probabilities.values():
        assert set(distribution) == {EMPTY_MASK, HAZARD_MASK}
        assert sum(distribution.values()) == pytest.approx(1.0)


def test_wu_amin_thresholds_match_paper_example() -> None:
    parameters = WuAminParameters()

    assert parameters.lambda_bottom == pytest.approx(0.13333333333333333)
    assert parameters.lambda_top == pytest.approx(0.25)


def test_reference_policy_evaluation_matches_reference_spillover() -> None:
    parameters = WuAminParameters(pop_lambda=0.2)
    game = build_game_zero(parameters=parameters)

    evaluation = game.evaluate_policy(parameters.reference_policy())

    assert evaluation["reference_regime"] == "L2"
    assert evaluation["pi_a_given_a"] == pytest.approx(
        parameters.pi_star(ACCIDENT_SIGNAL, ACCIDENT_SIGNAL)
    )
    assert evaluation["pi_a_given_n"] == pytest.approx(
        parameters.pi_star(ACCIDENT_SIGNAL, NO_ACCIDENT_SIGNAL)
    )
    assert evaluation["expected_spillover"] == pytest.approx(
        parameters.reference_spillover
    )


@pytest.mark.parametrize(
    ("pop_lambda", "regime", "expected_pi_a_given_a"),
    [
        (0.10, "L1", 1.0),
        (0.20, "L2", 2.0 / 3.0),
        (0.30, "L3", 8.0 / 15.0),
    ],
)
def test_finite_difference_solve_recovers_wu_amin_regimes(
    pop_lambda: float,
    regime: str,
    expected_pi_a_given_a: float,
) -> None:
    result = solve_wu_amin_case(
        pop_lambda=pop_lambda,
        seed=5,
        max_iter=500,
        step_size=0.08,
    )

    assert result["reference_regime"] == regime
    assert result["pi_a_given_a"] == pytest.approx(expected_pi_a_given_a, abs=0.06)
    assert result["pi_a_given_n"] == pytest.approx(0.0, abs=0.04)
    assert result["expected_spillover"] == pytest.approx(
        result["reference_spillover"],
        abs=0.03,
    )
