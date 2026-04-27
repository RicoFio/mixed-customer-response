from __future__ import annotations

from mcr.avinfra_persuasion.bp.signals import TypedStateDependentMaskSignalPolicy
from mcr.avinfra_persuasion.experiments.experiment_3_0 import (
    build_typed_state_dependent_game_three,
)


def test_experiment_3_0_typed_state_dependent_evaluate_policy_runs() -> None:
    game = build_typed_state_dependent_game_three(seed=3, n_humans=1, n_avs=1)

    evaluation = game.evaluate_policy()

    assert isinstance(game.sender.signal_policy, TypedStateDependentMaskSignalPolicy)
    assert game.sender.signal_policy.type_names == frozenset({"human", "av"})
    assert len(evaluation["breakdown_rows"]) == 64
    assert "masks_by_type" in evaluation["breakdown_rows"][0]
