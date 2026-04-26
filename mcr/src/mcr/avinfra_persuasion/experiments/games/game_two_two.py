from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from ...bp.game import ConvergenceGame
from ...bp.receivers import ExperiencedRouteChoiceReceiver, Receiver
from ...bp.senders import Sender
from ...bp.signals import Signal, StateDependentMaskSignalPolicy
from ...datastructures import (
    FinitePrior,
    MetricName,
    Prior,
    Scenario,
    World,
)
from ..helpers import sorted_metrics


@dataclass
class GameTwo(ConvergenceGame):
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: Prior
    seed: int
    horizon: int = 20

    _metric_order: tuple[MetricName, ...] = field(init=False, repr=False)
    _state_order: tuple[str, ...] = field(init=False, repr=False)
    _all_masks: tuple[frozenset[MetricName], ...] = field(init=False, repr=False)
    _logits: np.ndarray = field(init=False, repr=False)
    _rollout_scenarios: tuple[Scenario, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.receivers:
            raise ValueError("GameTwo requires at least one receiver.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not isinstance(self.public_prior, FinitePrior):
            raise NotImplementedError("GameTwo currently requires a FinitePrior.")
        if not isinstance(self.sender.signal_policy, StateDependentMaskSignalPolicy):
            raise NotImplementedError(
                "GameTwo currently requires a StateDependentMaskSignalPolicy sender."
            )
        if self.sender.prior != self.public_prior:
            raise ValueError("Sender prior must match the public prior.")
        if any(receiver.prior != self.public_prior for receiver in self.receivers):
            raise ValueError("Receiver priors must match the public prior.")
        if self.sender.world != self.world:
            raise ValueError("Sender world must match the game world.")
        if any(receiver.world != self.world for receiver in self.receivers):
            raise ValueError("Receiver worlds must match the game world.")
        if any(
            not isinstance(receiver, ExperiencedRouteChoiceReceiver)
            for receiver in self.receivers
        ):
            raise ValueError(
                "GameTwo currently requires ExperiencedRouteChoiceReceiver instances."
            )

        self._metric_order = sorted_metrics(
            self.sender.signal_policy.considered_metrics
        )
        self._state_order = tuple(sorted(self.public_prior.support))
        if self.sender.signal_policy.state_names != frozenset(self._state_order):
            raise ValueError(
                "State-dependent signal policy states must match the finite prior support."
            )
        self._all_masks = tuple(
            frozenset(
                metric
                for idx, metric in enumerate(self._metric_order)
                if bitmask & (1 << idx)
            )
            for bitmask in range(1 << len(self._metric_order))
        )

        probabilities = np.array(
            [
                [
                    self.sender.signal_policy.mask_probability(state_name, mask)
                    for mask in self._all_masks
                ]
                for state_name in self._state_order
            ],
            dtype=float,
        )
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        self._logits = np.log(clipped)
        self._rollout_scenarios = tuple(
            self.public_prior.sample(self.horizon, seed=self.seed)
        )

    @staticmethod
    def _softmax_rows(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def signaling_scheme(
        self,
        logits: np.ndarray | None = None,
    ) -> dict[str, dict[frozenset[MetricName], float]]:
        if logits is None:
            logits_array = self._logits
        else:
            logits_array = np.asarray(logits, dtype=float)
            if logits_array.ndim == 1:
                logits_array = logits_array.reshape(
                    len(self._state_order),
                    len(self._all_masks),
                )
        probabilities = self._softmax_rows(logits_array)
        return {
            state_name: {
                mask: float(probability)
                for mask, probability in zip(self._all_masks, state_probabilities)
            }
            for state_name, state_probabilities in zip(self._state_order, probabilities)
        }

    def _mask_probability(
        self,
        state_name: str,
        mask: frozenset[MetricName],
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]],
    ) -> float:
        return probabilities[state_name][mask]

    def _sender_metric(self) -> MetricName:
        if len(self.sender.preference.elements) != 1:
            raise ValueError("Sender preference must be degenerate.")
        return next(iter(self.sender.preference.elements))

    def _receiver_after_signal(
        self,
        receiver: ExperiencedRouteChoiceReceiver,
        signal: Signal,
    ) -> ExperiencedRouteChoiceReceiver:
        receiver.reset_public_belief()
        receiver.update_internal_belief(signal)
        return receiver

    def _evaluate_round(
        self,
        round_index: int,
        signal: Signal,
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receivers: list[ExperiencedRouteChoiceReceiver] = []
        path_choices = {}

        for receiver in self.receivers:
            updated_receiver = self._receiver_after_signal(receiver, signal)
            updated_receivers.append(updated_receiver)
            path_choices[updated_receiver.individual] = updated_receiver.get_path_choice()

        realized_scenario = self.world.get_realized_metrics(
            path_choices=path_choices,
            name=f"realized_{believed_scenario.name}",
            base_scenario=believed_scenario,
        )
        sender_metric = self._sender_metric()
        receiver_metrics: dict[str, Mapping[MetricName, float]] = {}
        for receiver in updated_receivers:
            realized_metrics = receiver.compute_realized_metrics(realized_scenario)
            receiver.update_private_route_belief(realized_metrics)
            receiver_metrics[receiver.id] = realized_metrics
        sender_metric_value = sum(
            metrics[sender_metric] for metrics in receiver_metrics.values()
        )
        updated_paths = tuple(
            sorted(
                {choice.path for choice in path_choices.values()},
                key=str,
            )
        )

        return {
            "round": round_index,
            "scenario_name": believed_scenario.name,
            "mask": sorted_metrics(signal.metrics),
            "realized_scenario": realized_scenario,
            "path_choices": path_choices,
            "receiver_metrics": receiver_metrics,
            "sender_metric_value": sender_metric_value,
            "path_counts": Counter(
                choice.path for choice in path_choices.values()
            ),
            "updated_paths": updated_paths,
        }

    def evaluate_policy(
        self,
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        previous_probabilities = {
            state_name: dict(distribution)
            for state_name, distribution in self.sender.signal_policy.state_probabilities.items()
        }
        self.sender.signal_policy.update_state_distributions(probabilities)

        total_sender_metric = 0.0
        breakdown_rows: list[dict[str, Any]] = []
        rollout_rng = np.random.default_rng(self.seed + 1)

        try:
            for receiver in self.receivers:
                receiver.reset_for_rollout()
            for round_index, scenario in enumerate(self._rollout_scenarios, start=1):
                signal = self.sender.emit_signal(
                    realized_scenario=scenario,
                    rng=rollout_rng,
                )
                evaluation = self._evaluate_round(
                    round_index=round_index,
                    signal=signal,
                    believed_scenario=scenario,
                )
                total_sender_metric += evaluation["sender_metric_value"]
                breakdown_rows.append(
                    {
                        "round": evaluation["round"],
                        "scenario_name": evaluation["scenario_name"],
                        "mask": evaluation["mask"],
                        "path_counts": dict(evaluation["path_counts"]),
                        "sender_metric_value": evaluation["sender_metric_value"],
                        "updated_paths": evaluation["updated_paths"],
                    }
                )
        finally:
            self.sender.signal_policy.update_state_distributions(previous_probabilities)

        average_sender_metric = total_sender_metric / self.horizon
        return {
            "expected_sender_utility": average_sender_metric,
            "average_sender_metric": average_sender_metric,
            "breakdown_rows": tuple(breakdown_rows),
            "final_private_route_beliefs": {
                receiver.id: receiver.private_route_beliefs
                for receiver in self.receivers
            },
        }

    def _deterministic_scheme(
        self,
        mask_indices: tuple[int, ...],
    ) -> dict[str, dict[frozenset[MetricName], float]]:
        return {
            state_name: {
                mask: 1.0 if mask_idx == chosen_mask_idx else 0.0
                for mask_idx, mask in enumerate(self._all_masks)
            }
            for state_name, chosen_mask_idx in zip(self._state_order, mask_indices)
        }

    def solve(
        self,
        max_iter: int = 100,
        step_size: float = 0.15,
        finite_diff_epsilon: float = 1e-4,
        convergence_tol: float = 1e-8,
        convergence_patience: int = 15,
    ) -> dict[str, Any]:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        utility_history: list[float] = []
        policy_history: list[dict[str, dict[frozenset[MetricName], float]]] = []
        best_policy: dict[str, dict[frozenset[MetricName], float]] | None = None
        best_evaluation: dict[str, Any] | None = None
        best_utility = float("-inf")

        for mask_indices in product(
            range(len(self._all_masks)),
            repeat=len(self._state_order),
        ):
            policy = self._deterministic_scheme(mask_indices)
            evaluation = self.evaluate_policy(policy)
            utility = float(evaluation["expected_sender_utility"])
            utility_history.append(utility)
            policy_history.append(policy)
            if utility > best_utility:
                best_utility = utility
                best_policy = policy
                best_evaluation = evaluation

        if best_policy is None or best_evaluation is None:
            raise RuntimeError("Deterministic policy enumeration produced no candidate.")

        final_probabilities = best_policy
        self.sender.signal_policy.update_state_distributions(final_probabilities)
        return {
            "iterations": len(utility_history),
            "converged": True,
            "search_mode": "deterministic_enumeration",
            "utility_history": utility_history,
            "grad_norm_history": [],
            "policy_history": policy_history + [final_probabilities],
            "final_probabilities": final_probabilities,
            **best_evaluation,
        }
