"""
Finite-prior toy experiment with a family of receivers and a per-metric policy.

The prior support is an epistemic model of the world used for signaling and
belief updates. Once all receivers choose routes, the actual realized network
metrics are computed endogenously from the joint path profile via
``world.get_realized_metrics(...)``.

sender has full commitment power
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from ..bp.game import ConvergenceGame, Preference
from ..bp.receivers import PriorRouteChoiceReceiver, Receiver
from ..bp.senders import ScalarSender, Sender
from ..bp.signals import Signal, StateDependentMaskSignalPolicy
from ..datastructures import (
    Arc,
    Demand,
    FinitePrior,
    Individual,
    MetricName,
    Node,
    Prior,
    Scenario,
    World,
)
from ..networks.toy_3 import create_sample_graph
from .helpers import format_mask, sorted_metrics
from .plotting import plot_state_mask_policy
from tqdm import tqdm


def build_informative_prior(world: World) -> FinitePrior:
    source_right = ((0, 0), (0, 1))
    right_target = ((0, 1), (1, 1))
    source_to_bottom = ((0, 0), (1, 0))
    bottom_to_target = ((1, 0), (1, 1))

    base_scenario = Scenario.from_world("base", world)
    def make_scenario(
        name: str,
        *,
        top_travel_time: float,
        top_hazard: float,
        node_policing: int,
    ) -> Scenario:
        return base_scenario.with_overrides(
            name=name,
            arc_overrides={
                MetricName.TRAVEL_TIME: {
                    source_right: top_travel_time,
                    right_target: top_travel_time,
                    source_to_bottom: 1.0,
                    bottom_to_target: 1.0,
                },
                MetricName.HAZARD: {
                    source_right: top_hazard,
                    right_target: top_hazard,
                    source_to_bottom: 0.5,
                    bottom_to_target: 0.5,
                },
                MetricName.COST: {
                    source_right: 1.0,
                    right_target: 1.0,
                    source_to_bottom: 0.0,
                    bottom_to_target: 0.0,
                },
                MetricName.EMISSIONS: {
                    source_right: 1.0,
                    right_target: 1.0,
                    source_to_bottom: 0.5,
                    bottom_to_target: 0.5,
                },
            },
            node_overrides={
                MetricName.POLICING: {
                    (0, 0): node_policing,
                    (1, 0): 0.5,
                    (1, 1): node_policing,
                }
            },
        )

    support = {
        "fast_safe": make_scenario(
            "fast_safe",
            top_travel_time=0.25,
            top_hazard=0.0,
            node_policing=1,
        ),
        "fast_risky": make_scenario(
            "fast_risky",
            top_travel_time=0.25,
            top_hazard=2.0,
            node_policing=0,
        ),
        "slow_safe": make_scenario(
            "slow_safe",
            top_travel_time=1.5,
            top_hazard=0.0,
            node_policing=1,
        ),
        "slow_risky": make_scenario(
            "slow_risky",
            top_travel_time=1.5,
            top_hazard=2.0,
            node_policing=0,
        ),
    }

    return FinitePrior(
        name="prior",
        support=support,
        probabilities={
            "fast_safe": 0.4,
            "fast_risky": 0.2,
            "slow_safe": 0.3,
            "slow_risky": 0.1,
        },
    )


@dataclass
class GameTwo(ConvergenceGame):
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: Prior
    seed: int

    _metric_order: tuple[MetricName, ...] = field(init=False, repr=False)
    _state_order: tuple[str, ...] = field(init=False, repr=False)
    _all_masks: tuple[frozenset[MetricName], ...] = field(init=False, repr=False)
    _logits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.receivers:
            raise ValueError("GameTwo requires at least one receiver.")
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
        receiver: Receiver,
        signal: Signal,
    ) -> Receiver:
        receiver.reset_for_evaluation()
        receiver.update_internal_belief(signal)
        return receiver

    def _evaluate_signal(
        self,
        signal: Signal,
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receivers: list[Receiver] = []
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
        receiver_metrics = {
            receiver.id: receiver.compute_realized_metrics(realized_scenario)
            for receiver in updated_receivers
        }
        sender_metric_value = sum(
            metrics[sender_metric] for metrics in receiver_metrics.values()
        )

        return {
            "realized_scenario": realized_scenario,
            "path_choices": path_choices,
            "receiver_metrics": receiver_metrics,
            "sender_metric_value": sender_metric_value,
            "path_counts": Counter(
                choice.path for choice in path_choices.values()
            ),
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

        expected_sender_metric = 0.0
        breakdown_rows: list[dict[str, Any]] = []

        try:
            for scenario_name, scenario in self.public_prior.support.items():
                scenario_probability = self.public_prior.probabilities[scenario_name]
                for mask in self._all_masks:
                    mask_probability = self._mask_probability(scenario_name, mask, probabilities)
                    if np.isclose(mask_probability, 0.0):
                        continue

                    signal = self.sender.materialize_signal(
                        mask=mask,
                        realized_scenario=scenario,
                    )
                    evaluation = self._evaluate_signal(signal, scenario)
                    weighted_contribution = (
                        scenario_probability
                        * mask_probability
                        * evaluation["sender_metric_value"]
                    )
                    expected_sender_metric += weighted_contribution
                    breakdown_rows.append(
                        {
                            "scenario_name": scenario_name,
                            "scenario_probability": scenario_probability,
                            "mask": sorted_metrics(mask),
                            "mask_probability": mask_probability,
                            "sender_metric_value": evaluation["sender_metric_value"],
                            "weighted_contribution": weighted_contribution,
                            "path_counts": dict(evaluation["path_counts"]),
                            "realized_scenario_name": evaluation["realized_scenario"].name,
                        }
                    )
        finally:
            self.sender.signal_policy.update_state_distributions(previous_probabilities)

        return {
            "expected_sender_utility": expected_sender_metric,
            "expected_sender_metric": expected_sender_metric,
            "breakdown_rows": tuple(breakdown_rows),
        }

    def _objective_from_flat_logits(self, flat_logits: np.ndarray) -> float:
        probabilities = self.signaling_scheme(logits=flat_logits)
        return float(self.evaluate_policy(probabilities)["expected_sender_utility"])

    def _finite_difference_gradient(
        self,
        flat_logits: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        gradient = np.zeros_like(flat_logits)
        for idx in range(flat_logits.size):
            direction = np.zeros_like(flat_logits)
            direction[idx] = epsilon
            plus = self._objective_from_flat_logits(flat_logits + direction)
            minus = self._objective_from_flat_logits(flat_logits - direction)
            gradient[idx] = (plus - minus) / (2.0 * epsilon)
        return gradient

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
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if finite_diff_epsilon <= 0:
            raise ValueError("finite_diff_epsilon must be positive.")
        if convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive.")

        flat_logits = self._logits.reshape(-1).copy()
        m = np.zeros_like(flat_logits)
        v = np.zeros_like(flat_logits)
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-4

        utility_history: list[float] = []
        grad_norm_history: list[float] = []
        policy_history: list[dict[str, dict[frozenset[MetricName], float]]] = []
        stagnant_steps = 0
        converged = False

        for step in range(1, max_iter + 1):
            policy_history.append(self.signaling_scheme(logits=flat_logits))
            utility = self._objective_from_flat_logits(flat_logits)
            gradient = self._finite_difference_gradient(
                flat_logits=flat_logits,
                epsilon=finite_diff_epsilon,
            )
            grad_norm = float(np.linalg.norm(gradient))

            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            flat_logits = flat_logits + step_size * m_hat / (np.sqrt(v_hat) + adam_eps)

            utility_history.append(float(utility))
            grad_norm_history.append(grad_norm)

            if len(utility_history) >= 2:
                utility_delta = abs(utility_history[-1] - utility_history[-2])
                if utility_delta < convergence_tol:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0

            if stagnant_steps >= convergence_patience:
                converged = True
                break

        self._logits = flat_logits.reshape(len(self._state_order), len(self._all_masks))
        final_probabilities = self.signaling_scheme()
        self.sender.signal_policy.update_state_distributions(final_probabilities)
        evaluation = self.evaluate_policy(final_probabilities)
        return {
            "iterations": len(utility_history),
            "converged": converged,
            "utility_history": utility_history,
            "grad_norm_history": grad_norm_history,
            "policy_history": policy_history + [final_probabilities],
            "final_probabilities": final_probabilities,
            **evaluation,
        }


def build_informative_game_two(
    seed: int = 1,
    n_humans: int = 10,
    n_avs: int = 10,
) -> GameTwo:
    network = create_sample_graph()
    origin: Node = (0, 0)
    target: Node = (1, 1)
    individuals = frozenset(
        [Individual(id=f"h{i}", demand=Demand(origin, target)) for i in range(n_humans)]
        + [Individual(id=f"a{i}", demand=Demand(origin, target)) for i in range(n_avs)]
    )
    world = World(network=network, individuals=individuals)
    prior = build_informative_prior(world)

    human_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.COST,
            MetricName.EMISSIONS,
        },
        relations={
            (MetricName.COST, MetricName.TRAVEL_TIME),
            (MetricName.EMISSIONS, MetricName.TRAVEL_TIME),
        },
    )
    av_preference = Preference(
        elements={
            MetricName.HAZARD,
            MetricName.COST,
        },
        relations={
            (MetricName.COST, MetricName.HAZARD),
        },
    )
    # av_preference.draw_hasse_diagram()
    # plt.show()
    sender_preference = Preference(
        elements={MetricName.TRAVEL_TIME},
        relations=set(),
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        signal_policy=StateDependentMaskSignalPolicy(
            seed=seed,
            state_names=frozenset(prior.support),
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    MetricName.HAZARD,
                }
            ),
        ),
    )
    receivers = [
        PriorRouteChoiceReceiver(
            individual=individual,
            rtype="human" if individual.id.startswith("h") else "av",
            preference=human_preference if individual.id.startswith("h") else av_preference,
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=len(prior.support),
        )
        for individual in sorted(individuals, key=lambda individual: individual.id)
    ]
    return GameTwo(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=seed,
    )


if __name__ == "__main__":
    game = build_informative_game_two(seed=1, n_humans=5, n_avs=5)
    result = game.solve(max_iter=100)

    print("Converged:", result["converged"])
    print("Iterations:", result["iterations"])
    print("Final state-dependent mask distributions:")
    for state_name, distribution in result["final_probabilities"].items():
        print(f"  {state_name}:")
        for mask in sorted(
            distribution,
            key=lambda mask: (
                len(mask),
                tuple(
                    metric.value
                    for metric in sorted(mask, key=lambda metric: metric.value)
                ),
            ),
        ):
            print(f"    {format_mask(mask)}: {distribution[mask]:.4f}")

    print("Per-scenario / per-mask breakdown:")
    for row in result["breakdown_rows"]:
        print(
            f"  scenario={row['scenario_name']} "
            f"(p={row['scenario_probability']:.3f}) | "
            f"mask={format_mask(row['mask'])} "
            f"(p={row['mask_probability']:.3f}) | "
            f"path_counts={row['path_counts']} | "
            f"sender_metric={row['sender_metric_value']:.3f} | "
            f"contribution={row['weighted_contribution']:.3f}"
        )

    print("Final expected sender metric:", f"{result['expected_sender_metric']:.4f}")

    plot_state_mask_policy(result)
    plt.show()
