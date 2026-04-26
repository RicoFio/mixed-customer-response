from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from ..datastructures import (
    Arc,
    Demand,
    FinitePrior,
    Individual,
    MetricName,
    Scenario,
    World,
    WorldBelief,
    Prior,
)
from ..orders import PreOrder
from ..opt import (
    RoutingSolution,
    RoutingSolutionPoint,
)
from ..routing.routing_solvers import (
    RoutingSolverConfig,
    RoutingSolverConfigLike,
    solve_routes,
)
from .game import AVERAGE_SAMPLING, N_SCENARIOS, RNG_SEED, Preference
from .senders import Sender
from .signals import Signal

ReceiverType: TypeAlias = str


@dataclass
class Receiver:
    individual: Individual
    rtype: ReceiverType
    preference: Preference
    prior: Prior
    world: World
    sender: Sender | None = None
    belief: WorldBelief | Prior | None = None
    n_scenarios: int = N_SCENARIOS
    average_sampling: bool = AVERAGE_SAMPLING
    rng_seed: int | None = RNG_SEED
    routing_solver_config: RoutingSolverConfigLike = field(
        default_factory=RoutingSolverConfig,
    )

    _rng: np.random.Generator = field(init=False, repr=False, compare=False)
    _action_history: list[RoutingSolutionPoint] = field(
        init=False,
        default_factory=list,
    )
    _realized_payoff_history: list[Mapping[MetricName, float]] = field(
        init=False,
        default_factory=list,
    )

    def __post_init__(self) -> None:
        self._validate_demand()
        if self.belief is None:
            self.belief = self.prior
        self._rng = np.random.default_rng(self.rng_seed)

    @property
    def id(self) -> str:
        return self.individual.id

    @property
    def demand(self) -> Demand:
        return self.individual.demand

    def update_internal_belief(self, signal: Signal) -> None:
        if not signal.value:
            return

        current_belief = self._current_belief()
        if not isinstance(current_belief, FinitePrior):
            raise NotImplementedError(
                "Exact Bayesian signal updates currently require a FinitePrior."
            )

        posterior_support: dict[str, Scenario] = {}
        posterior_probabilities: dict[str, float] = {}
        for scenario_name, scenario in current_belief.support.items():
            if self._scenario_matches_signal(scenario=scenario, signal=signal):
                posterior_support[scenario_name] = scenario
                posterior_probabilities[scenario_name] = (
                    current_belief.probabilities[scenario_name]
                )

        if not posterior_support:
            raise ValueError(
                "Signal is inconsistent with the receiver belief support."
            )

        self.belief = FinitePrior(
            name=f"{current_belief.name}_posterior",
            support=posterior_support,
            probabilities=posterior_probabilities,
        )

    def reset_for_evaluation(self) -> None:
        """Reset transient belief and histories while preserving cached routes."""
        self.belief = self.prior
        self._action_history.clear()
        self._realized_payoff_history.clear()

    def _compute_paths(self) -> RoutingSolution:
        belief = self._current_belief()
        if isinstance(belief, FinitePrior):
            scenarios = tuple(belief.support.values())
        else:
            scenarios = belief.sample(
                n_samples=self.n_scenarios,
                seed=self.rng_seed,
            )
        return solve_routes(
            world=self.world,
            source=self.demand.origin,
            target=self.demand.destination,
            scenarios=scenarios,
            config=self.routing_solver_config,
            use_average=self.average_sampling,
        )

    def _choose_max_element(
        self,
        solution: RoutingSolution,
        induced_pre_order: PreOrder,
    ) -> RoutingSolutionPoint:
        # If the sender is provided and we have more than one maximal element / BR
        # we break ties in favor of the sender
        # Otherwise, we choose randomly
        max_elements = induced_pre_order.maximal_elements()
        if len(max_elements) > 1 and self.sender:
            sender_pre_order = solution.induced_preorder(
                preference=self.sender.preference,
                labels=max_elements,
            )
            max_elements = sender_pre_order.maximal_elements()
        max_element = self._rng.choice(tuple(sorted(max_elements, key=str)))
        return solution.by_label(str(max_element))

    def _current_belief(self) -> WorldBelief | Prior:
        if self.belief is None:
            raise RuntimeError("Receiver has no internal belief.")
        return self.belief

    def _scenario_matches_signal(
        self,
        *,
        scenario: Scenario,
        signal: Signal,
    ) -> bool:
        for metric, observed_value in signal.value.items():
            try:
                realized_value = getattr(scenario, metric.value)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Scenario observations for {metric.value!r} are not supported."
                ) from exc
            if not self._values_match(
                realized_value=realized_value,
                observed_value=observed_value,
            ):
                return False
        return True

    @staticmethod
    def _values_match(
        *,
        realized_value: object,
        observed_value: object,
    ) -> bool:
        if isinstance(observed_value, Mapping):
            if not isinstance(realized_value, Mapping):
                return False
            return all(
                key in realized_value and realized_value[key] == value
                for key, value in observed_value.items()
            )
        return realized_value == observed_value

    def _validate_demand(self) -> None:
        if self.demand.origin not in self.world.V:
            raise ValueError(
                f"Receiver {self.id!r} has origin outside V: {self.demand.origin!r}."
            )
        if self.demand.destination not in self.world.V:
            raise ValueError(
                "Receiver "
                f"{self.id!r} has destination outside V: {self.demand.destination!r}."
            )

    def get_path_choice(self) -> RoutingSolutionPoint:
        # Order paths by self.preference
        solution = self._compute_paths()
        induced_pre_order = solution.induced_preorder(self.preference)
        max_element = self._choose_max_element(solution, induced_pre_order)
        self._action_history.append(max_element)
        return max_element

    def compute_realized_metrics(
        self,
        realized_scenario: Scenario,
    ) -> Mapping[MetricName, float]:
        """
        Return realized metric totals for the receiver's latest chosen path.

        The world computes one realized scenario after all receivers choose.
        Each receiver then extracts the metrics for its own path from that
        scenario. This keeps regret calculations local to the receiver while
        avoiding repeated world-level congestion calculations.
        """
        if not self._action_history:
            raise ValueError("Receiver has no path choice to evaluate.")

        path_choice = self._action_history[-1]
        realized_metrics = self._path_metric_totals(
            scenario=realized_scenario,
            path=path_choice.path,
        )
        self._realized_payoff_history.append(realized_metrics)
        return realized_metrics

    def _path_metric_totals(
        self,
        scenario: Scenario,
        path: tuple[Arc, ...],
    ) -> Mapping[MetricName, float]:
        left_turns = 0.0
        for previous_arc, next_arc in zip(path, path[1:]):
            turn = previous_arc[0], previous_arc[1], next_arc[1]
            if turn in self.world.L:
                left_turns += 1.0

        return {
            MetricName.TRAVEL_TIME: sum(scenario.travel_time[arc] for arc in path),
            MetricName.LEFT_TURNS: left_turns,
            MetricName.DISCOMFORT: sum(scenario.discomfort[arc] for arc in path),
            MetricName.HAZARD: sum(scenario.hazard[arc] for arc in path),
            MetricName.COST: sum(scenario.cost[arc] for arc in path),
            MetricName.EMISSIONS: sum(scenario.emissions[arc] for arc in path),
            MetricName.POLICING: sum(scenario.policing[arc[1]] for arc in path),
        }


@dataclass
class PriorRouteChoiceReceiver(Receiver):
    """
    Similar to receiver but finds shortest paths on prior and then chooses
    best route based on updated beliefs of the state.
    """

    _cached_prior_solution: RoutingSolution | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def _compute_paths(self) -> RoutingSolution:
        if self._cached_prior_solution is None:
            if isinstance(self.prior, FinitePrior):
                scenarios = tuple(self.prior.support.values())
            else:
                scenarios = self.prior.sample(
                    n_samples=self.n_scenarios,
                    seed=self.rng_seed,
                )
            self._cached_prior_solution = solve_routes(
                world=self.world,
                source=self.demand.origin,
                target=self.demand.destination,
                scenarios=scenarios,
                config=self.routing_solver_config,
                use_average=self.average_sampling,
            )
        return self._rescore_cached_paths(self._cached_prior_solution)

    def _rescore_cached_paths(
        self,
        cached_solution: RoutingSolution,
    ) -> RoutingSolution:
        return cached_solution.rescore(
            scorer=self._expected_objective_values_for_path,
        )

    def _expected_objective_values_for_path(
        self,
        path: tuple[Arc, ...],
    ) -> Mapping[MetricName, float]:
        belief = self._current_belief()
        totals = {metric: 0.0 for metric in MetricName}

        if isinstance(belief, FinitePrior):
            weighted_scenarios = tuple(
                (belief.probabilities[name], scenario)
                for name, scenario in belief.support.items()
            )
        else:
            sampled_scenarios = belief.sample(
                n_samples=self.n_scenarios,
                seed=self.rng_seed,
            )
            scenario_weight = (
                1.0 / len(sampled_scenarios)
                if self.average_sampling
                else 1.0
            )
            weighted_scenarios = tuple(
                (scenario_weight, scenario)
                for scenario in sampled_scenarios
            )

        for scenario_weight, scenario in weighted_scenarios:
            path_totals = self._path_metric_totals(scenario=scenario, path=path)
            for metric, value in path_totals.items():
                totals[metric] += scenario_weight * value

        return totals
