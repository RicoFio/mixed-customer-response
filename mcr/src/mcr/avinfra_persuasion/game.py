from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

import numpy as np

from .datastructures import (
    Arc,
    Demand,
    Individual,
    MetricName,
    Scenario,
    World,
    WorldBelief,
    Prior
)
from .orders import PartialOrder, PreOrder
from .opt import (
    RoutingSolution,
    RoutingSolutionPoint,
)
from .routing.routing_solvers import (
    RoutingSolverConfig,
    RoutingSolverConfigLike,
    solve_routes,
)

TimeStep: TypeAlias = int
ReceiverType: TypeAlias = str
Preference: TypeAlias = PartialOrder

RNG_SEED: int = 1
N_SCENARIOS: int = 10
AVERAGE_SAMPLING: bool = True


@dataclass
class Signal:
    pass


@dataclass
class SignalPolicy:
    pass


@dataclass
class Sender:
    preference: Preference
    prior: WorldBelief
    world: World
    signal_policy: SignalPolicy

    _regret_history: list[float]


@dataclass
class Receiver:
    individual: Individual
    rtype: ReceiverType
    preference: Preference
    prior: Prior
    world: World
    sender: Sender | None = None
    belief: WorldBelief | None = None
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
        # Do a Bayesian update of the internal belief given the provided signal 
        pass

    def _compute_paths(self) -> RoutingSolution:
        scenarios = self._current_belief().sample(
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

    def _current_belief(self) -> WorldBelief:
        if self.belief is None:
            raise RuntimeError("Receiver has no internal belief.")
        return self.belief

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
class Game:
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: WorldBelief
    horizon: TimeStep

    current_step: TimeStep = 0

    def step(self):
        # First we sample from the world
        # Then we have the receiver commit to a signaling policy
        # We pass the sample of the world to the receiver and get back one message per receiver
        # We pass the messages to the receivers, they update their beliefs and take an action
        # We pass the actions to the current state of the world and receive the realized state of the world (congestion, travel times, emissions, etc.)
        # We pass that back to both the sender and the receivers.
        # The receiver updates their signaling policy (Not Implemented Yet).
        pass
