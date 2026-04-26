from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, TypeAlias

import benpy
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .opt import RoutingSolutionPoint


Node: TypeAlias = str | tuple[int, int]
Arc: TypeAlias = tuple[Node, Node]
Turn: TypeAlias = tuple[Node, Node, Node]


class MetricName(str, Enum):
    TRAVEL_TIME = "travel_time"
    LEFT_TURNS = "left_turns"
    DISCOMFORT = "discomfort"
    HAZARD = "hazard"
    COST = "cost"
    EMISSIONS = "emissions"
    POLICING = "policing"

    @classmethod
    def coerce(cls, value: MetricName | str) -> MetricName:
        if isinstance(value, cls):
            return value
        return cls(value)

    def __str__(self) -> str:
        return self.value


ArcScenarioOverrides: TypeAlias = (
    Mapping[MetricName, Mapping[Arc, float]]
    | Mapping[str, Mapping[Arc, float]]
)
NodeScenarioOverrides: TypeAlias = (
    Mapping[MetricName, Mapping[Node, float]]
    | Mapping[str, Mapping[Node, float]]
)


METRIC_SET = frozenset(MetricName)

OBJECTIVE_VECTOR_ORDER: tuple[MetricName, ...] = (
    MetricName.TRAVEL_TIME,
    MetricName.LEFT_TURNS,
    MetricName.DISCOMFORT,
    MetricName.HAZARD,
    MetricName.COST,
    MetricName.EMISSIONS,
    MetricName.POLICING,
)

if frozenset(OBJECTIVE_VECTOR_ORDER) != METRIC_SET:
    raise ValueError("OBJECTIVE_VECTOR_ORDER must contain exactly METRIC_SET.")


class Singleton(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class InfrastructureGraph:
    """Physical network and nominal infrastructure attributes."""

    V: set[Node] = field(default_factory=set)
    A: set[Arc] = field(default_factory=set)
    L: set[Turn] = field(default_factory=set)
    I: set[Arc] = field(default_factory=set)

    nominal_travel_time: Mapping[Arc, float] = field(default_factory=dict)
    nominal_link_capacity: Mapping[Arc, float] = field(default_factory=dict)
    arc_distances: Mapping[Arc, float] = field(default_factory=dict)
    nominal_discomfort: Mapping[Arc, float] = field(default_factory=dict)
    nominal_hazards: Mapping[Arc, float] = field(default_factory=dict)
    nominal_cost: Mapping[Arc, float] = field(default_factory=dict)
    nominal_policing: Mapping[Node, float] = field(default_factory=dict)
    
    bpr_alpha: float = 0.15
    bpr_beta: float = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "V", set(self.V))
        object.__setattr__(self, "A", set(self.A))
        object.__setattr__(self, "L", set(self.L))
        object.__setattr__(self, "I", set(self.I))
        object.__setattr__(self, "nominal_travel_time", dict(self.nominal_travel_time))
        object.__setattr__(
            self,
            "nominal_link_capacity",
            self._complete_arc_map(
                values=self.nominal_link_capacity,
                default=1.0,
            ),
        )
        object.__setattr__(
            self,
            "arc_distances",
            self._complete_arc_map(
                values=self.arc_distances,
                default=lambda arc: float(self.nominal_travel_time[arc]),
            ),
        )
        object.__setattr__(self, "nominal_discomfort", dict(self.nominal_discomfort))
        object.__setattr__(self, "nominal_hazards", dict(self.nominal_hazards))
        object.__setattr__(self, "nominal_cost", dict(self.nominal_cost))
        object.__setattr__(self, "nominal_policing", dict(self.nominal_policing))
        self._validate()

    @classmethod
    def from_networkx(cls, G: nx.DiGraph) -> InfrastructureGraph:
        """
        Build infrastructure from a directed NetworkX graph.

        Required edge attribute:
        - travel_time

        Optional edge attributes default to simple neutral values:
        - capacity: 1.0
        - distance: travel_time
        - discomfort: 0.5 for instrumented arcs, 1.0 otherwise
        - hazard: 0.0
        - toll: 0.0
        - instrumented: false

        Optional node attribute:
        - policing: bool
        """
        V = set(G.nodes)
        A = set(G.edges)
        I = {
            (u, v)
            for u, v, data in G.edges(data=True)
            if bool(data.get("instrumented", False))
        }

        def edge_value(
            attr: str,
            default: Callable[[Arc], float] | float,
        ) -> dict[Arc, float]:
            values: dict[Arc, float] = {}
            for arc in A:
                data = G.edges[arc]
                fallback = default(arc) if callable(default) else default
                values[arc] = float(data.get(attr, fallback))
            return values

        missing_travel_time = [
            arc for arc in A if "travel_time" not in G.edges[arc]
        ]
        if missing_travel_time:
            raise ValueError(
                f"Missing travel_time for arcs: {missing_travel_time!r}"
            )

        return cls(
            V=V,
            A=A,
            I=I,
            nominal_travel_time=edge_value("travel_time", 0.0),
            nominal_link_capacity=edge_value("capacity", 1.0),
            arc_distances=edge_value(
                "distance",
                lambda arc: float(G.edges[arc]["travel_time"]),
            ),
            nominal_discomfort=edge_value(
                "discomfort",
                lambda arc: 0.5 if arc in I else 1.0,
            ),
            nominal_hazards=edge_value("hazard", 0.0),
            nominal_cost=edge_value("toll", 0.0),
            nominal_policing={
                node: float(G.nodes[node].get("policing", 0.0))
                for node in V
            },
        )

    def _validate(self) -> None:
        for i, j in self.A:
            if i not in self.V or j not in self.V:
                raise ValueError(f"Arc {(i, j)!r} references a node outside V.")

        if not self.I.issubset(self.A):
            raise ValueError("Instrumented arcs I must be a subset of A.")

        for i, j, k in self.L:
            if (i, j) not in self.A or (j, k) not in self.A:
                raise ValueError(f"Turn {(i, j, k)!r} references an arc outside A.")

        self._validate_arc_map("nominal_travel_time", self.nominal_travel_time)
        self._validate_arc_map("nominal_link_capacity", self.nominal_link_capacity)
        self._validate_arc_map("arc_distances", self.arc_distances)
        self._validate_arc_map("nominal_discomfort", self.nominal_discomfort)
        self._validate_arc_map("nominal_hazards", self.nominal_hazards)
        self._validate_arc_map("nominal_cost", self.nominal_cost)
        self._validate_node_map("nominal_policing", self.nominal_policing)
        if any(capacity <= 0 for capacity in self.nominal_link_capacity.values()):
            raise ValueError("nominal_link_capacity values must be positive.")
        if any(distance < 0 for distance in self.arc_distances.values()):
            raise ValueError("arc_distances values cannot be negative.")

    def _complete_arc_map(
        self,
        values: Mapping[Arc, float],
        default: Callable[[Arc], float] | float,
    ) -> dict[Arc, float]:
        raw_values = dict(values)
        extra = set(raw_values) - self.A
        if extra:
            raise ValueError(f"Arc map has arcs outside A: {extra!r}")
        return {
            arc: float(
                raw_values[arc]
                if arc in raw_values
                else (default(arc) if callable(default) else default)
            )
            for arc in self.A
        }

    def _validate_arc_map(self, name: str, values: Mapping[Arc, float]) -> None:
        keys = set(values)
        missing = self.A - keys
        extra = keys - self.A
        if missing:
            raise ValueError(f"{name} is missing arcs: {missing!r}")
        if extra:
            raise ValueError(f"{name} has arcs outside A: {extra!r}")

    def _validate_node_map(self, name: str, values: Mapping[Node, float]) -> None:
        keys = set(values)
        missing = self.V - keys
        extra = keys - self.V
        if missing:
            raise ValueError(f"{name} is missing nodes: {missing!r}")
        if extra:
            raise ValueError(f"{name} has nodes outside V: {extra!r}")
    
    @cached_property
    def ordered_arcs(self) -> tuple[Arc, ...]:
        return tuple(sorted(self.A, key=str))
    
    @cached_property
    def ordered_capacities(self) -> np.ndarray:
        return np.array(
            [self.nominal_link_capacity[arc] for arc in self.ordered_arcs],
            dtype=float,
        )
    
    @cached_property
    def ordered_travel_times(self) -> np.ndarray:
        return np.array(
            [self.nominal_travel_time[arc] for arc in self.ordered_arcs],
            dtype=float,
        )
    
    @cached_property
    def ordered_arc_distances(self) -> np.ndarray:
        return np.array(
            [self.arc_distances[arc] for arc in self.ordered_arcs],
            dtype=float,
        )

    def get_actual_travel_times(self, volumes: Mapping[Arc, float]) -> Mapping[Arc, float]:
        """Return congestion-adjusted per-arc travel times using the BPR formula."""
        self._validate_volume_map(volumes)
        ordered_volumes = np.array(
            [float(volumes.get(arc, 0.0)) for arc in self.ordered_arcs],
            dtype=float,
        )
        ratios = ordered_volumes / self.ordered_capacities
        actual_travel_times = self.ordered_travel_times * (
            1 + self.bpr_alpha * (ratios ** self.bpr_beta)
        )
        
        return {
            arc: float(value)
            for arc, value in zip(self.ordered_arcs, actual_travel_times)
        }

    def get_actual_emissions(self, volumes: Mapping[Arc, float]) -> Mapping[Arc, float]:
        """
        Return congestion-adjusted per-arc emissions.

        This is currently a simple distance-weighted congestion proxy:
        base emissions scale with arc distance, then increase in proportion to
        realized travel time divided by nominal travel time. It keeps emissions
        scenario-level and replaceable once a calibrated emissions model exists.
        """
        actual_travel_times = self.get_actual_travel_times(volumes)
        ordered_actual_travel_times = np.array(
            [actual_travel_times[arc] for arc in self.ordered_arcs],
            dtype=float,
        )
        travel_time_ratio = np.divide(
            ordered_actual_travel_times,
            self.ordered_travel_times,
            out=np.ones_like(ordered_actual_travel_times),
            where=self.ordered_travel_times > 0,
        )
        actual_emissions = self.ordered_arc_distances * travel_time_ratio
        
        return {
            arc: float(value)
            for arc, value in zip(self.ordered_arcs, actual_emissions)
        }

    def _validate_volume_map(self, volumes: Mapping[Arc, float]) -> None:
        extra = set(volumes) - self.A
        if extra:
            raise ValueError(f"Volume map has arcs outside A: {extra!r}")
        if any(volume < 0 for volume in volumes.values()):
            raise ValueError("Arc volumes cannot be negative.")


@dataclass(frozen=True)
class Demand:
    origin: Node
    destination: Node


@dataclass(frozen=True)
class Individual:
    id: str
    demand: Demand


@dataclass(frozen=True)
class World:
    """Infrastructure plus the current population using it."""

    network: InfrastructureGraph
    individuals: frozenset[Individual] = field(default_factory=frozenset)
    normalized_population: bool = False
    
    # Ordered elements for later observation
    ordered_V: tuple[Node, ...] = field(init=False, repr=False)
    ordered_A: tuple[Arc, ...] = field(init=False, repr=False)
    ordered_L: tuple[Turn, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "individuals", frozenset(self.individuals))
        object.__setattr__(self, "ordered_V", tuple(sorted(self.network.V, key=str)))
        object.__setattr__(self, "ordered_A", tuple(sorted(self.network.A, key=str)))
        object.__setattr__(self, "ordered_L", tuple(sorted(self.network.L, key=str)))
        for individual in self.individuals:
            demand = individual.demand
            if demand.origin not in self.network.V:
                raise ValueError(
                    f"Individual {individual.id!r} has origin outside V: "
                    f"{demand.origin!r}"
                )
            if demand.destination not in self.network.V:
                raise ValueError(
                    f"Individual {individual.id!r} has destination outside V: "
                    f"{demand.destination!r}"
                )

    @property
    def V(self) -> set[Node]:
        return self.network.V

    @property
    def A(self) -> set[Arc]:
        return self.network.A

    @property
    def L(self) -> set[Turn]:
        return self.network.L

    @property
    def I(self) -> set[Arc]:
        return self.network.I

    @property
    def travel_time(self) -> Mapping[Arc, float]:
        return self.network.nominal_travel_time

    @property
    def discomfort(self) -> Mapping[Arc, float]:
        return self.network.nominal_discomfort

    @property
    def hazard(self) -> Mapping[Arc, float]:
        return self.network.nominal_hazards

    @property
    def cost(self) -> Mapping[Arc, float]:
        return self.network.nominal_cost

    @property
    def policing(self) -> Mapping[Node, float]:
        return self.network.nominal_policing

    @property
    def total_population(self) -> int | float:
        return 1.0 if self.normalized_population else len(self.individuals)

    def population_at_node(self, node: Node) -> int | float:
        if node not in self.network.V:
            raise ValueError(f"Node {node!r} is not in the infrastructure.")

        population = sum(
            1 for individual in self.individuals if individual.demand.origin == node
        )
        return population / self.total_population if self.normalized_population else population

    def total_demand(self, origin: Node, destination: Node) -> int:
        return sum(
            1
            for individual in self.individuals
            if individual.demand.origin == origin
            and individual.demand.destination == destination
        )    

    def get_realized_metrics(
        self,
        path_choices: Mapping[Individual, RoutingSolutionPoint],
        name: str = "realized",
        base_scenario: Scenario | None = None,
    ) -> Scenario:
        """
        Return one realized scenario after all individual path choices are known.

        The realization is computed at the world level. First, every selected
        path contributes to an arc-volume map. The infrastructure then turns
        those volumes into congestion-dependent travel times and emissions.
        Static metrics such as discomfort, hazard, cost, and policing are copied
        either from a provided base scenario or, if none is provided, from the
        nominal infrastructure. Individual regret can later be computed by
        summing this returned scenario over each individual's chosen path.
        """
        self._validate_path_choices(path_choices)
        arc_volumes = self._arc_volumes_from_path_choices(path_choices)
        actual_travel_times = self.network.get_actual_travel_times(arc_volumes)
        actual_emissions = self.network.get_actual_emissions(arc_volumes)
        return Scenario(
            name=name,
            travel_time=actual_travel_times,
            discomfort=(
                dict(base_scenario.discomfort)
                if base_scenario is not None
                else dict(self.discomfort)
            ),
            hazard=(
                dict(base_scenario.hazard)
                if base_scenario is not None
                else dict(self.hazard)
            ),
            cost=(
                dict(base_scenario.cost)
                if base_scenario is not None
                else dict(self.cost)
            ),
            emissions=actual_emissions,
            policing=(
                dict(base_scenario.policing)
                if base_scenario is not None
                else dict(self.policing)
            ),
        )

    def _validate_path_choices(
        self,
        path_choices: Mapping[Individual, RoutingSolutionPoint],
    ) -> None:
        chosen_individuals = set(path_choices)
        if chosen_individuals != self.individuals:
            missing = self.individuals - chosen_individuals
            extra = chosen_individuals - self.individuals
            raise ValueError(
                "Path choices must contain exactly the world's individuals. "
                f"Missing: {missing!r}; extra: {extra!r}."
            )

        for individual, choice in path_choices.items():
            if not choice.path:
                raise ValueError(f"Individual {individual.id!r} has an empty path.")
            for arc in choice.path:
                if arc not in self.A:
                    raise ValueError(
                        f"Individual {individual.id!r} chose arc outside A: {arc!r}."
                    )

    def _arc_volumes_from_path_choices(
        self,
        path_choices: Mapping[Individual, RoutingSolutionPoint],
    ) -> dict[Arc, float]:
        unit_flow = 1.0 / self.total_population if self.normalized_population else 1.0
        arc_volumes = {arc: 0.0 for arc in self.A}
        for choice in path_choices.values():
            for arc in choice.path:
                arc_volumes[arc] += unit_flow
        return arc_volumes


@dataclass(frozen=True)
class Scenario:
    """One sampled world realization."""

    name: str
    travel_time: Mapping[Arc, float]
    discomfort: Mapping[Arc, float]
    hazard: Mapping[Arc, float]
    cost: Mapping[Arc, float]
    emissions: Mapping[Arc, float]
    policing: Mapping[Node, float]
    
    @classmethod
    def from_world(
        cls,
        name: str,
        world: World,
        arc_overrides: ArcScenarioOverrides | None = None,
        node_overrides: NodeScenarioOverrides | None = None,
    ) -> Scenario:
        """Create a deterministic nominal scenario from a world."""
        zero_volumes = {arc: 0.0 for arc in world.A}
        scenario = cls(
            name=name,
            travel_time=dict(world.travel_time),
            discomfort=dict(world.discomfort),
            hazard=dict(world.hazard),
            cost=dict(world.cost),
            emissions=world.network.get_actual_emissions(zero_volumes),
            policing=dict(world.policing),
        )
        return scenario.with_overrides(
            arc_overrides=arc_overrides,
            node_overrides=node_overrides,
        )

    def with_overrides(
        self,
        *,
        name: str | None = None,
        arc_overrides: ArcScenarioOverrides | None = None,
        node_overrides: NodeScenarioOverrides | None = None,
    ) -> Scenario:
        """Return a new scenario with selected arc or node metrics replaced."""
        arc_metrics = {
            MetricName.TRAVEL_TIME: dict(self.travel_time),
            MetricName.DISCOMFORT: dict(self.discomfort),
            MetricName.HAZARD: dict(self.hazard),
            MetricName.COST: dict(self.cost),
            MetricName.EMISSIONS: dict(self.emissions),
        }
        node_metrics = {
            MetricName.POLICING: dict(self.policing),
        }

        for metric, updates in (arc_overrides or {}).items():
            metric_name = MetricName.coerce(metric)
            if metric_name not in arc_metrics:
                raise ValueError(f"{metric_name.value!r} is not an arc scenario metric.")
            _apply_metric_updates(
                values=arc_metrics[metric_name],
                updates=updates,
                metric_name=metric_name.value,
            )

        for metric, updates in (node_overrides or {}).items():
            metric_name = MetricName.coerce(metric)
            if metric_name not in node_metrics:
                raise ValueError(f"{metric_name.value!r} is not a node scenario metric.")
            _apply_metric_updates(
                values=node_metrics[metric_name],
                updates=updates,
                metric_name=metric_name.value,
            )

        return replace(
            self,
            name=self.name if name is None else name,
            travel_time=arc_metrics[MetricName.TRAVEL_TIME],
            discomfort=arc_metrics[MetricName.DISCOMFORT],
            hazard=arc_metrics[MetricName.HAZARD],
            cost=arc_metrics[MetricName.COST],
            emissions=arc_metrics[MetricName.EMISSIONS],
            policing=node_metrics[MetricName.POLICING],
        )


def _apply_metric_updates(
    values: dict[Any, float],
    updates: Mapping[Any, float],
    metric_name: str,
) -> None:
    extra_keys = set(updates) - set(values)
    if extra_keys:
        raise ValueError(
            f"{metric_name!r} overrides reference unknown keys: {extra_keys!r}"
        )
    for key, value in updates.items():
        values[key] = float(value)


class Belief(Protocol):
    name: str

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]:
        ...


@dataclass(frozen=True)
class WorldBelief(Belief):
    pass


@dataclass(frozen=True)
class Prior(Belief):
    pass


@dataclass(frozen=True)
class FinitePrior(Prior):
    """Finite scenario prior sampled with replacement."""

    name: str
    support: Mapping[str, Scenario]
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError("FinitePrior support cannot be empty.")

        support = dict(self.support)
        probabilities = dict(self.probabilities)

        if set(support) != set(probabilities):
            raise ValueError("FinitePrior support and probability keys must match.")
        if any(probability < 0 for probability in probabilities.values()):
            raise ValueError("FinitePrior probabilities cannot be negative.")

        total_probability = sum(probabilities.values())
        if total_probability <= 0:
            raise ValueError("FinitePrior probabilities must sum to a positive value.")

        normalized = {
            name: probability / total_probability
            for name, probability in probabilities.items()
        }
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "probabilities", normalized)

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        rng = np.random.default_rng(seed)
        names = list(self.support)
        probabilities = np.array([self.probabilities[name] for name in names])
        sampled_names = rng.choice(names, size=n_samples, replace=True, p=probabilities)
        return [self.support[name] for name in sampled_names]


@dataclass(frozen=True)
class ContinuousPrior(Prior):
    pass


@dataclass(frozen=True)
class BetaPrior(ContinuousPrior):
    pass


@dataclass(frozen=True)
class SampledPrior(Prior):
    """Sampler-backed uncertainty over scenarios."""

    name: str
    sampler: Callable[[np.random.Generator, int], list[Scenario]]

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng = np.random.default_rng(seed)
        scenarios = self.sampler(rng, n_samples)
        if len(scenarios) != n_samples:
            raise ValueError("Sampler must return exactly n_samples scenarios.")
        return scenarios


@dataclass(frozen=True)
class BenpyModel:
    """Named arrays for benpy.solve."""

    B: np.ndarray
    P: np.ndarray
    b: np.ndarray
    l: np.ndarray
    s: np.ndarray
    index: Mapping[str, Mapping[Any, int]]
    meta: Mapping[str, Any]

    def solve(self, options: Mapping[str, Any] | None = None):
        solver_options = {
            "message_level": 0,
            "lp_message_level": 0,
        }
        if options is not None:
            solver_options.update(options)

        return benpy.solve(
            self.B,
            self.P,
            b=self.b,
            l=self.l,
            s=self.s,
            opt_dir=1,
            options=solver_options,
        )
