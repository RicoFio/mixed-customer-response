from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, TypeAlias

import benpy
import networkx as nx
import numpy as np


Node: TypeAlias = str | tuple[int, int]
Arc: TypeAlias = tuple[Node, Node]
Turn: TypeAlias = tuple[Node, Node, Node]
MetricName: TypeAlias = str

METRIC_SET = frozenset(
    {
        "travel_time",
        "left_turns",
        "discomfort",
        "hazard",
        "cost",
        "emissions",
        "policing",
    }
)

OBJECTIVE_VECTOR_ORDER: tuple[MetricName, ...] = (
    "travel_time",
    "left_turns",
    "discomfort",
    "hazard",
    "cost",
    "emissions",
    "policing",
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
    nominal_discomfort: Mapping[Arc, float] = field(default_factory=dict)
    nominal_hazards: Mapping[Arc, float] = field(default_factory=dict)
    nominal_cost: Mapping[Arc, float] = field(default_factory=dict)
    nominal_policing: Mapping[Node, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "V", set(self.V))
        object.__setattr__(self, "A", set(self.A))
        object.__setattr__(self, "L", set(self.L))
        object.__setattr__(self, "I", set(self.I))
        object.__setattr__(self, "nominal_travel_time", dict(self.nominal_travel_time))
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
        self._validate_arc_map("nominal_discomfort", self.nominal_discomfort)
        self._validate_arc_map("nominal_hazards", self.nominal_hazards)
        self._validate_arc_map("nominal_cost", self.nominal_cost)
        self._validate_node_map("nominal_policing", self.nominal_policing)

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "individuals", frozenset(self.individuals))
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


class WorldBelief(Protocol):
    name: str

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]:
        ...


@dataclass(frozen=True)
class FinitePrior:
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
class SampledPrior:
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
class Sender:
    id: str
    preference: Any | None = None
    belief: WorldBelief | None = None


@dataclass(frozen=True)
class Receiver(Individual):
    preference: Any | None = None
    belief: WorldBelief | None = None


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
