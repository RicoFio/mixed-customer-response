from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from .datastructures import (
    Node,
    Arc,
    Turn,
    MetricName,
    Scenario,
    BenpyModel,
    OBJECTIVE_VECTOR_ORDER,
    SampledPrior,
)
from .orders import PartialOrder


@dataclass(frozen=True)
class RoutingSolutionPoint:
    """One efficient routing solution recovered from BenPy."""

    label: str
    index: int
    objective_values: Mapping[MetricName, float]
    path: tuple[Arc, ...]
    arc_flows: Mapping[Arc, float]
    variable_values: tuple[float, ...]
    vertex_type: int | None = None

    def projection(
        self,
        x_metric: MetricName,
        y_metric: MetricName,
    ) -> tuple[float, float]:
        return self.objective_values[x_metric], self.objective_values[y_metric]


@dataclass(frozen=True)
class RoutingSolution:
    """Readable wrapper around a BenPy routing solution."""

    raw_solution: Any
    model: BenpyModel
    points: tuple[RoutingSolutionPoint, ...]

    @property
    def status(self) -> Any:
        return getattr(self.raw_solution, "status", None)

    @property
    def objective_names(self) -> tuple[MetricName, ...]:
        return tuple(self.model.meta["objective_names"])

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(point.label for point in self.points)

    @property
    def objective_values(self) -> Mapping[str, Mapping[MetricName, float]]:
        return {point.label: point.objective_values for point in self.points}

    @property
    def paths(self) -> Mapping[str, tuple[Arc, ...]]:
        return {point.label: point.path for point in self.points}

    @property
    def objective_array(self) -> np.ndarray:
        return np.array(
            [
                [point.objective_values[name] for name in self.objective_names]
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def vertex_value(self) -> np.ndarray:
        """Compatibility alias for plotting functions that accept BenPy output."""
        return self.objective_array

    @property
    def vertex_type(self) -> np.ndarray:
        return np.ones(len(self.points), dtype=int)

    @property
    def num_vertices_upper(self) -> Any:
        return getattr(self.raw_solution, "num_vertices_upper", None)

    def projection(
        self,
        x_metric: MetricName,
        y_metric: MetricName,
    ) -> list[tuple[str, float, float]]:
        return [
            (point.label, *point.projection(x_metric, y_metric))
            for point in self.points
        ]

    def best_by_metric(self, metric: MetricName) -> RoutingSolutionPoint:
        if metric not in self.objective_names:
            raise ValueError(f"Unknown metric: {metric!r}.")
        return min(self.points, key=lambda point: point.objective_values[metric])

    def by_label(self, label: str) -> RoutingSolutionPoint:
        for point in self.points:
            if point.label == label:
                return point
        raise KeyError(f"Unknown solution label: {label!r}.")

    def get_ordered_results(
        self,
        preference: PartialOrder,
    ) -> list[RoutingSolutionPoint]:
        """
        Return solutions sorted by the preference-induced metric order.

        The existing order convention is ``a <= b``. We therefore use minimal
        metrics first and compare objective values lexicographically. All current
        objectives are minimization objectives.
        """
        metric_order = _metric_order_from_preference(
            preference=preference,
            objective_names=self.objective_names,
        )
        return sorted(
            self.points,
            key=lambda point: tuple(point.objective_values[name] for name in metric_order),
        )

    @classmethod
    def from_benpy_solution(
        cls,
        raw_solution: Any,
        model: BenpyModel,
        tol: float = 1e-7,
    ) -> RoutingSolution:
        primal = getattr(raw_solution, "Primal", None)
        if primal is None:
            raise ValueError("BenPy solution does not include primal data.")

        vertex_values = np.asarray(getattr(primal, "vertex_value", None), dtype=float)
        if vertex_values.ndim != 2 or vertex_values.shape[0] == 0:
            raise ValueError("BenPy solution has no primal objective vertices.")

        preimages = np.asarray(getattr(primal, "preimage", None), dtype=float)
        if preimages.ndim not in {1, 2}:
            raise ValueError("Solve with options={'solution': True} to recover paths.")

        vertex_types = _vertex_types(primal=primal, n_points=vertex_values.shape[0])
        vertex_indices = _bounded_vertex_indices(vertex_types, vertex_values.shape[0])
        objective_names = tuple(model.meta["objective_names"])

        points: list[RoutingSolutionPoint] = []
        for label_idx, vertex_idx in enumerate(vertex_indices):
            variable_values = (
                preimages
                if preimages.ndim == 1
                else preimages[vertex_idx]
            )
            arc_flows = _arc_flows_from_values(
                model=model,
                variable_values=variable_values,
                tol=tol,
            )
            path = _ordered_path_from_arc_flows(
                arc_flows=arc_flows,
                source=model.meta.get("source"),
                target=model.meta.get("target"),
            )
            points.append(
                RoutingSolutionPoint(
                    label=_solution_label(label_idx),
                    index=int(vertex_idx),
                    objective_values={
                        objective_name: float(vertex_values[vertex_idx, objective_idx])
                        for objective_idx, objective_name in enumerate(objective_names)
                    },
                    path=path,
                    arc_flows=arc_flows,
                    variable_values=tuple(float(value) for value in variable_values),
                    vertex_type=(
                        None
                        if vertex_types is None
                        else int(vertex_types[vertex_idx])
                    ),
                )
            )

        return cls(
            raw_solution=raw_solution,
            model=model,
            points=tuple(points),
        )


def _vertex_types(primal: Any, n_points: int) -> np.ndarray | None:
    raw_vertex_types = getattr(primal, "vertex_type", None)
    if raw_vertex_types is None:
        return None

    vertex_types = np.asarray(raw_vertex_types)
    if vertex_types.shape != (n_points,):
        return None
    return vertex_types


def _bounded_vertex_indices(
    vertex_types: np.ndarray | None,
    n_points: int,
) -> np.ndarray:
    if vertex_types is None:
        return np.arange(n_points)

    bounded_indices = np.flatnonzero(vertex_types == 1)
    return bounded_indices if bounded_indices.size > 0 else np.arange(n_points)


def _arc_flows_from_values(
    model: BenpyModel,
    variable_values: np.ndarray,
    tol: float,
) -> dict[Arc, float]:
    return {
        arc: float(variable_values[variable_idx])
        for arc, variable_idx in model.index["x"].items()
        if variable_values[variable_idx] > tol
    }


def _ordered_path_from_arc_flows(
    arc_flows: Mapping[Arc, float],
    source: Node | None,
    target: Node | None,
) -> tuple[Arc, ...]:
    active_arcs = tuple(sorted(arc_flows, key=str))
    if source is None or target is None:
        return active_arcs

    outgoing: dict[Node, list[Arc]] = {}
    for arc in active_arcs:
        outgoing.setdefault(arc[0], []).append(arc)

    path: list[Arc] = []
    seen: set[Arc] = set()
    current = source
    while current != target:
        next_arcs = outgoing.get(current, [])
        if len(next_arcs) != 1:
            return active_arcs

        arc = next_arcs[0]
        if arc in seen:
            return active_arcs
        seen.add(arc)
        path.append(arc)
        current = arc[1]

    return tuple(path) if len(path) == len(active_arcs) else active_arcs


def _metric_order_from_preference(
    preference: PartialOrder,
    objective_names: Sequence[MetricName],
) -> tuple[MetricName, ...]:
    remaining = set(preference.elements).intersection(objective_names)
    metric_order: list[MetricName] = []

    while remaining:
        minimal_metrics = sorted(
            (
                metric
                for metric in remaining
                if not any(preference.less(other, metric) for other in remaining)
            ),
            key=str,
        )
        metric_order.extend(minimal_metrics)
        remaining.difference_update(minimal_metrics)

    metric_order.extend(
        metric for metric in objective_names if metric not in metric_order
    )
    return tuple(metric_order)


def _solution_label(point_idx: int) -> str:
    if point_idx < 26:
        return chr(ord("A") + point_idx)
    return str(point_idx + 1)


def make_independent_world_belief(
    V: Sequence[Node],
    A: Sequence[Arc],
    base_t: Mapping[Arc, float],
    base_discomfort: Mapping[Arc, float],
    base_h: Mapping[Arc, float],
    base_c: Mapping[Arc, float],
    base_e: Mapping[Arc, float],
    base_p: Mapping[Node, float],
    rel_noise: float = 0.1,
) -> SampledPrior:
    """Independent multiplicative perturbation model around base coefficients."""
    if rel_noise < 0:
        raise ValueError("rel_noise must be non-negative.")

    def sampler(rng: np.random.Generator, n_samples: int) -> list[Scenario]:
        def perturb_arcs(data: Mapping[Arc, float]) -> dict[Arc, float]:
            return {
                arc: max(0.0, data[arc] * (1.0 + rng.uniform(-rel_noise, rel_noise)))
                for arc in A
            }

        def perturb_nodes(data: Mapping[Node, float]) -> dict[Node, float]:
            return {
                node: max(0.0, data[node] * (1.0 + rng.uniform(-rel_noise, rel_noise)))
                for node in V
            }

        return [
            Scenario(
                name=f"rho_{sample_idx}",
                travel_time=perturb_arcs(base_t),
                discomfort=perturb_arcs(base_discomfort),
                hazard=perturb_arcs(base_h),
                cost=perturb_arcs(base_c),
                emissions=perturb_arcs(base_e),
                policing=perturb_nodes(base_p),
            )
            for sample_idx in range(n_samples)
        ]

    return SampledPrior(
        name="independent_multiplicative_noise",
        sampler=sampler,
    )


def build_benpy_model_sample_average(
    V: Sequence[Node],
    A: Sequence[Arc],
    L: Sequence[Turn],
    s: Node,
    t: Node,
    scenarios: Sequence[Scenario],
    use_average: bool = True,
) -> BenpyModel:
    """
    Build the sampled multi-objective routing LP for benpy.

    Decision variables are scenario-independent:
        x_{ij} in [0, 1]   arc flow
        y_{ijk} in [0, 1]  selected counted left turns
        z_i in [0, 1]      node exposure (to policing)

    The objective matrix has rows ordered as OBJECTIVE_VECTOR_ORDER.
    """
    _validate_routing_inputs(V=V, A=A, L=L, source=s, target=t)

    realized_scenarios = list(scenarios)
    _validate_scenarios(V=V, A=A, scenarios=realized_scenarios)
    scale = 1.0 / len(realized_scenarios) if use_average else 1.0

    x_index = {arc: idx for idx, arc in enumerate(A)}
    y_index = {turn: len(A) + idx for idx, turn in enumerate(L)}
    z_index = {node: len(A) + len(L) + idx for idx, node in enumerate(V)}

    n_variables = len(A) + len(L) + len(V)
    P = np.zeros((len(OBJECTIVE_VECTOR_ORDER), n_variables), dtype=float)
    objective_index = {
        metric_name: idx for idx, metric_name in enumerate(OBJECTIVE_VECTOR_ORDER)
    }

    for rho in realized_scenarios:
        for arc in A:
            P[objective_index["travel_time"], x_index[arc]] += (
                scale * rho.travel_time[arc]
            )
            P[objective_index["discomfort"], x_index[arc]] += (
                scale * rho.discomfort[arc]
            )
            P[objective_index["hazard"], x_index[arc]] += scale * rho.hazard[arc]
            P[objective_index["cost"], x_index[arc]] += scale * rho.cost[arc]
            P[objective_index["emissions"], x_index[arc]] += (
                scale * rho.emissions[arc]
            )
        for turn in L:
            P[objective_index["left_turns"], y_index[turn]] += scale
        for node in V:
            P[objective_index["policing"], z_index[node]] += (
                scale * rho.policing[node]
            )

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def add_leq(coeffs: Mapping[int, float], upper_bound: float) -> None:
        row = np.zeros(n_variables, dtype=float)
        for idx, value in coeffs.items():
            row[idx] += value
        rows.append(row)
        rhs.append(upper_bound)

    def add_eq(coeffs: Mapping[int, float], value: float) -> None:
        add_leq(coeffs, value)
        add_leq({idx: -coef for idx, coef in coeffs.items()}, -value)

    for node in V:
        coeffs: dict[int, float] = {}
        for i, j in A:
            if i == node:
                coeffs[x_index[(i, j)]] = coeffs.get(x_index[(i, j)], 0.0) + 1.0
            if j == node:
                coeffs[x_index[(i, j)]] = coeffs.get(x_index[(i, j)], 0.0) - 1.0

        flow_rhs = 1.0 if node == s else (-1.0 if node == t else 0.0)
        add_eq(coeffs, flow_rhs)

    for i, j, k in L:
        add_leq({y_index[(i, j, k)]: 1.0, x_index[(i, j)]: -1.0}, 0.0)
        add_leq({y_index[(i, j, k)]: 1.0, x_index[(j, k)]: -1.0}, 0.0)
        add_leq(
            {
                x_index[(i, j)]: 1.0,
                x_index[(j, k)]: 1.0,
                y_index[(i, j, k)]: -1.0,
            },
            1.0,
        )

    for i, j in A:
        coeffs = {x_index[(i, j)]: -1.0}
        for ii, jj, k in L:
            if ii == i and jj == j:
                coeffs[y_index[(ii, jj, k)]] = 1.0
        add_leq(coeffs, 0.0)

    for j, k in A:
        coeffs = {x_index[(j, k)]: -1.0}
        for i, jj, kk in L:
            if jj == j and kk == k:
                coeffs[y_index[(i, jj, kk)]] = 1.0
        add_leq(coeffs, 0.0)

    for node in V:
        if node == s:
            add_eq({z_index[node]: 1.0}, 0.0)
            continue

        coeffs = {z_index[node]: 1.0}
        for i, j in A:
            if j == node:
                coeffs[x_index[(i, j)]] = coeffs.get(x_index[(i, j)], 0.0) - 1.0
        add_eq(coeffs, 0.0)

    return BenpyModel(
        B=np.vstack(rows),
        P=P,
        b=np.array(rhs, dtype=float),
        l=np.zeros(n_variables, dtype=float),
        s=np.ones(n_variables, dtype=float),
        index={"x": x_index, "y": y_index, "z": z_index},
        meta={
            "objective_names": OBJECTIVE_VECTOR_ORDER,
            "scenarios": realized_scenarios,
            "source": s,
            "target": t,
            "use_average": use_average,
        },
    )


def _validate_routing_inputs(
    V: Sequence[Node],
    A: Sequence[Arc],
    L: Sequence[Turn],
    source: Node,
    target: Node,
) -> None:
    nodes = set(V)
    arcs = set(A)

    if source not in nodes:
        raise ValueError("source must be in V.")
    if target not in nodes:
        raise ValueError("target must be in V.")
    if source == target:
        raise ValueError("source and target must be distinct.")
    if not arcs:
        raise ValueError("A must contain at least one arc.")

    for i, j in arcs:
        if i not in nodes or j not in nodes:
            raise ValueError(f"arc {(i, j)!r} references a node outside V.")

    for i, j, k in L:
        if (i, j) not in arcs or (j, k) not in arcs:
            raise ValueError(f"turn {(i, j, k)!r} references an arc outside A.")


def _validate_scenarios(
    V: Sequence[Node],
    A: Sequence[Arc],
    scenarios: Sequence[Scenario],
) -> None:
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    arcs = set(A)
    nodes = set(V)
    for scenario in scenarios:
        _validate_scenario_arc_map(
            scenario.name,
            "travel_time",
            scenario.travel_time,
            arcs,
        )
        _validate_scenario_arc_map(
            scenario.name,
            "discomfort",
            scenario.discomfort,
            arcs,
        )
        _validate_scenario_arc_map(scenario.name, "hazard", scenario.hazard, arcs)
        _validate_scenario_arc_map(scenario.name, "cost", scenario.cost, arcs)
        _validate_scenario_arc_map(
            scenario.name,
            "emissions",
            scenario.emissions,
            arcs,
        )
        _validate_scenario_node_map(
            scenario.name,
            "policing",
            scenario.policing,
            nodes,
        )


def _validate_scenario_arc_map(
    scenario_name: str,
    metric_name: str,
    values: Mapping[Arc, float],
    arcs: set[Arc],
) -> None:
    keys = set(values)
    missing = arcs - keys
    extra = keys - arcs
    if missing:
        raise ValueError(
            f"Scenario {scenario_name!r} metric {metric_name!r} is missing arcs: "
            f"{missing!r}"
        )
    if extra:
        raise ValueError(
            f"Scenario {scenario_name!r} metric {metric_name!r} has arcs outside A: "
            f"{extra!r}"
        )


def _validate_scenario_node_map(
    scenario_name: str,
    metric_name: str,
    values: Mapping[Node, float],
    nodes: set[Node],
) -> None:
    keys = set(values)
    missing = nodes - keys
    extra = keys - nodes
    if missing:
        raise ValueError(
            f"Scenario {scenario_name!r} metric {metric_name!r} is missing nodes: "
            f"{missing!r}"
        )
    if extra:
        raise ValueError(
            f"Scenario {scenario_name!r} metric {metric_name!r} has nodes outside V: "
            f"{extra!r}"
        )
