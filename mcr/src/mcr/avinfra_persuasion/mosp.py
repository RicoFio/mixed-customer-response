from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .datastructures import (
    Arc,
    BenpyModel,
    MetricName,
    Node,
    OBJECTIVE_VECTOR_ORDER,
    Scenario,
    Turn,
    World,
)
from .opt import (
    EXPANDED_END,
    EXPANDED_START,
    RoutingSolution,
    RoutingSolutionPoint,
    _build_turn_state_edges,
    _build_turn_state_nodes,
    _original_arc_for_turn_state_edge,
    _solution_label,
    _turn_for_turn_state_edge,
    _validate_routing_inputs,
    _validate_scenarios,
)

try:
    from . import _mosp_ext
except ImportError as exc:  # pragma: no cover - exercised only before extension build.
    _mosp_ext = None
    _EXTENSION_IMPORT_ERROR: ImportError | None = exc
else:
    _EXTENSION_IMPORT_ERROR = None


DEFAULT_SCALE = 1000.0
UINT32_MAX = 2**32 - 1

__all__ = [
    "DEFAULT_SCALE",
    "MospRawSolution",
    "solve_mosp_routes",
    "solve_mosp_routes_from_components",
]


@dataclass(frozen=True)
class MospRawSolution:
    """Small raw-solution shim for compatibility with RoutingSolution."""

    stats: Mapping[str, Any]
    status: str = "optimal"

    @property
    def num_vertices_upper(self) -> int:
        return int(self.stats.get("solutions_count", 0))


def solve_mosp_routes(
    world: World,
    source: Node,
    target: Node,
    scenarios: Sequence[Scenario],
    *,
    use_average: bool = True,
    scale: float = DEFAULT_SCALE,
) -> RoutingSolution:
    """Solve one MCR routing query with the wrapped C++ MDA algorithm."""
    return solve_mosp_routes_from_components(
        V=world.ordered_V,
        A=world.ordered_A,
        L=world.ordered_L,
        source=source,
        target=target,
        scenarios=scenarios,
        use_average=use_average,
        scale=scale,
    )


def solve_mosp_routes_from_components(
    *,
    V: Sequence[Node],
    A: Sequence[Arc],
    L: Sequence[Turn],
    source: Node,
    target: Node,
    scenarios: Sequence[Scenario],
    use_average: bool = True,
    scale: float = DEFAULT_SCALE,
) -> RoutingSolution:
    """Solve a routing query from explicit MCR graph/scenario components."""
    _require_extension()
    _validate_scale(scale)

    nodes = tuple(V)
    arcs = tuple(A)
    turns = tuple(L)
    realized_scenarios = tuple(scenarios)
    _validate_routing_inputs(V=nodes, A=arcs, L=turns, source=source, target=target)
    _validate_scenarios(V=nodes, A=arcs, scenarios=realized_scenarios)

    compiled_dimension = _mosp_ext.compiled_dimension()
    if compiled_dimension != len(OBJECTIVE_VECTOR_ORDER):
        raise RuntimeError(
            "The MDA extension was compiled with dimension "
            f"{compiled_dimension}, expected {len(OBJECTIVE_VECTOR_ORDER)}."
        )

    expanded_nodes = _build_turn_state_nodes(arcs)
    expanded_edges = _build_turn_state_edges(A=arcs, source=source, target=target)
    expanded_node_index = {
        expanded_node: node_idx
        for node_idx, expanded_node in enumerate(expanded_nodes)
    }
    tails = [expanded_node_index[tail] for tail, _ in expanded_edges]
    heads = [expanded_node_index[head] for _, head in expanded_edges]

    original_arc_by_expanded_arc = tuple(
        _original_arc_for_turn_state_edge(edge)
        for edge in expanded_edges
    )
    turn_by_expanded_arc = tuple(
        _turn_for_turn_state_edge(edge)
        for edge in expanded_edges
    )
    flat_costs = _scaled_edge_costs(
        original_arc_by_expanded_arc=original_arc_by_expanded_arc,
        turn_by_expanded_arc=turn_by_expanded_arc,
        counted_turns=set(turns),
        scenarios=realized_scenarios,
        use_average=use_average,
        scale=scale,
        max_path_edges=max(1, len(expanded_nodes) - 1),
    )

    raw_result = _mosp_ext.run_mda(
        len(expanded_nodes),
        tails,
        heads,
        flat_costs,
        expanded_node_index[EXPANDED_START],
        expanded_node_index[EXPANDED_END],
    )
    stats = dict(raw_result["stats"])

    points = []
    for point_idx, path_result in enumerate(raw_result["paths"]):
        path = _original_path_from_expanded_arc_ids(
            arc_ids=path_result["arc_ids"],
            original_arc_by_expanded_arc=original_arc_by_expanded_arc,
        )
        objective_values = _objective_values_for_path(
            path=path,
            scenarios=realized_scenarios,
            counted_turns=set(turns),
            use_average=use_average,
        )
        points.append(
            RoutingSolutionPoint(
                label=_solution_label(point_idx),
                index=point_idx,
                objective_values=objective_values,
                path=path,
                arc_flows=_arc_flows_for_path(path),
                variable_values=tuple(
                    cost / scale
                    for cost in path_result["costs"]
                ),
                vertex_type=1,
            )
        )

    return RoutingSolution(
        raw_solution=MospRawSolution(stats=stats),
        model=_mosp_model_stub(
            source=source,
            target=target,
            scenarios=realized_scenarios,
            use_average=use_average,
            scale=scale,
            stats=stats,
        ),
        points=tuple(points),
    )


def _require_extension() -> None:
    if _mosp_ext is None:
        raise RuntimeError(
            "The MDA Cython extension is not built. Run "
            "`uv run python mcr/setup.py build_ext --inplace` from the repo root."
        ) from _EXTENSION_IMPORT_ERROR


def _validate_scale(scale: float) -> None:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number.")


def _scaled_edge_costs(
    *,
    original_arc_by_expanded_arc: Sequence[Arc | None],
    turn_by_expanded_arc: Sequence[Turn | None],
    counted_turns: set[Turn],
    scenarios: Sequence[Scenario],
    use_average: bool,
    scale: float,
    max_path_edges: int,
) -> list[int]:
    flat_costs: list[int] = []
    max_scaled_cost = 0
    for expanded_arc_idx, (original_arc, turn) in enumerate(
        zip(original_arc_by_expanded_arc, turn_by_expanded_arc)
    ):
        values = _edge_objective_values(
            original_arc=original_arc,
            turn=turn,
            counted_turns=counted_turns,
            scenarios=scenarios,
            use_average=use_average,
        )
        for metric, value in zip(OBJECTIVE_VECTOR_ORDER, values):
            scaled = _scale_cost(
                value=value,
                scale=scale,
                context=f"expanded arc {expanded_arc_idx} metric {metric}",
            )
            max_scaled_cost = max(max_scaled_cost, scaled)
            flat_costs.append(scaled)

    if max_scaled_cost * max_path_edges > UINT32_MAX:
        raise OverflowError(
            "scaled path costs may overflow uint32_t; reduce scale or metric magnitudes."
        )
    return flat_costs


def _scale_cost(*, value: float, scale: float, context: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite.")
    if value < 0:
        raise ValueError(f"{context} must be non-negative.")

    scaled = int(round(value * scale))
    if scaled > UINT32_MAX:
        raise OverflowError(f"{context} exceeds uint32_t after scaling.")
    return scaled


def _edge_objective_values(
    *,
    original_arc: Arc | None,
    turn: Turn | None,
    counted_turns: set[Turn],
    scenarios: Sequence[Scenario],
    use_average: bool,
) -> tuple[float, ...]:
    if original_arc is None:
        return tuple(0.0 for _ in OBJECTIVE_VECTOR_ORDER)

    weight = _scenario_weight(scenarios=scenarios, use_average=use_average)
    values = {
        MetricName.TRAVEL_TIME: sum(
            weight * scenario.travel_time[original_arc]
            for scenario in scenarios
        ),
        MetricName.LEFT_TURNS: (
            weight * len(scenarios)
            if turn in counted_turns
            else 0.0
        ),
        MetricName.DISCOMFORT: sum(
            weight * scenario.discomfort[original_arc]
            for scenario in scenarios
        ),
        MetricName.HAZARD: sum(
            weight * scenario.hazard[original_arc]
            for scenario in scenarios
        ),
        MetricName.COST: sum(
            weight * scenario.cost[original_arc]
            for scenario in scenarios
        ),
        MetricName.EMISSIONS: sum(
            weight * scenario.emissions[original_arc]
            for scenario in scenarios
        ),
        MetricName.POLICING: sum(
            weight * scenario.policing[original_arc[1]]
            for scenario in scenarios
        ),
    }
    return tuple(values[metric] for metric in OBJECTIVE_VECTOR_ORDER)


def _objective_values_for_path(
    *,
    path: tuple[Arc, ...],
    scenarios: Sequence[Scenario],
    counted_turns: set[Turn],
    use_average: bool,
) -> Mapping[MetricName, float]:
    weight = _scenario_weight(scenarios=scenarios, use_average=use_average)
    left_turn_count = sum(
        1.0
        for previous_arc, next_arc in zip(path, path[1:])
        if (previous_arc[0], previous_arc[1], next_arc[1]) in counted_turns
    )
    return {
        MetricName.TRAVEL_TIME: sum(
            weight * scenario.travel_time[arc]
            for scenario in scenarios
            for arc in path
        ),
        MetricName.LEFT_TURNS: weight * len(scenarios) * left_turn_count,
        MetricName.DISCOMFORT: sum(
            weight * scenario.discomfort[arc]
            for scenario in scenarios
            for arc in path
        ),
        MetricName.HAZARD: sum(
            weight * scenario.hazard[arc]
            for scenario in scenarios
            for arc in path
        ),
        MetricName.COST: sum(
            weight * scenario.cost[arc]
            for scenario in scenarios
            for arc in path
        ),
        MetricName.EMISSIONS: sum(
            weight * scenario.emissions[arc]
            for scenario in scenarios
            for arc in path
        ),
        MetricName.POLICING: sum(
            weight * scenario.policing[arc[1]]
            for scenario in scenarios
            for arc in path
        ),
    }


def _scenario_weight(
    *,
    scenarios: Sequence[Scenario],
    use_average: bool,
) -> float:
    return 1.0 / len(scenarios) if use_average else 1.0


def _original_path_from_expanded_arc_ids(
    *,
    arc_ids: Sequence[int],
    original_arc_by_expanded_arc: Sequence[Arc | None],
) -> tuple[Arc, ...]:
    path: list[Arc] = []
    for arc_id in arc_ids:
        if arc_id < 0 or arc_id >= len(original_arc_by_expanded_arc):
            raise RuntimeError(f"C++ MDA returned unknown expanded arc id: {arc_id}.")
        original_arc = original_arc_by_expanded_arc[arc_id]
        if original_arc is not None:
            path.append(original_arc)
    return tuple(path)


def _arc_flows_for_path(path: tuple[Arc, ...]) -> Mapping[Arc, float]:
    arc_flows: dict[Arc, float] = {}
    for arc in path:
        arc_flows[arc] = arc_flows.get(arc, 0.0) + 1.0
    return arc_flows


def _mosp_model_stub(
    *,
    source: Node,
    target: Node,
    scenarios: Sequence[Scenario],
    use_average: bool,
    scale: float,
    stats: Mapping[str, Any],
) -> BenpyModel:
    return BenpyModel(
        B=np.zeros((0, 0), dtype=float),
        P=np.zeros((len(OBJECTIVE_VECTOR_ORDER), 0), dtype=float),
        b=np.zeros(0, dtype=float),
        l=np.zeros(0, dtype=float),
        s=np.zeros(0, dtype=float),
        index={"x": {}},
        meta={
            "objective_names": OBJECTIVE_VECTOR_ORDER,
            "source": source,
            "target": target,
            "scenarios": tuple(scenarios),
            "use_average": use_average,
            "solver": "mda",
            "scale": scale,
            "mosp_stats": dict(stats),
        },
    )
