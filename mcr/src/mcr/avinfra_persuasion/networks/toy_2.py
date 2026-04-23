from __future__ import annotations

import networkx as nx
from typing import Any
from ..datastructures import (
    InfrastructureGraph,
    Arc,
    Node,
    Turn,
    World,
    Demand,
    Individual,
)
import numpy as np


def create_sample_graph(
    n_rows: int,
    n_columns: int,
    center: tuple[int, int] = (-1, -1),
    seed: int = 1,
) -> InfrastructureGraph:
    """Create a directed grid with bidirectional horizontal and vertical arcs."""
    rng = np.random.default_rng(seed)
    
    if n_rows <= 0 or n_columns <= 0:
        raise ValueError("n_rows and n_columns must be positive.")

    G = nx.DiGraph()
    grid_nodes: list[tuple[int, int]] = [
        (row, col) for row in range(n_rows) for col in range(n_columns)
    ]
    G.add_nodes_from(grid_nodes)
    center_row, center_col = center
    center_in_grid = 0 <= center_row < n_rows and 0 <= center_col < n_columns
    instrumented_edges: set[Arc] = set()
    
    def _get_attrs(is_instrumented: bool) -> dict[str, Any]:
        return {
            'travel_time': 1,
            'distance': 1,
            'capacity': 2,
            'discomfort': 0.5 if is_instrumented else 1.0,
            'hazard': float(rng.binomial(1, 0.1 if is_instrumented else 0.4)),
            'cost': 1.0 if is_instrumented else 0.0,
        }

    for row, col in grid_nodes:
        if col + 1 < n_columns:
            arc = ((row, col), (row, col + 1))
            rev_arc = ((row, col + 1), (row, col))
            is_instrumented = center_in_grid and row == center_row
            edge_attrs = _get_attrs(is_instrumented)
            if row == 1:
                edge_attrs.update({'travel_time': 0.1})
            G.add_edge(*arc, **edge_attrs)
            G.add_edge(*rev_arc, **edge_attrs)
            if is_instrumented:
                instrumented_edges.add(arc)
                instrumented_edges.add(rev_arc)

        if row + 1 < n_rows:
            arc = ((row, col), (row + 1, col))
            rev_arc = ((row + 1, col), (row, col))
            is_instrumented = center_in_grid and col == center_col
            G.add_edge(*arc, **_get_attrs(is_instrumented))
            G.add_edge(*rev_arc, **_get_attrs(is_instrumented))
            if is_instrumented:
                instrumented_edges.add(arc)
                instrumented_edges.add(rev_arc)

    arcs: set[Arc] = set(G.edges)

    def _arc_attr(key: str) -> dict[Arc, float]:
        return {(u, v): data[key] for u, v, data in G.edges(data=True)}

    def _node_policing() -> dict[Any, float]:
        policed = (
            {(center_row, col) for col in range(n_columns)}
            | {(row, center_col) for row in range(n_rows)}
        ) if center_in_grid else set()
        return {
            node: float(rng.binomial(1, 0.9 if node in policed else 0.1))
            for node in grid_nodes
        }

    return InfrastructureGraph(
        V=set(grid_nodes),
        A=arcs,
        # L=_grid_left_turns(arcs),
        I=instrumented_edges,
        nominal_travel_time=_arc_attr('travel_time'),
        nominal_link_capacity=_arc_attr('capacity'),
        arc_distances=_arc_attr('distance'),
        nominal_discomfort=_arc_attr('discomfort'),
        nominal_hazards=_arc_attr('hazard'),
        nominal_cost=_arc_attr('cost'),
        nominal_policing=_node_policing(),
    )


def _grid_left_turns(arcs: set[Arc]) -> set[Turn]:
    """Return turn triples that are left turns on a row/column grid."""
    left_turns: set[Turn] = set()
    for i, j in arcs:
        if not (_is_grid_node(i) and _is_grid_node(j)):
            continue
        for jj, k in arcs:
            if j != jj or not _is_grid_node(k):
                continue

            dr_in = j[0] - i[0]
            dc_in = j[1] - i[1]
            dr_out = k[0] - j[0]
            dc_out = k[1] - j[1]

            # Grid nodes are (row, col), while plotting uses x=col, y=-row.
            # This sign convention makes a positive cross product a left turn.
            if dr_in * dc_out - dc_in * dr_out > 0:
                left_turns.add((i, j, k))

    return left_turns


def _is_grid_node(node: Node) -> bool:
    return (
        isinstance(node, tuple)
        and len(node) == 2
        and isinstance(node[0], int)
        and isinstance(node[1], int)
    )


def make_toy_world() -> World:
    """Return a tiny routing instance with three competing s-t route patterns."""
    n_rows, n_cols = 5, 5
    center = (1, 1)
    network = create_sample_graph(n_rows=n_rows, n_columns=n_cols, center=center)

    source = (0, 0)
    target = (n_rows - 1, n_cols - 1)
    individuals = frozenset(
        Individual(id=f"driver_{idx}", demand=Demand(source, target))
        for idx in range(3)
    )

    return World(
        network=network,
        individuals=individuals,
    )
