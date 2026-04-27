from __future__ import annotations

import networkx as nx
from typing import Any
from ..datastructures import (
    InfrastructureGraph,
    Arc,
    World,
    Demand,
    Individual,
)


def create_sample_graph(instrumented: str = "tlbr") -> InfrastructureGraph:
    """Create a directed grid with bidirectional horizontal and vertical arcs."""

    G = nx.DiGraph()
    grid_nodes: list[tuple[int, int]] = [
        (row, col) for row in range(2) for col in range(2)
    ]
    G.add_nodes_from(grid_nodes)
    
    instrumented_edge_list: list[Arc] = []
    
    if "t" in instrumented:
        instrumented_edge_list.append(((0, 0), (0, 1))) 
    if "r" in instrumented:
        instrumented_edge_list.append(((0, 1), (1, 1))) 
    if "l" in instrumented:
        instrumented_edge_list.append(((0, 0), (1, 0))) 
    if "b" in instrumented:
        instrumented_edge_list.append(((1, 0), (1, 1)))
        
    instrumented_edges: set[Arc] = set(instrumented_edge_list)

    # Right
    G.add_edge((0, 0), (0, 1), travel_time=0.5, distance=1, capacity=4, discomfort=0.5, hazard=1, cost=0.5)
    # Down
    G.add_edge((0, 0), (1, 0), travel_time=1.5, distance=1, capacity=4, discomfort=1, hazard=0, cost=1)
    # Right Down
    G.add_edge((0, 1), (1, 1), travel_time=0.5, distance=1, capacity=4, discomfort=1, hazard=2, cost=0.5)
    # Down Left
    G.add_edge((1, 0), (1, 1), travel_time=1.5, distance=1, capacity=4, discomfort=1, hazard=0, cost=1)

    arcs: set[Arc] = set(G.edges)

    def _arc_attr(key: str) -> dict[Arc, float]:
        return {(u, v): data[key] for u, v, data in G.edges(data=True)}

    return InfrastructureGraph(
        V=set(grid_nodes),
        A=arcs,
        L={
            ((0, 0), (1, 0), (1, 1))
        },
        I=instrumented_edges,
        nominal_travel_time=_arc_attr('travel_time'),
        nominal_link_capacity=_arc_attr('capacity'),
        arc_distances=_arc_attr('distance'),
        nominal_discomfort=_arc_attr('discomfort'),
        nominal_hazards=_arc_attr('hazard'),
        nominal_cost=_arc_attr('cost'),
        nominal_policing={
            node: float(node in {(1, 0), (1, 1)})
            for node in grid_nodes
        },
    )
