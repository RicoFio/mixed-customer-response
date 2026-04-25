from __future__ import annotations

import networkx as nx
from ..datastructures import InfrastructureGraph


def create_sample_graph() -> InfrastructureGraph:
    """Minimal two-route network: a fast path and a slow path from O to D.

    Because Arc is identified solely by (source, destination), true parallel
    arcs between the same pair of nodes are not supported — they would be
    the same key in the arc set/maps.  A diamond topology is used instead:

        O --[1.0]--> F --[1.0]--> D   (fast path, total travel time = 2.0)
        O --[5.0]--> S --[5.0]--> D   (slow path, total travel time = 10.0)

    The fast-path arcs are marked as instrumented (I), meaning the sender
    can observe and report on their congestion state.
    """
    G = nx.DiGraph()

    # Fast path — instrumented so the sender can signal on these arcs
    G.add_edge("O", "F", travel_time=1.0, instrumented=True)
    G.add_edge("F", "D", travel_time=1.0, instrumented=True)

    # Slow path — not instrumented
    G.add_edge("O", "S", travel_time=5.0)
    G.add_edge("S", "D", travel_time=5.0)

    return InfrastructureGraph.from_networkx(G)
