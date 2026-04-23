from __future__ import annotations

from ..datastructures import (
    InfrastructureGraph,
    Node,
    Arc,
    Turn,
    World,
    Demand,
    Individual,
)


def make_toy_world() -> World:
    """Return a tiny routing instance with three competing s-t route patterns."""
    source = (0, 0)
    fast = (1, 0)
    safe = (0, 1)
    target = (1, 1)

    V: tuple[Node, ...] = (source, fast, safe, target)
    A: tuple[Arc, ...] = (
        (source, fast),
        (fast, target),
        (source, safe),
        (safe, target),
        (source, target),
    )
    L: tuple[Turn, ...] = (
        (source, fast, target),
    )

    travel_time: dict[Arc, float] = {
        (source, fast): 2.0,
        (fast, target): 2.0,
        (source, safe): 2.8,
        (safe, target): 2.8,
        (source, target): 10.3,
    }
    discomfort: dict[Arc, float] = {
        (source, fast): 1.0,
        (fast, target): 1.0,
        (source, safe): 0.5,
        (safe, target): 0.5,
        (source, target): 1.2,
    }
    hazard: dict[Arc, float] = {
        (source, fast): 1.4,
        (fast, target): 1.2,
        (source, safe): 0.3,
        (safe, target): 0.4,
        (source, target): 2.0,
    }
    cost: dict[Arc, float] = {
        (source, fast): 0.0,
        (fast, target): 0.0,
        (source, safe): 0.4,
        (safe, target): 0.4,
        (source, target): 0.9,
    }
    policing: dict[Node, float] = {
        source: 0.0,
        fast: 0.2,
        safe: 0.1,
        target: 0.3,
    }

    network = InfrastructureGraph(
        V=V,
        A=A,
        L=L,
        I={(source, safe), (safe, target)},
        nominal_travel_time=travel_time,
        nominal_discomfort=discomfort,
        nominal_hazards=hazard,
        nominal_cost=cost,
        nominal_policing=policing,
    )
    
    individuals = frozenset(
        Individual(id=f"driver_{idx}", demand=Demand(source, target))
        for idx in range(3)
    )

    return World(
        network=network,
        individuals=individuals,
    )


def _toy_base_emissions() -> dict[Arc, float]:
    source = (0, 0)
    fast = (1, 0)
    safe = (0, 1)
    target = (1, 1)

    return {
        (source, fast): 1.1,
        (fast, target): 1.1,
        (source, safe): 0.8,
        (safe, target): 0.8,
        (source, target): 1.4,
    }
