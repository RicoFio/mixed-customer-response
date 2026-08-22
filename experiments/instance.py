from dataclasses import dataclass, field
from collections.abc import Sequence, Mapping, Set
import itertools
import networkx as nx
import numpy as np

from bp.world import random_grid_world, Scenario, Arc, Node


def mph_to_min(mph: float):
    # minutes to cross a unit mile @ x mph
    return 60 / mph

def edge_path(path: Sequence[Node]) -> Sequence[Arc]:
    return tuple(itertools.pairwise(path))

@dataclass
class Config:
    seed: int = 0
    rows: int = 3
    cols: int = 4

    k: int = 4
    alpha: float = .15
    beta: float = 4

    # drawn uniformly
    default_mph_range: tuple[int, int] = (25, 55)
    artery_mph_range: tuple[int, int] = (65, 85)
    default_capacity_range: tuple[int, int] = (3, 10)
    artery_capacity_range: tuple[int, int] = (35, 60) # google says we can be even more aggressive, relative to default_capacity_range, but is ok
    accident_capacity_range: tuple[int, int] = (3, 10)
    accident_prior: float = .4

    demands: Mapping[Arc, int] = field(
        default_factory=lambda: {
            ((0, 0), (0, 3)): 10,
            ((2, 0), (2, 3)): 10,
        }
    )
    arteries: Set[Arc] = field(
        default_factory=lambda: set(edge_path(
            (1, x) for x in range(1)
        ))
    )

class Instance:
    def __init__(
        self,
        config: Config=Config()
    ):
        rng = np.random.default_rng(config.seed)
        world = self.world = random_grid_world(
            rows=config.rows,
            cols=config.cols,
            demands=config.demands,
            seed=config.seed,
            travel_time_range=tuple(mph_to_min(x) for x in reversed(config.default_mph_range)),
        )
        network = world.network
        network.bpr_alpha = config.alpha
        network.bpr_beta = config.beta

        edges = network.graph.edges
        for arc in edges:
            edge = edges[arc]
            if arc in config.arteries:
                edge["travel_time"] = mph_to_min(rng.uniform(*config.artery_mph_range))
                edge["capacity"] = rng.uniform(*config.artery_capacity_range)
            else:
                edge["capacity"] = rng.uniform(*config.default_capacity_range)

        travel_time = network.travel_time
        capacities = {
            "nominal": network.capacity,
            "accident": {
                arc: (
                    rng.uniform(*config.accident_capacity_range)
                    if arc in config.arteries else
                    capacity
                )
                for arc, capacity in network.capacity.items()
            },
        }

        scenarios = self.scenarios = {
            "nominal": 1 - config.accident_prior,
            "accident": config.accident_prior,
        }

        self.demands = config.demands
        self.arteries = config.arteries

        # NOTE: shortest paths calculated wrt nominal state
        paths_per_od: Mapping[Node, Sequence[Sequence[Arc]]] = {}
        self.paths_per_od = paths_per_od
        for od in config.demands.keys():
            paths = paths_per_od[od] = []
            shortest_paths = nx.shortest_simple_paths(network.graph, od[0], od[1], weight="travel_time")

            # path cannot be composed of >75% of the same edges as an existing path in the profile
            while len(paths) < config.k :
                p0 = edge_path(next(shortest_paths))

                edges = sum(1 for p1 in paths for edge in p1 if edge in p0)
                if edges <= len(p0) * .75:
                    paths.append(p0)

        active_arcs = set(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(paths_per_od.values())
            )
        )

        # tau[omega, a, k] := average cost for k players on arc a under state omega
        tau: Mapping[tuple[str, Arc, int], float] = {}
        self.tau = tau
        for scenario_name in scenarios:
            scenario_capacities = capacities[scenario_name]

            # for a in active_arcs:
            for a in world.ordered_arcs:
                for k in range(world.total_population + 1):
                    tau[scenario_name, a, k] = (0 if k == 0 else
                        travel_time[a] * (1 + config.alpha * ((k - 1) / scenario_capacities[a]) ** config.beta))

        # phi[omega, a, k] := potential over arc for k players under state omega
        phi: Mapping[tuple[str, Arc, float]] = {}
        self.phi = phi
        for scenario_name in scenarios:
            for arc in active_arcs:
                potential = 0
                for k in range(world.total_population + 1):
                    potential += tau[scenario_name, arc, k]
                    phi[scenario_name, arc, k] = potential
