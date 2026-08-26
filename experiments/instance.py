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
    rows: int = 5
    cols: int = 4

    k: int = 4
    alpha: float = .15
    beta: float = 4

    # drawn uniformly
    default_mph_range: tuple[int, int] = (20, 40)
    artery_mph_range: tuple[int, int] = (65, 85)
    default_capacity_range: tuple[int, int] = (3, 3)
    artery_capacity_range: tuple[int, int] = (35, 60) # google says we can be even more aggressive, relative to default_capacity_range, but is ok
    accident_capacity_range: tuple[int, int] = (3, 3)
    accident_prior: float = .4

    demands: Mapping[Arc, int] = field(
        default_factory=lambda: {
            ((1, 0), (1, 3)): 10,
            ((3, 0), (3, 3)): 10,
        }
    )
    arteries: Set[Arc] = field(
        default_factory=lambda: set(edge_path(
            (2, x) for x in range(4)
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
        capacities = self.capacities = {
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

        self.scenarios = {
            "nominal": 1 - config.accident_prior,
            "accident": config.accident_prior,
        }

        self.demands = config.demands
        self.arteries = config.arteries

        # NOTE: shortest paths calculated wrt nominal state
        self.paths_per_od: Mapping[Node, Sequence[Sequence[Arc]]] = {}
        for od in config.demands.keys():
            paths = self.paths_per_od[od] = []
            shortest_paths = nx.shortest_simple_paths(network.graph, od[0], od[1], weight="travel_time")

            # path cannot be composed of >75% of the same edges as an existing path in the profile
            while len(paths) < config.k :
                p0 = edge_path(next(shortest_paths))

                edges = sum(1 for p1 in paths for edge in p1 if edge in p0)
                if edges <= len(p0) * .75:
                    paths.append(p0)

        active_arcs = self.active_arcs = set(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(self.paths_per_od.values())
            )
        )

        self.arc_to_idx = {a: i for i, a in enumerate(self.active_arcs)}

        num_arcs = len(self.active_arcs)
        max_flow = world.total_population + 1 # +1 b/c deviations
        k_range = np.arange(1, max_flow + 1)

        self.tau: Mapping[str, np.ndarray] = {}
        self.phi: Mapping[str, np.ndarray] = {}

        for scenario_name, scenario_capacities in capacities.items():
            # +1 b/c [0, max_flow]
            tau_matrix = np.zeros((num_arcs, max_flow + 1))
            phi_matrix = np.zeros((num_arcs, max_flow + 1))

            for i, arc in enumerate(active_arcs):
                t0 = travel_time[arc]
                c = scenario_capacities[arc]
                costs = t0 * (1 + config.alpha * ((k_range - 1) / c) ** config.beta)
                tau_matrix[i, 1:] = costs
                phi_matrix[i, 1:] = np.cumsum(costs)

            self.tau[scenario_name] = tau_matrix
            self.phi[scenario_name] = phi_matrix
