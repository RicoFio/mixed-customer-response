from dataclasses import dataclass, field
from collections.abc import Sequence, Mapping
import itertools
import networkx as nx

from bp.world import random_grid_world, World, Scenario, Arc, Node


@dataclass
class Config:
    seed: int = 0
    rows: int = 3
    cols: int = 4

    k: int = 4
    alpha: float = .15
    beta: float = 4
    bottleneck_capacity: float = 15
    default_capacity: float = 3
    accident_delay: float = 35
    accident_prior: float = .4

    demands: Mapping[Arc, int] = field(
        default_factory=lambda: {
            ((0, 0), (0, 3)): 10,
            ((2, 0), (2, 3)): 10,
        }
    )
    bottlenecks: tuple[Arc, ...] = (
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
    )

class Instance:
    def __init__(
        self,
        config: Config=Config()
    ):
        world = self.world = random_grid_world(
            rows=config.rows,
            cols=config.cols,
            demands=config.demands,
            seed=config.seed,
        )
        network = world.network
        network.bpr_alpha = config.alpha
        network.bpr_beta = config.beta

        for arc in network.ordered_arcs:
            if arc in config.bottlenecks or (arc[1], arc[0]) in config.bottlenecks:
                network.capacity[arc] = config.bottleneck_capacity
            else:
                network.capacity[arc] = config.default_capacity

        nominal = Scenario.from_world("nominal", world)

        acc_travel_time = dict(nominal.travel_time)
        for arc in config.bottlenecks:
            acc_travel_time[arc] += config.accident_delay

        accident = Scenario(
            name="accident",
            travel_time=acc_travel_time,
            discomfort=nominal.discomfort,
            hazard=nominal.hazard,
            cost=nominal.cost,
            emissions=nominal.emissions,
            policing=nominal.policing
        )

        scenarios = self.scenarios = {
            "nominal": (nominal, 1 - config.accident_prior),
            "accident": (accident, config.accident_prior),
        }

        self.demands = config.demands
        self.bottlenecks = config.bottlenecks

        # NOTE: shortest paths calculates wrt nominal state
        paths_per_od: Mapping[Node, Sequence[Sequence[Arc]]] = {}
        self.paths_per_od = paths_per_od
        for od in config.demands.keys():
            paths_per_od[od] = [
                edge_path(p)
                for p in itertools.islice(
                    nx.shortest_simple_paths(network.graph, od[0], od[1], weight="travel_time"),
                    config.k
                )
            ]

        # tau[omega, a, k] := average cost for k players on arc a under state omega
        tau: Mapping[tuple[str, Arc, int], float] = {}
        self.tau = tau
        for scenario_name, (omega, _) in scenarios.items():
            for a in world.ordered_arcs:
                for k in range(world.total_population + 1):
                    tau[scenario_name, a, k] = omega.travel_time[a] * (1 + config.alpha * ((k - 1) / network.capacity[a]) ** config.beta)

def edge_path(path: Sequence[Node]) -> Sequence[Arc]:
    return tuple(itertools.pairwise(path))
