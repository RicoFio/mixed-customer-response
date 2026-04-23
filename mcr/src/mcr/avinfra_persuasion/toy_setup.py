from __future__ import annotations

from collections.abc import Sequence
import math

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .datastructures import (
    InfrastructureGraph,
    MetricName,
    Node,
    Arc,
    Turn,
    World,
    BenpyModel,
    Demand,
    Individual,
    Scenario,
)

from typing import Callable

from .opt import (
    RoutingSolution,
    RoutingSolutionPoint,
    make_independent_world_belief,
    build_benpy_model_sample_average,
    build_turn_state_benpy_model_sample_average,
)
from .plotting import (
    plot_infrastructure,
    plot_pareto_frontier,
    plot_scenario,
    plot_world,
)

from .networks.toy_1 import make_toy_world as make_toy1_world
from .networks.toy_2 import make_toy_world as make_toy2_world


DEFAULT_SOLUTION_METRICS = (
    MetricName.TRAVEL_TIME,
    MetricName.DISCOMFORT,
    MetricName.HAZARD,
    MetricName.COST,
    MetricName.EMISSIONS,
    MetricName.POLICING,
)


def sample_toy_scenarios(
    world: World,
    n_samples: int = 5,
    seed: int | None = 1,
    rel_noise: float = 0.05,
) -> tuple[World, list[Scenario]]:
    """Sample toy scenarios separately from BenPy model construction."""
    belief = make_independent_world_belief(
        V=world.ordered_V,
        A=world.ordered_A,
        base_t=world.travel_time,
        base_discomfort=world.discomfort,
        base_h=world.hazard,
        base_c=world.cost,
        base_e={a: 1 for a in world.ordered_A},
        base_p=world.policing,
        rel_noise=rel_noise,
    )
    return world, belief.sample(n_samples=n_samples, seed=seed)


def build_toy_benpy_model_from_scenarios(
    world: World,
    scenarios: Sequence[Scenario],
    use_turn_state: bool = True,
) -> BenpyModel:
    """Build the toy BenPy model for already-realized scenarios."""
    def _single_toy_demand(world: World) -> tuple[Node, Node]:
        demands = {individual.demand for individual in world.individuals}
        if len(demands) != 1:
            raise ValueError("The toy BenPy builder expects exactly one demand pair.")

        demand = next(iter(demands))
        return demand.origin, demand.destination
    
    source, target = _single_toy_demand(world)
    builder = (
        build_turn_state_benpy_model_sample_average
        if use_turn_state
        else build_benpy_model_sample_average
    )

    return builder(
        V=world.ordered_V,
        A=world.ordered_A,
        L=world.ordered_L,
        s=source,
        t=target,
        scenarios=scenarios,
        use_average=True,
    )


def solve_toy_network(
    world: World,
    n_samples: int = 5,
    seed: int | None = 1,
    rel_noise: float = 0.05,
    use_turn_state: bool = True,
) -> tuple[list[Scenario], RoutingSolution]:
    """Solve the toy network VLP and return a readable solution wrapper."""
    
    world, scenarios = sample_toy_scenarios(
        world=world,
        n_samples=n_samples,
        seed=seed,
        rel_noise=rel_noise,
    )
    model = build_toy_benpy_model_from_scenarios(
        world=world,
        scenarios=scenarios,
        use_turn_state=use_turn_state,
    )
    raw_solution = model.solve(options={"solution": True})
    return scenarios, RoutingSolution.from_benpy_solution(
        raw_solution=raw_solution,
        model=model,
    )

def run_toy_pareto_frontier(
    world: World,
    x_metric: MetricName = MetricName.DISCOMFORT,
    y_metric: MetricName = MetricName.TRAVEL_TIME,
    n_samples: int = 1,
    seed: int | None = 1,
    rel_noise: float = 0.05,
    use_turn_state: bool = True,
) -> None:
    """Solve the toy model and plot a two-objective Pareto projection."""
    scenarios, solution = solve_toy_network(
        world=world,
        n_samples=n_samples,
        seed=seed,
        rel_noise=rel_noise,
        use_turn_state=use_turn_state,
    )
    model = solution.model
    print(f"status: {solution.status}")
    print(f"objectives: {', '.join(solution.objective_names)}")
    print(f"constraints: {model.B.shape[0]}")
    print(f"variables: {model.P.shape[1]}")
    print(f"upper image vertices: {solution.num_vertices_upper}")
    print(f"efficient route solutions: {len(solution.points)}")
    
    _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_infrastructure(
        world.network,
        ax=ax,
    )
    
    plot_scenario(
        world=world,
        scenario=scenarios[0],
        paths=solution.paths,
    )
    
    plt.show()
    
    _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_pareto_frontier(
        solution,
        x_metric=x_metric,
        y_metric=y_metric,
        objective_names=solution.objective_names,
        ax=ax,
    )
    plt.show()





if __name__ == "__main__":
    world = make_toy2_world()
    run_toy_pareto_frontier(
        world=world,
        x_metric = MetricName.DISCOMFORT,
        y_metric = MetricName.TRAVEL_TIME,
        n_samples = 10,
        seed = 1,
        rel_noise = 5,
    )
