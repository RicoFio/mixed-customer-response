from __future__ import annotations

from collections.abc import Sequence
import math

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .datastructures import (
    InfrastructureGraph,
    Node,
    Arc,
    Turn,
    World,
    BenpyModel,
    Demand,
    Individual,
    Scenario,
)

from .opt import (
    RoutingSolution,
    RoutingSolutionPoint,
    make_independent_world_belief,
    build_benpy_model_sample_average,
)
from .plotting import (
    plot_infrastructure,
    plot_pareto_frontier,
    plot_scenario,
    plot_world,
)


DEFAULT_SOLUTION_METRICS = (
    "travel_time",
    "discomfort",
    "hazard",
    "cost",
    "emissions",
    "policing",
)


def create_sample_graph(
    n_rows: int,
    n_columns: int,
    n_population: int = 0,
    center: tuple[int, int] = (-1, -1),
) -> InfrastructureGraph:
    """Create a directed grid with eastbound and southbound arcs."""
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

    for row, col in grid_nodes:
        if col + 1 < n_columns:
            arc = ((row, col), (row, col + 1))
            G.add_edge(*arc)
            if center_in_grid and row == center_row and col < center_col:
                instrumented_edges.add(arc)

        if row + 1 < n_rows:
            arc = ((row, col), (row + 1, col))
            G.add_edge(*arc)
            if center_in_grid and col == center_col and row < center_row:
                instrumented_edges.add(arc)

    arcs: set[Arc] = set(G.edges)

    return InfrastructureGraph(
        V=set(grid_nodes),
        A=arcs,
        I=instrumented_edges,
        nominal_travel_time={arc: 5.0 for arc in arcs},
        nominal_discomfort={arc: 1.0 for arc in arcs},
        nominal_hazards={arc: 0.0 for arc in arcs},
        nominal_cost={arc: 0.0 for arc in arcs},
        nominal_policing={node: 0.0 for node in grid_nodes},
    )


def make_toy_world() -> World:
    """Return a tiny routing instance with three competing s-t route patterns."""
    source = "source"
    fast = "fast"
    safe = "safe"
    target = "target"

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
        (source, safe, target),
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
    source = "source"
    fast = "fast"
    safe = "safe"
    target = "target"

    return {
        (source, fast): 1.1,
        (fast, target): 1.1,
        (source, safe): 0.8,
        (safe, target): 0.8,
        (source, target): 1.4,
    }


def sample_toy_scenarios(
    n_samples: int = 5,
    seed: int | None = 1,
    rel_noise: float = 0.05,
) -> tuple[World, list[Scenario]]:
    """Sample toy scenarios separately from BenPy model construction."""
    world = make_toy_world()
    V, A, _ = _ordered_toy_components(world)

    belief = make_independent_world_belief(
        V=V,
        A=A,
        base_t=world.travel_time,
        base_discomfort=world.discomfort,
        base_h=world.hazard,
        base_c=world.cost,
        base_e=_toy_base_emissions(),
        base_p=world.policing,
        rel_noise=rel_noise,
    )
    return world, belief.sample(n_samples=n_samples, seed=seed)


def build_toy_benpy_model(
    n_samples: int = 5,
    seed: int | None = 1,
    rel_noise: float = 0.05,
) -> BenpyModel:
    """Build the toy network model used for testing the benpy setup."""
    world, scenarios = sample_toy_scenarios(
        n_samples=n_samples,
        seed=seed,
        rel_noise=rel_noise,
    )
    return build_toy_benpy_model_from_scenarios(world=world, scenarios=scenarios)


def build_toy_benpy_model_from_scenarios(
    world: World,
    scenarios: Sequence[Scenario],
) -> BenpyModel:
    """Build the toy BenPy model for already-realized scenarios."""
    source, target = _single_toy_demand(world)
    V = tuple(sorted(world.V, key=str))
    A = tuple(sorted(world.A, key=str))
    L = tuple(sorted(world.L, key=str))

    return build_benpy_model_sample_average(
        V=V,
        A=A,
        L=L,
        s=source,
        t=target,
        scenarios=scenarios,
        use_average=True,
    )


def solve_toy_network(
    n_samples: int = 5,
    seed: int | None = 1,
    rel_noise: float = 0.05,
) -> RoutingSolution:
    """Solve the toy network VLP and return a readable solution wrapper."""
    model = build_toy_benpy_model(
        n_samples=n_samples,
        seed=seed,
        rel_noise=rel_noise,
    )
    raw_solution = model.solve(options={"solution": True})
    return RoutingSolution.from_benpy_solution(
        raw_solution=raw_solution,
        model=model,
    )


def solve_toy_single_scenario(
    seed: int | None = 1,
    rel_noise: float = 0.05,
    selection_metric: str = "travel_time",
) -> tuple[World, Scenario, RoutingSolution, RoutingSolutionPoint]:
    """
    Solve one sampled toy scenario and return the selected route solution.

    BenPy returns all efficient vertices. For a single route overlay, we select the
    bounded efficient vertex minimizing ``selection_metric``.
    """
    world, scenarios = sample_toy_scenarios(
        n_samples=1,
        seed=seed,
        rel_noise=rel_noise,
    )
    scenario = scenarios[0]
    model = build_toy_benpy_model_from_scenarios(world=world, scenarios=scenarios)
    raw_solution = model.solve(options={"solution": True})
    solution = RoutingSolution.from_benpy_solution(
        raw_solution=raw_solution,
        model=model,
    )
    selected_point = solution.best_by_metric(selection_metric)
    return world, scenario, solution, selected_point


# def plot_toy_setup(seed: int | None = 1, rel_noise: float = 0.05) -> Figure:
#     """Plot infrastructure, nominal world travel times, and one sampled realization."""
#     world, scenarios = sample_toy_scenarios(
#         n_samples=1,
#         seed=seed,
#         rel_noise=rel_noise,
#     )
#     scenario = scenarios[0]
#     paths = {
#         "driver_0": ["source", "fast", "target"],
#         "driver_1": ["source", "safe", "target"],
#         "driver_2": ["source", "target"],
#     }

#     fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
#     plot_infrastructure(world.network, ax=axes[0])
#     plot_world(world, arc_metric="travel_time", ax=axes[1])
#     plot_scenario(
#         world,
#         scenario,
#         paths=paths,
#         arc_metric="travel_time",
#         node_metric="policing",
#         ax=axes[2],
#     )
#     return fig


def plot_toy_single_scenario_solution(
    metrics: Sequence[str] = DEFAULT_SOLUTION_METRICS,
    seed: int | None = 1,
    rel_noise: float = 0.05,
    selection_metric: str = "travel_time",
) -> Figure:
    """Plot one BenPy-selected toy route over the same scenario metric panels."""
    if not metrics:
        raise ValueError("At least one metric is required.")

    world, scenario, _, selected_point = solve_toy_single_scenario(
        seed=seed,
        rel_noise=rel_noise,
        selection_metric=selection_metric,
    )
    paths = {
        individual: selected_point.path
        for individual in sorted(world.individuals, key=lambda item: item.id)
    }

    n_cols = min(3, len(metrics))
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    axes_list = list(np.asarray(axes).ravel())

    for ax, metric in zip(axes_list, metrics):
        arc_metric, node_metric = _metric_plot_fields(metric)
        plot_scenario(
            world,
            scenario,
            paths=paths,
            arc_metric=arc_metric,
            node_metric=node_metric,
            ax=ax,
        )

    for ax in axes_list[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f"Single-scenario BenPy solution by {selection_metric}")
    return fig


def plot_toy_pareto_frontier(
    x_metric: str = "discomfort",
    y_metric: str = "travel_time",
    n_samples: int = 1,
    seed: int | None = 1,
    rel_noise: float = 0.05,
) -> Figure:
    """Solve the toy model and plot a two-objective Pareto projection."""
    solution = solve_toy_network(
        n_samples=n_samples,
        seed=seed,
        rel_noise=rel_noise,
    )
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_pareto_frontier(
        solution,
        x_metric=x_metric,
        y_metric=y_metric,
        objective_names=solution.objective_names,
        ax=ax,
    )
    return fig


def _ordered_toy_components(
    world: World,
) -> tuple[tuple[Node, ...], tuple[Arc, ...], tuple[Turn, ...]]:
    return (
        tuple(sorted(world.V, key=str)),
        tuple(sorted(world.A, key=str)),
        tuple(sorted(world.L, key=str)),
    )


def _single_toy_demand(world: World) -> tuple[Node, Node]:
    demands = {individual.demand for individual in world.individuals}
    if len(demands) != 1:
        raise ValueError("The toy BenPy builder expects exactly one demand pair.")

    demand = next(iter(demands))
    return demand.origin, demand.destination


def _metric_plot_fields(metric: str) -> tuple[str | None, str | None]:
    if metric == "policing":
        return None, "policing"
    if metric in {"travel_time", "discomfort", "hazard", "cost", "emissions"}:
        return metric, None
    raise ValueError(
        "Toy solution plots support arc metrics travel_time, discomfort, hazard, "
        "cost, emissions, and node metric policing."
    )


if __name__ == "__main__":
    solution = solve_toy_network()
    model = solution.model
    print(f"status: {solution.status}")
    print(f"objectives: {', '.join(solution.objective_names)}")
    print(f"constraints: {model.B.shape[0]}")
    print(f"variables: {model.P.shape[1]}")
    print(f"upper image vertices: {solution.num_vertices_upper}")
    print(f"efficient route solutions: {len(solution.points)}")
    # plot_toy_setup()
    # plot_toy_single_scenario_solution()
    plot_toy_pareto_frontier(
        x_metric = "discomfort",
        y_metric = "travel_time",
        n_samples = 1,
        seed = 1,
        rel_noise = 0,
    )
    plt.show()
