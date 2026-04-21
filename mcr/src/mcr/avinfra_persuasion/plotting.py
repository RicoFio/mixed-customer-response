from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .datastructures import Arc, Individual, InfrastructureGraph, Node, Scenario, World


ArcMetric = str | Mapping[Arc, float] | None
NodeMetric = str | Mapping[Node, float] | None
PathByIndividual = Mapping[str | Individual, Sequence[Node] | Sequence[Arc]]
PositionMap = Mapping[Node, tuple[float, float]]
ObjectiveSelector = int | str


def plot_infrastructure(
    network: InfrastructureGraph,
    ax: Axes | None = None,
    pos: PositionMap | None = None,
    show_labels: bool = True,
) -> Axes:
    """Plot only the physical infrastructure graph."""
    ax = _get_ax(ax)
    graph = _to_digraph(network)
    positions = _node_positions(graph, pos)

    regular_arcs = [arc for arc in graph.edges if arc not in network.I]
    instrumented_arcs = [arc for arc in graph.edges if arc in network.I]

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color="#f7f7f7",
        edgecolors="#242424",
        node_size=520,
        linewidths=1.2,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edgelist=regular_arcs,
        edge_color="#777777",
        arrows=True,
        arrowsize=16,
        width=1.6,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edgelist=instrumented_arcs,
        edge_color="#1f77b4",
        arrows=True,
        arrowsize=18,
        width=3.0,
    )
    if show_labels:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9)

    _finish_network_ax(ax, "Infrastructure")
    return ax


def plot_world(
    world: World,
    arc_metric: ArcMetric = None,
    node_metric: NodeMetric = None,
    ax: Axes | None = None,
    pos: PositionMap | None = None,
    show_labels: bool = True,
) -> Axes:
    """Plot infrastructure with optional nominal arc and node metrics."""
    ax = _get_ax(ax)
    graph = _to_digraph(world.network)
    positions = _node_positions(graph, pos)
    arc_values = _resolve_world_arc_metric(world, arc_metric)
    node_values = _resolve_world_node_metric(world, node_metric)

    _draw_metric_network(
        graph=graph,
        positions=positions,
        ax=ax,
        arc_values=arc_values,
        node_values=node_values,
        instrumented_arcs=world.I,
        show_labels=show_labels,
    )

    title_parts = ["World"]
    if isinstance(arc_metric, str):
        title_parts.append(f"arc: {arc_metric}")
    if isinstance(node_metric, str):
        title_parts.append(f"node: {node_metric}")
    _finish_network_ax(ax, " | ".join(title_parts))
    return ax


def plot_scenario(
    world: World,
    scenario: Scenario,
    paths: PathByIndividual,
    arc_metric: ArcMetric = "travel_time",
    node_metric: NodeMetric = "policing",
    ax: Axes | None = None,
    pos: PositionMap | None = None,
    show_labels: bool = True,
) -> Axes:
    """Plot a sampled scenario and overlay each individual's chosen path."""
    ax = _get_ax(ax)
    graph = _to_digraph(world.network)
    positions = _node_positions(graph, pos)
    arc_values = _resolve_scenario_arc_metric(scenario, arc_metric)
    node_values = _resolve_scenario_node_metric(scenario, node_metric)

    _draw_metric_network(
        graph=graph,
        positions=positions,
        ax=ax,
        arc_values=arc_values,
        node_values=node_values,
        instrumented_arcs=world.I,
        show_labels=show_labels,
    )
    _draw_paths(
        ax=ax,
        graph=graph,
        positions=positions,
        paths=paths,
    )

    title_parts = [f"Scenario: {scenario.name}"]
    if isinstance(arc_metric, str):
        title_parts.append(f"arc: {arc_metric}")
    if isinstance(node_metric, str):
        title_parts.append(f"node: {node_metric}")
    _finish_network_ax(ax, " | ".join(title_parts))
    return ax


def plot_network(
    network: InfrastructureGraph,
    ax: Axes | None = None,
    pos: PositionMap | None = None,
) -> Axes:
    """Backward-compatible alias for plot_infrastructure."""
    return plot_infrastructure(network=network, ax=ax, pos=pos)


def plot_dynamic_game(*args, **kwargs) -> Axes:
    raise NotImplementedError("Dynamic game plotting is not implemented yet.")


def plot_pareto_frontier(
    points: Any,
    x_metric: ObjectiveSelector,
    y_metric: ObjectiveSelector,
    objective_names: Sequence[str] | None = None,
    ax: Axes | None = None,
    connect: bool = True,
    annotate: bool = True,
    include_unbounded: bool = False,
) -> Axes:
    """
    Plot a two-dimensional projection of generated Pareto points.

    ``points`` can be either an array-like object of shape
    ``(n_points, n_objectives)`` or a BenPy solution with
    ``solution.Primal.vertex_value``. Objectives are treated as quantities to
    minimize, so the frontier is built from the nondominated points in the
    selected two dimensions.
    """
    ax = _get_ax(ax)
    point_array = _pareto_point_array(points)
    point_array = _filter_bounded_points(
        point_array=point_array,
        points=points,
        include_unbounded=include_unbounded,
    )
    x_idx = _objective_index(x_metric, objective_names, point_array.shape[1])
    y_idx = _objective_index(y_metric, objective_names, point_array.shape[1])
    if x_idx == y_idx:
        raise ValueError("x_metric and y_metric must refer to different objectives.")

    xy = point_array[:, [x_idx, y_idx]]
    point_labels = _pareto_point_labels(points, len(point_array))
    nondominated_mask = _nondominated_minimization_mask(xy)
    dominated_xy = xy[~nondominated_mask]
    frontier_xy = xy[nondominated_mask]
    frontier_order = np.argsort(frontier_xy[:, 0])
    frontier_xy = frontier_xy[frontier_order]

    if len(dominated_xy) > 0:
        ax.scatter(
            dominated_xy[:, 0],
            dominated_xy[:, 1],
            s=42,
            color="#ff7f0e",
            edgecolor="#202020",
            linewidth=0.7,
            alpha=0.5,
            zorder=3,
            label="dominated",
        )

    if len(frontier_xy) > 0:
        ax.scatter(
            xy[nondominated_mask, 0],
            xy[nondominated_mask, 1],
            s=46,
            color="#ff7f0e",
            edgecolor="#202020",
            linewidth=0.8,
            alpha=1.0,
            zorder=4,
            label="nondominated",
        )

    ax.margins(x=0.08, y=0.08)
    if connect and len(frontier_xy) > 0:
        _draw_axis_connected_frontier(ax=ax, frontier_xy=frontier_xy)
        # _draw_axis_aligned_frontier(ax=ax, frontier_xy=frontier_xy)

    if annotate:
        optimum_labels = _metric_optimum_labels(
            point_array=point_array,
            metric_indices=(x_idx, y_idx),
            objective_names=objective_names,
        )
        for point_idx, (x_value, y_value) in enumerate(xy):
            label = point_labels[point_idx]
            if optimum_labels[point_idx]:
                label = f"{label}: {', '.join(optimum_labels[point_idx])}"
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    x_label = _objective_label(x_metric, objective_names)
    y_label = _objective_label(y_metric, objective_names)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Pareto frontier: {x_label} vs {y_label}")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax


def _pareto_point_labels(points: Any, n_points: int) -> tuple[str, ...]:
    raw_labels = getattr(points, "labels", None)
    if raw_labels is not None and len(raw_labels) == n_points:
        return tuple(str(label) for label in raw_labels)
    return tuple(_solution_label(point_idx) for point_idx in range(n_points))


def _draw_axis_connected_frontier(ax: Axes, frontier_xy: np.ndarray) -> None:
    """"""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_right = x_limits[1]
    y_top = y_limits[1]
    
    ax.plot(
        frontier_xy[:, 0],
        frontier_xy[:, 1],
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
        label="frontier",
    )
    # Plot outer frontier
    ax.plot(
        [frontier_xy[0, 0], frontier_xy[0, 0]],
        [y_top, frontier_xy[0, 1]],
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
        label="_nolegend_",
    )
    ax.plot(
        [frontier_xy[-1, 0], x_right],
        [frontier_xy[-1, 1], frontier_xy[-1, 1]],
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
        label="_nolegend_",
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    
    

def _draw_axis_aligned_frontier(ax: Axes, frontier_xy: np.ndarray) -> None:
    """Draw a minimization frontier with horizontal and vertical segments."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_right = x_limits[1]
    y_top = y_limits[1]

    step_x = [float(frontier_xy[0, 0]), float(frontier_xy[0, 0])]
    step_y = [float(y_top), float(frontier_xy[0, 1])]

    current_y = float(frontier_xy[0, 1])
    for x_value, y_value in frontier_xy[1:]:
        x_float = float(x_value)
        y_float = float(y_value)
        step_x.extend([x_float, x_float])
        step_y.extend([current_y, y_float])
        current_y = y_float

    step_x.append(float(x_right))
    step_y.append(current_y)

    ax.plot(
        step_x,
        step_y,
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
        label="frontier",
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)


def _nondominated_minimization_mask(values: np.ndarray) -> np.ndarray:
    """Return points not dominated by another point under minimization."""
    mask = np.ones(values.shape[0], dtype=bool)
    for point_idx, point in enumerate(values):
        no_worse = np.all(values <= point, axis=1)
        strictly_better = np.any(values < point, axis=1)
        mask[point_idx] = not bool(np.any(no_worse & strictly_better))
    return mask


def _metric_optimum_labels(
    point_array: np.ndarray,
    metric_indices: Sequence[int],
    objective_names: Sequence[str] | None,
) -> list[list[str]]:
    labels: list[list[str]] = [[] for _ in range(point_array.shape[0])]
    for metric_idx in metric_indices:
        metric_values = point_array[:, metric_idx]
        best_value = float(np.min(metric_values))
        metric_label = _objective_label(metric_idx, objective_names)
        for point_idx in np.flatnonzero(np.isclose(metric_values, best_value)):
            labels[point_idx].append(metric_label)
    return labels


def _solution_label(point_idx: int) -> str:
    if point_idx < 26:
        return chr(ord("A") + point_idx)
    return str(point_idx + 1)


def _pareto_point_array(points: Any) -> np.ndarray:
    primal = getattr(points, "Primal", None)
    raw_points = getattr(primal, "vertex_value", None) if primal is not None else None
    if raw_points is None:
        raw_points = getattr(points, "vertex_value", None)
    if raw_points is None:
        raw_points = points

    point_array = np.asarray(raw_points, dtype=float)
    if point_array.ndim != 2:
        raise ValueError("Pareto points must have shape (n_points, n_objectives).")
    if point_array.shape[0] == 0:
        raise ValueError("At least one Pareto point is required.")
    if point_array.shape[1] < 2:
        raise ValueError("Pareto points must have at least two objectives.")
    return point_array


def _filter_bounded_points(
    point_array: np.ndarray,
    points: Any,
    include_unbounded: bool,
) -> np.ndarray:
    if include_unbounded:
        return point_array

    primal = getattr(points, "Primal", None)
    raw_vertex_types = (
        getattr(primal, "vertex_type", None) if primal is not None else None
    )
    if raw_vertex_types is None:
        raw_vertex_types = getattr(points, "vertex_type", None)
    if raw_vertex_types is None:
        return point_array

    vertex_types = np.asarray(raw_vertex_types)
    if vertex_types.shape != (point_array.shape[0],):
        return point_array

    bounded_points = point_array[vertex_types == 1]
    return bounded_points if len(bounded_points) > 0 else point_array


def _objective_index(
    selector: ObjectiveSelector,
    objective_names: Sequence[str] | None,
    n_objectives: int,
) -> int:
    if isinstance(selector, bool):
        raise ValueError("Objective selectors must be metric names or integer indices.")
    if isinstance(selector, int):
        if selector < 0 or selector >= n_objectives:
            raise ValueError(
                f"Objective index {selector} is outside [0, {n_objectives})."
            )
        return selector

    if objective_names is None:
        raise ValueError("objective_names is required when selecting metrics by name.")
    if selector not in objective_names:
        raise ValueError(f"Unknown objective metric: {selector!r}.")
    return list(objective_names).index(selector)


def _objective_label(
    selector: ObjectiveSelector,
    objective_names: Sequence[str] | None,
) -> str:
    if (
        isinstance(selector, int)
        and objective_names is not None
        and 0 <= selector < len(objective_names)
    ):
        return objective_names[selector]
    return str(selector)


def _to_digraph(network: InfrastructureGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(network.V)
    graph.add_edges_from(network.A)
    return graph


def _node_positions(
    graph: nx.DiGraph,
    pos: PositionMap | None,
) -> dict[Node, tuple[float, float]]:
    if pos is not None:
        return dict(pos)

    nodes = set(graph.nodes)
    if all(_is_grid_node(node) for node in nodes):
        return {
            node: (float(node[1]), -float(node[0]))
            for node in nodes
            if isinstance(node, tuple)
        }

    return nx.spring_layout(graph.to_undirected(), seed=7)


def _is_grid_node(node: Node) -> bool:
    return (
        isinstance(node, tuple)
        and len(node) == 2
        and all(isinstance(value, int | float) for value in node)
    )


def _resolve_world_arc_metric(world: World, metric: ArcMetric) -> Mapping[Arc, float] | None:
    if metric is None or isinstance(metric, Mapping):
        return metric
    if metric == "travel_time":
        return world.travel_time
    if metric == "discomfort":
        return world.discomfort
    if metric == "hazard":
        return world.hazard
    if metric == "cost":
        return world.cost
    raise ValueError(f"World arc metric {metric!r} is not available.")


def _resolve_world_node_metric(
    world: World,
    metric: NodeMetric,
) -> Mapping[Node, float] | None:
    if metric is None or isinstance(metric, Mapping):
        return metric
    if metric == "policing":
        return world.policing
    if metric == "population":
        return {node: float(world.population_at_node(node)) for node in world.V}
    raise ValueError(f"World node metric {metric!r} is not available.")


def _resolve_scenario_arc_metric(
    scenario: Scenario,
    metric: ArcMetric,
) -> Mapping[Arc, float] | None:
    if metric is None or isinstance(metric, Mapping):
        return metric
    metric_map = {
        "travel_time": scenario.travel_time,
        "discomfort": scenario.discomfort,
        "hazard": scenario.hazard,
        "cost": scenario.cost,
        "emissions": scenario.emissions,
    }
    if metric not in metric_map:
        raise ValueError(f"Scenario arc metric {metric!r} is not available.")
    return metric_map[metric]


def _resolve_scenario_node_metric(
    scenario: Scenario,
    metric: NodeMetric,
) -> Mapping[Node, float] | None:
    if metric is None or isinstance(metric, Mapping):
        return metric
    if metric == "policing":
        return scenario.policing
    raise ValueError(f"Scenario node metric {metric!r} is not available.")


def _draw_metric_network(
    graph: nx.DiGraph,
    positions: Mapping[Node, tuple[float, float]],
    ax: Axes,
    arc_values: Mapping[Arc, float] | None,
    node_values: Mapping[Node, float] | None,
    instrumented_arcs: set[Arc],
    show_labels: bool,
) -> None:
    edges = list(graph.edges)
    edge_widths = [3.0 if edge in instrumented_arcs else 1.6 for edge in edges]

    if arc_values is None:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edgelist=edges,
            edge_color="#777777",
            arrows=True,
            arrowsize=16,
            width=edge_widths,
        )
    else:
        edge_colors = [arc_values[edge] for edge in edges]
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edgelist=edges,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
            arrows=True,
            arrowsize=16,
            width=edge_widths,
        )
        _add_metric_colorbar(ax, edge_colors, plt.cm.viridis, "arc metric")

    if node_values is None:
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color="#f7f7f7",
            edgecolors="#242424",
            node_size=520,
            linewidths=1.2,
        )
    else:
        node_colors = [node_values[node] for node in graph.nodes]
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color=node_colors,
            cmap=plt.cm.plasma,
            edgecolors="#242424",
            node_size=560,
            linewidths=1.2,
        )
        _add_metric_colorbar(ax, node_colors, plt.cm.plasma, "node metric")

    if show_labels:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9)


def _draw_paths(
    ax: Axes,
    graph: nx.DiGraph,
    positions: Mapping[Node, tuple[float, float]],
    paths: PathByIndividual,
) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_handles: list[Line2D] = []

    for path_idx, (individual, path) in enumerate(paths.items()):
        path_arcs = _path_to_arcs(path)
        invalid_arcs = [arc for arc in path_arcs if not graph.has_edge(*arc)]
        if invalid_arcs:
            raise ValueError(f"Path for {individual!r} contains invalid arcs: {invalid_arcs!r}")

        color = colors[path_idx % len(colors)]
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edgelist=path_arcs,
            edge_color=color,
            arrows=True,
            arrowsize=22,
            width=4.5,
            connectionstyle=f"arc3,rad={0.08 + 0.04 * path_idx}",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=3,
                label=_individual_label(individual),
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=False)


def _path_to_arcs(path: Sequence[Node] | Sequence[Arc]) -> list[Arc]:
    if len(path) == 0:
        return []

    first = path[0]
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[0], tuple | str):
        return list(path)  # type: ignore[arg-type]

    nodes = list(path)  # type: ignore[arg-type]
    return list(zip(nodes[:-1], nodes[1:]))


def _individual_label(individual: str | Individual) -> str:
    if isinstance(individual, Individual):
        return individual.id
    return individual


def _add_metric_colorbar(ax: Axes, values: Sequence[float], cmap, label: str) -> None:
    if not values:
        return
    norm = Normalize(vmin=float(min(values)), vmax=float(max(values)))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(values)
    ax.figure.colorbar(mappable, ax=ax, shrink=0.8, label=label)


def _get_ax(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, new_ax = plt.subplots()
    return new_ax


def _finish_network_ax(ax: Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0.15)
