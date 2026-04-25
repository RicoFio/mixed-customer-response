from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..datastructures import MetricName


def plot_policy_learning(
    metric_x: MetricName | str,
    metric_y: MetricName | str,
    result: Mapping[str, object],
    ax: Axes | None = None,
) -> Axes:
    """
    Plot the learned disclosure-policy trajectory in two selected dimensions.

    The ``result`` argument is expected to be the dictionary returned by
    ``GameOne.solve()``, with a ``policy_history`` entry containing one
    probability dictionary per iteration.
    """
    metric_x = MetricName.coerce(metric_x)
    metric_y = MetricName.coerce(metric_y)
    if metric_x == metric_y:
        raise ValueError("metric_x and metric_y must be different.")

    policy_history = result.get("policy_history")
    if not isinstance(policy_history, Sequence) or not policy_history:
        raise ValueError(
            "result must contain a non-empty 'policy_history' sequence."
        )

    points = np.asarray(
        [
            [
                float(_policy_probability(policy, metric_x)),
                float(_policy_probability(policy, metric_y)),
            ]
            for policy in policy_history
        ],
        dtype=float,
    )

    ax = ax or plt.subplots(figsize=(6, 6))[1]
    colors = np.linspace(0.2, 0.95, len(points))

    if len(points) > 1:
        deltas = points[1:] - points[:-1]
        ax.quiver(
            points[:-1, 0],
            points[:-1, 1],
            deltas[:, 0],
            deltas[:, 1],
            colors[:-1],
            angles="xy",
            scale_units="xy",
            scale=1,
            cmap="viridis",
            width=0.004,
            alpha=0.85,
            zorder=2,
        )

    ax.plot(
        points[:, 0],
        points[:, 1],
        color="#7f8c8d",
        linewidth=1.1,
        alpha=0.6,
        zorder=1,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        cmap="viridis",
        s=28,
        edgecolors="#202020",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        [points[0, 0]],
        [points[0, 1]],
        s=70,
        color="#f39c12",
        edgecolors="#202020",
        linewidths=0.7,
        zorder=4,
        label="start",
    )
    ax.scatter(
        [points[-1, 0]],
        [points[-1, 1]],
        s=80,
        color="#c0392b",
        edgecolors="#202020",
        linewidths=0.8,
        zorder=5,
        label="final",
    )

    for idx, (x_value, y_value) in enumerate(points):
        if idx in {0, len(points) - 1}:
            ax.annotate(
                str(idx),
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"P(reveal {metric_x.value})")
    ax.set_ylabel(f"P(reveal {metric_y.value})")
    ax.set_title("Policy learning")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    return ax


def _policy_probability(
    policy: object,
    metric: MetricName,
) -> float:
    if not isinstance(policy, Mapping):
        raise ValueError("Each policy_history entry must be a mapping.")

    if metric in policy:
        return float(policy[metric])

    metric_name = metric.value
    if metric_name in policy:
        return float(policy[metric_name])

    raise ValueError(f"Policy history does not contain {metric.value!r}.")
