from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..datastructures import MetricName
from .helpers import format_mask


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


def plot_state_mask_policy(
    result: Mapping[str, object],
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a compact heatmap of the learned state-conditional mask distribution.

    The ``result`` argument is expected to be the dictionary returned by
    ``GameTwo.solve()``, with a ``final_probabilities`` entry mapping each
    state name to a mask-probability table.
    """
    final_probabilities = result.get("final_probabilities")
    if not isinstance(final_probabilities, Mapping) or not final_probabilities:
        raise ValueError(
            "result must contain a non-empty 'final_probabilities' mapping."
        )

    state_names = tuple(sorted(str(state_name) for state_name in final_probabilities))
    first_distribution = final_probabilities[state_names[0]]
    if not isinstance(first_distribution, Mapping) or not first_distribution:
        raise ValueError(
            "Each state distribution in 'final_probabilities' must be a "
            "non-empty mapping."
        )

    masks = tuple(
        sorted(
            first_distribution,
            key=lambda mask: (
                len(mask),
                tuple(
                    metric.value
                    for metric in sorted(mask, key=lambda metric: metric.value)
                ),
            ),
        )
    )
    values = np.asarray(
        [
            [
                float(
                    _state_mask_probability(
                        final_probabilities[state_name],
                        mask,
                    )
                )
                for mask in masks
            ]
            for state_name in state_names
        ],
        dtype=float,
    )

    width = max(6.0, 1.3 * len(masks))
    height = max(3.0, 0.8 * len(state_names) + 1.5)
    ax = ax or plt.subplots(figsize=(width, height))[1]
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels([format_mask(mask) for mask in masks], rotation=25, ha="right")
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels(state_names)
    ax.set_xlabel("Mask")
    ax.set_ylabel("State")
    ax.set_title("State-dependent mask policy")

    for row_idx, state_name in enumerate(state_names):
        for col_idx, mask in enumerate(masks):
            value = values[row_idx, col_idx]
            text_color = "#111111" if value > 0.62 else "#f5f5f5"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("P(mask | state)")
    return ax


def _state_mask_probability(
    distribution: object,
    mask: frozenset[MetricName],
) -> float:
    if not isinstance(distribution, Mapping):
        raise ValueError("Each state distribution must be a mapping.")
    if mask in distribution:
        return float(distribution[mask])
    raise ValueError(f"State distribution does not contain mask {format_mask(mask)!r}.")
