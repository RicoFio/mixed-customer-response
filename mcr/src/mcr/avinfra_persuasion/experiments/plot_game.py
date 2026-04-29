from matplotlib import pyplot as plt

from .experiment_2_1 import build_informative_game_two
from ..datastructures import MetricName
from ..plotting import plot_scenario

if __name__ == "__main__":
    game = build_informative_game_two(seed=1)
    world = game.world
    prior = game.public_prior

    scenarios = list(prior.support.values())
    metrics = [MetricName.TRAVEL_TIME, MetricName.HAZARD]
    metric_labels = ["Travel Time", "Cost"]

    fig, axes = plt.subplots(
        len(scenarios), len(metrics), figsize=(5 * len(scenarios), 4 * len(metrics))
    )

    for row, metric in enumerate(metrics):
        for col, scenario in enumerate(scenarios):
            ax = axes[col][row]
            plot_scenario(
                world=world,
                scenario=scenario,
                paths={},
                arc_metric=metric,
                node_metric=None,
                ax=ax,
            )
            ax.set_title(f"{scenario.name}\n{metric_labels[row]}")

    fig.tight_layout()
    plt.show()
    