"""
Finite-prior toy experiment with a family of receivers.

The prior support is an epistemic model of the world used for signaling and
belief updates. Once all receivers choose routes, the actual realized network
metrics are computed endogenously from the joint path profile via
``world.get_realized_metrics(...)``.
"""

from __future__ import annotations

from ..bp.game import Preference
from ..bp.receivers import PriorRouteChoiceReceiver
from ..bp.senders import ScalarSender, Objective
from ..bp.signals import MaskSignalPolicy
from ..datastructures import (
    Demand,
    FinitePrior,
    Individual,
    MetricName,
    Node,
    Scenario,
    World,
)
from ..networks.toy_3 import create_sample_graph
from .helpers import format_mask
from .plotting import plot_policy_gradient_field, plot_policy_learning
from .games.osmr import OSMRGame

from matplotlib import pyplot as plt


def build_informative_prior(world: World) -> FinitePrior:
    source_right = ((0, 0), (0, 1))
    right_target = ((0, 1), (1, 1))
    source_to_bottom = ((0, 0), (1, 0))
    bottom_to_target = ((1, 0), (1, 1))

    base_scenario = Scenario.from_world("base", world)

    def make_scenario(
        name: str,
        *,
        top_travel_time: float,
        top_hazard: float,
        node_policing: int,
    ) -> Scenario:
        return base_scenario.with_overrides(
            name=name,
            arc_overrides={
                MetricName.TRAVEL_TIME: {
                    source_right: top_travel_time,
                    right_target: top_travel_time,
                    source_to_bottom: 1.0,
                    bottom_to_target: 1.0,
                },
                MetricName.HAZARD: {
                    source_right: top_hazard,
                    right_target: top_hazard,
                    source_to_bottom: 0.5,
                    bottom_to_target: 0.5,
                },
                MetricName.COST: {
                    source_right: 1.0,
                    right_target: 1.0,
                    source_to_bottom: 0.0,
                    bottom_to_target: 0.0,
                },
                MetricName.EMISSIONS: {
                    source_right: 1.0,
                    right_target: 1.0,
                    source_to_bottom: 0.5,
                    bottom_to_target: 0.5,
                },
            },
            node_overrides={
                MetricName.POLICING: {
                    (0, 0): node_policing,
                    (1, 0): 0.5,
                    (1, 1): node_policing,
                }
            },
        )

    support = {
        "fast_safe": make_scenario(
            "fast_safe",
            top_travel_time=0.25,
            top_hazard=0.0,
            node_policing=1,
        ),
        "fast_risky": make_scenario(
            "fast_risky",
            top_travel_time=0.25,
            top_hazard=2.0,
            node_policing=0,
        ),
        "slow_safe": make_scenario(
            "slow_safe",
            top_travel_time=1.25,
            top_hazard=0.0,
            node_policing=1,
        ),
        "slow_risky": make_scenario(
            "slow_risky",
            top_travel_time=1.25,
            top_hazard=2.0,
            node_policing=0,
        ),
    }

    return FinitePrior(
        name="prior",
        support=support,
        probabilities={
            "fast_safe": 0.4,
            "fast_risky": 0.2,
            "slow_safe": 0.3,
            "slow_risky": 0.1,
        },
    )


def build_informative_game_two(
    seed: int = 1,
    n_humans: int = 10,
    n_avs: int = 10,
) -> OSMRGame:
    network = create_sample_graph("tl")
    origin: Node = (0, 0)
    target: Node = (1, 1)
    individuals = frozenset(
        [Individual(id=f"h{i}", demand=Demand(origin, target)) for i in range(n_humans)]
        + [Individual(id=f"a{i}", demand=Demand(origin, target)) for i in range(n_avs)]
    )
    world = World(network=network, individuals=individuals)
    prior = build_informative_prior(world)

    human_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.COST,
            MetricName.EMISSIONS,
        },
        relations={
            (MetricName.COST, MetricName.TRAVEL_TIME),
            (MetricName.EMISSIONS, MetricName.TRAVEL_TIME),
        },
    )
    # human_preference.draw_hasse_diagram(ax=ax)
    # plt.show()
    av_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.HAZARD,
            MetricName.COST,
        },
        relations={
            (MetricName.COST, MetricName.HAZARD),
            (MetricName.TRAVEL_TIME, MetricName.HAZARD),
        },
    )
    # av_preference.draw_hasse_diagram()
    # plt.show()
    sender_preference = Preference(
        elements={MetricName.TRAVEL_TIME},
        relations=set(),
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        objective=Objective.MINIMIZE,
        signal_policy=MaskSignalPolicy(
            seed=seed,
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    MetricName.HAZARD,
                }
            ),
            probabilities={
                MetricName.TRAVEL_TIME: 0.5,
                MetricName.HAZARD: 0.5,
            },
        ),
    )
    receivers = [
        PriorRouteChoiceReceiver(
            individual=individual,
            rtype="human" if individual.id.startswith("h") else "av" + str(individual.demand),
            preference=human_preference
            if individual.id.startswith("h")
            else av_preference,
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=len(prior.support),
        )
        for individual in sorted(individuals, key=lambda individual: individual.id)
    ]
    return OSMRGame(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=seed,
    )


if __name__ == "__main__":
    game = build_informative_game_two(seed=1, n_humans=5, n_avs=5)
    result = game.solve(max_iter=200)

    print("Converged:", result["converged"])
    print("Iterations:", result["iterations"])
    print("Final probabilities:")
    for metric, probability in result["final_probabilities"].items():
        print(f"  {metric.value}: {probability:.4f}")

    print("Per-scenario / per-mask breakdown:")
    for row in result["breakdown_rows"]:
        print(
            f"  scenario={row['scenario_name']} "
            f"(p={row['scenario_probability']:.3f}) | "
            f"mask={format_mask(row['mask'])} "
            f"(p={row['mask_probability']:.3f}) | "
            f"path_counts={row['path_counts']} | "
            f"sender_metric={row['sender_metric_value']:.3f} | "
            f"contribution={row['weighted_contribution']:.3f}"
        )

    print("Final expected sender utility:", f"{result['expected_sender_utility']:.4f}")

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_policy_learning(MetricName.HAZARD, MetricName.TRAVEL_TIME, result, ax=axes[0])
    plot_policy_gradient_field(
        MetricName.HAZARD,
        MetricName.TRAVEL_TIME,
        game,
        result=result,
        ax=axes[1],
    )
    plt.show()
