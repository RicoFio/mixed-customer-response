"""
Finite-prior toy experiment for independent Bernoulli disclosure by metric.

The sender chooses one disclosure probability per metric. For each realized
scenario, an independent mask is sampled, the receiver conditions on the
revealed network values, and then chooses a route. The sender policy is
optimized with the same Adam + finite-difference pattern as the basic
Bayesian persuasion toy, but on top of the routing model.
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

from .helpers import (
    format_mask,
    format_posterior,
)

from .plotting import plot_policy_gradient_field, plot_policy_learning
from matplotlib import pyplot as plt

from .games.game_one import GameOne


def build_informative_prior(world: World) -> FinitePrior:
    source_right = ((0, 0), (0, 1))
    right_target = ((0, 1), (1, 1))
    source_to_bottom = ((0, 0), (1, 0))
    bottom_to_target = ((1, 0), (1, 1))

    base_scenario = Scenario.from_world("base", world)

    good = base_scenario.with_overrides(
        name="instrumented_good",
        arc_overrides={
            MetricName.TRAVEL_TIME: {
                source_right: 0.45,
                right_target: 0.45,
                source_to_bottom: 1.20,
                bottom_to_target: 1.20,
            },
            MetricName.HAZARD: {
                source_right: 0.30,
                right_target: 0.30,
                source_to_bottom: 1.00,
                bottom_to_target: 1.00,
            },
            MetricName.COST: {
                source_right: 2.00,
                right_target: 2.00,
                source_to_bottom: 0.00,
                bottom_to_target: 0.40,
            },
            MetricName.EMISSIONS: {
                source_right: 0.60,
                right_target: 0.60,
                source_to_bottom: 1.40,
                bottom_to_target: 1.40,
            },
        },
        node_overrides={
            MetricName.POLICING: {
                (0, 0): 0.0,
                (1, 0): 0.0,
                (1, 1): 0.0,
            }
        },
    )
    bad = base_scenario.with_overrides(
        name="instrumented_bad",
        arc_overrides={
            MetricName.TRAVEL_TIME: {
                source_right: 1.80,
                right_target: 1.80,
                source_to_bottom: 0.95,
                bottom_to_target: 0.95,
            },
            MetricName.HAZARD: {
                source_right: 1.20,
                right_target: 1.20,
                source_to_bottom: 0.50,
                bottom_to_target: 0.50,
            },
            MetricName.COST: {
                source_right: 0.60,
                right_target: 0.60,
                source_to_bottom: 1.40,
                bottom_to_target: 1.40,
            },
            MetricName.EMISSIONS: {
                source_right: 1.60,
                right_target: 1.60,
                source_to_bottom: 0.90,
                bottom_to_target: 0.90,
            },
        },
        node_overrides={
            MetricName.POLICING: {
                (0, 0): 1.0,
                (1, 0): 1.0,
                (1, 1): 1.0,
            }
        },
    )

    return FinitePrior(
        name="prior",
        support={
            good.name: good,
            bad.name: bad,
        },
        probabilities={
            good.name: 0.5,
            bad.name: 0.5,
        },
    )


def build_informative_game_one(seed: int = 1) -> GameOne:
    network = create_sample_graph()
    origin: Node = (0, 0)
    target: Node = (1, 1)
    individual = Individual(id="robert", demand=Demand(origin, target))
    world = World(network=network, individuals=frozenset({individual}))
    prior = build_informative_prior(world)

    human_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.COST,
            # MetricName.HAZARD,
            # MetricName.EMISSIONS,
        },
        relations={
            (MetricName.COST, MetricName.TRAVEL_TIME),
            # (MetricName.EMISSIONS, MetricName.TRAVEL_TIME),
        },
    )
    sender_preference = Preference(
        elements={MetricName.COST},
        relations=set(),
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        objective=Objective.MAXIMIZE,
        signal_policy=MaskSignalPolicy(
            seed=seed,
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    # MetricName.HAZARD,
                    MetricName.COST,
                }
            ),
            probabilities={
                MetricName.TRAVEL_TIME: 0.1,
                # MetricName.HAZARD: 0.2,
                MetricName.COST: 0.4,
            },
        ),
    )
    receiver = PriorRouteChoiceReceiver(
        individual=individual,
        rtype="human",
        preference=human_preference,
        prior=prior,
        world=world,
        sender=sender,
        n_scenarios=len(prior.support),
    )
    return GameOne(
        sender=sender,
        receivers=[receiver],
        world=world,
        public_prior=prior,
        seed=seed,
    )


if __name__ == "__main__":
    game = build_informative_game_one(seed=1)
    result = game.solve(max_iter=5000, progress=True)
    diagnostics = game.diagnostics(result["final_probabilities"])

    print("Converged:", result["converged"])
    print("Iterations:", result["iterations"])
    print("Final probabilities:")
    for metric, probability in result["final_probabilities"].items():
        print(f"  {metric.value}: {probability:.4f}")

    bayes_report = diagnostics["bayes_report"]
    print("Bayes plausibility max error:", f"{bayes_report['max_error']:.3e}")
    print("Bayes posteriors:")
    for row in bayes_report["rows"]:
        print(
            f"  {row['signal_summary']} | p={row['signal_probability']:.3f} | "
            f"{format_posterior(row['posterior_probabilities'])}"
        )

    print("Mask semantics:")
    for row in diagnostics["mask_verification_rows"]:
        observation_text = ", ".join(
            (
                f"{observed['metric'].value}[{observed['key']!r}] "
                f"nominal={observed['nominal_value']:.3f} "
                f"realized={observed['realized_value']:.3f}"
            )
            for observed in row["observed_values"]
        ) or "no observed values"
        print(
            f"  scenario={row['scenario_name']} | "
            f"mask={format_mask(row['mask'])} | "
            f"posterior_changed={row['posterior_changed']} | {observation_text}"
        )

    print("Per-scenario / per-mask breakdown:")
    for row in result["breakdown_rows"]:
        print(
            f"  scenario={row['scenario_name']} "
            f"(p={row['scenario_probability']:.3f}) | "
            f"mask={format_mask(row['mask'])} "
            f"(p={row['mask_probability']:.3f}) | "
            f"signal={row['signal_summary']} | "
            f"posterior={format_posterior(row['posterior_probabilities'])} | "
            f"path={row['chosen_path']} | "
            f"sender_metric={row['sender_metric_value']:.3f} | "
            f"contribution={row['weighted_contribution']:.3f}"
        )

    print("Final expected sender metric:", f"{result['expected_sender_metric']:.4f}")

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_policy_learning(MetricName.COST, MetricName.TRAVEL_TIME, result, ax=axes[0])
    plot_policy_gradient_field(
        MetricName.COST,
        MetricName.TRAVEL_TIME,
        game,
        result=result,
        ax=axes[1],
    )
    plt.show()
