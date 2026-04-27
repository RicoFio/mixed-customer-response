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

from .plotting import plot_policy_learning
from matplotlib import pyplot as plt

from .games.game_one import GameOne


def build_informative_prior(world: World) -> FinitePrior:
    source_right = ((0, 0), (0, 1))
    right_target = ((0, 1), (1, 1))
    source_to_bottom = ((0, 0), (1, 0))
    bottom_to_target = ((1, 0), (1, 1))

    base_scenario = Scenario.from_world("base", world)

    def scenario(
        *,
        name: str,
        top_travel_time: float,
        bottom_travel_time: float,
        top_cost: float,
        bottom_cost: float,
        top_hazard: float,
        bottom_hazard: float,
        top_emissions: float,
        bottom_emissions: float,
        policing: float,
    ) -> Scenario:
        return base_scenario.with_overrides(
            name=name,
            arc_overrides={
                MetricName.TRAVEL_TIME: {
                    source_right: top_travel_time,
                    right_target: top_travel_time,
                    source_to_bottom: bottom_travel_time,
                    bottom_to_target: bottom_travel_time,
                },
                MetricName.HAZARD: {
                    source_right: top_hazard,
                    right_target: top_hazard,
                    source_to_bottom: bottom_hazard,
                    bottom_to_target: bottom_hazard,
                },
                MetricName.COST: {
                    source_right: top_cost,
                    right_target: top_cost,
                    source_to_bottom: bottom_cost,
                    bottom_to_target: bottom_cost,
                },
                MetricName.EMISSIONS: {
                    source_right: top_emissions,
                    right_target: top_emissions,
                    source_to_bottom: bottom_emissions,
                    bottom_to_target: bottom_emissions,
                },
            },
            node_overrides={
                MetricName.POLICING: {
                    (0, 0): policing,
                    (1, 0): policing,
                    (1, 1): policing,
                }
            },
        )

    fast_expensive = scenario(
        name="fast_expensive",
        top_travel_time=0.45,
        bottom_travel_time=0.90,
        top_cost=2.00,
        bottom_cost=0.20,
        top_hazard=0.30,
        bottom_hazard=0.80,
        top_emissions=0.60,
        bottom_emissions=1.10,
        policing=0.0,
    )
    fast_cheap = scenario(
        name="fast_cheap",
        top_travel_time=0.45,
        bottom_travel_time=0.35,
        top_cost=0.60,
        bottom_cost=1.40,
        top_hazard=0.30,
        bottom_hazard=0.20,
        top_emissions=0.60,
        bottom_emissions=0.50,
        policing=0.0,
    )
    slow_expensive = scenario(
        name="slow_expensive",
        top_travel_time=1.40,
        bottom_travel_time=0.80,
        top_cost=2.00,
        bottom_cost=1.40,
        top_hazard=1.20,
        bottom_hazard=0.50,
        top_emissions=1.60,
        bottom_emissions=0.90,
        policing=1.0,
    )
    slow_cheap = scenario(
        name="slow_cheap",
        top_travel_time=1.40,
        bottom_travel_time=1.60,
        top_cost=0.60,
        bottom_cost=1.40,
        top_hazard=1.20,
        bottom_hazard=1.60,
        top_emissions=1.60,
        bottom_emissions=1.80,
        policing=1.0,
    )

    support = {
        fast_expensive.name: fast_expensive,
        fast_cheap.name: fast_cheap,
        slow_expensive.name: slow_expensive,
        slow_cheap.name: slow_cheap,
    }
    return FinitePrior(
        name="prior",
        support=support,
        probabilities={scenario_name: 1.0 for scenario_name in support},
    )


def build_informative_game_one(seed: int = 1) -> GameOne:
    network = create_sample_graph(instrumented="tr")
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
                MetricName.TRAVEL_TIME: 0.2,
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

    plot_policy_learning(MetricName.COST, MetricName.TRAVEL_TIME, result)
    plt.show()
