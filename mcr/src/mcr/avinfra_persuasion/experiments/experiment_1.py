"""
In this experiment we consider:
- World
    - Is a grid world with information about 
- A finite set of priors with support
- A single sender
    - With a scalar utility metric
    - The sender only has to choose the probability of sending or not sending
- A single receiver
    - With a posetal preference
    - Routing based on the prior
    - Route selection based on the posterior
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..bp.game import OfflineGame, Preference
from ..bp.receivers import Receiver
from ..bp.senders import ScalarSender, Sender
from ..bp.signals import MaskSignalPolicy
from ..datastructures import (
    Arc,
    Demand,
    Prior,
    FinitePrior,
    Individual,
    MetricName,
    Node,
    Scenario,
    World,
)
from ..networks.toy_3 import create_sample_graph


class GameOne(OfflineGame):
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: Prior
    seed: int


if __name__ == "__main__":
    network = create_sample_graph()

    origin: Node = (0, 0)
    target: Node = (1, 1)
    individual = Individual(id="robert", demand=Demand(origin, target))
    individuals = frozenset([individual])

    world = World(
        network=network,
        individuals=individuals,
    )

    rng = np.random.default_rng(7)
    base_scenario = Scenario.from_world("base", world)

    def sample_relative_arc_metric(
        values: Mapping[Arc, float],
        lower_factor: float,
        upper_factor: float,
    ) -> dict[Arc, float]:
        if lower_factor > upper_factor:
            raise ValueError("lower_factor must be less than or equal to upper_factor.")
        return {
            key: max(0.0, value * rng.uniform(lower_factor, upper_factor))
            for key, value in values.items()
        }

    def make_prior_scenario(index: int) -> Scenario:
        return base_scenario.with_overrides(
            name=f"s{index}",
            arc_overrides={
                MetricName.TRAVEL_TIME: sample_relative_arc_metric(
                    dict(base_scenario.travel_time),
                    0.8,
                    1.2,
                ),
                MetricName.HAZARD: sample_relative_arc_metric(
                    dict(base_scenario.hazard),
                    1,
                    3,
                ),
            },
            node_overrides={
                MetricName.POLICING: {
                    target: float(index % 2 == 0),
                },
            },
        )

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

    prior = FinitePrior(
        name="prior",
        support={f"s{i}": make_prior_scenario(i) for i in range(5)},
        probabilities={f"s{i}": 0.2 for i in range(5)},
    )

    receiver = Receiver(
        individual=individual,
        rtype="human",
        preference=human_preference,
        prior=prior,
        world=world,
    )

    sender_preference = Preference(
        elements={MetricName.TRAVEL_TIME},
        relations=set(),
    )
    seed = 1
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        signal_policy=MaskSignalPolicy(
            seed=seed,
            considered_metrics=frozenset({
                MetricName.TRAVEL_TIME,
                MetricName.HAZARD,
                MetricName.POLICING,
            }),
            probabilities={
                MetricName.TRAVEL_TIME: 0.8,
                MetricName.HAZARD: 0.5,
            },
        ),
    )
    
    game = GameOne(
        sender=sender,
        receivers=[receiver],
        world=world,
        public_prior=prior,
        seed=seed
    )
