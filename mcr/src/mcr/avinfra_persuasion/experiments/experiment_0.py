"""
Finite-prior toy experiment for independent Bernoulli disclosure by metric.

The sender chooses one disclosure probability per metric. For each realized
scenario, an independent mask is sampled, the receiver conditions on the
revealed network values, and then chooses a route. The sender policy is
optimized with the same Adam + finite-difference pattern as the basic
Bayesian persuasion toy, but on top of the routing model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..bp.game import ConvergenceGame, Preference
from ..bp.receivers import Receiver
from ..bp.senders import ScalarSender, Sender
from ..bp.signals import MaskSignalPolicy, Signal
from ..datastructures import (
    Arc,
    Demand,
    FinitePrior,
    Individual,
    MetricName,
    Node,
    Prior,
    Scenario,
    World,
)
from ..networks.toy_0 import create_sample_graph
from ..opt import RoutingSolution, RoutingSolutionPoint
from ..routing.routing_solvers import solve_routes

SignalObservationKey = tuple[
    tuple[MetricName, tuple[tuple[object, float], ...]],
    ...,
]


class GameZero(ConvergenceGame):
    """
    This is to reconstruct the game by Manxi Wu & Saurabh Amin 
    with the finite differences case and the disclosure of accident/no accident
    on a simple two-route network.
    """
    pass


if __name__=="__main__":
    pass