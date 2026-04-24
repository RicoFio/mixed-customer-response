from dataclasses import dataclass
from ..datastructures import Prior, World, MetricName
from .signals import SignalPolicy
from .game import Preference


@dataclass
class Sender:
    prior: Prior
    world: World
    preference: Preference
    signal_policy: SignalPolicy


@dataclass
class ScalarSender(Sender):

    def __post_init__(self):
        if not self.preference.is_degenerate():
            raise ValueError(
                "A ScalarSender can only consider one metric. Preference needs to be `degenerate`"
            )


@dataclass
class OnlineSender(Sender):
    pass
