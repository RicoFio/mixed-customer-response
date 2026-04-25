from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from ..orders.partial_order import PartialOrder

from ..datastructures import World
from ..datastructures import Prior

if TYPE_CHECKING:
    from .receivers import Receiver
    from .senders import Sender

TimeStep: TypeAlias = int
Preference: TypeAlias = PartialOrder

RNG_SEED: int = 1
N_SCENARIOS: int = 10
AVERAGE_SAMPLING: bool = True



@dataclass
class Game(Protocol):
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: Prior
    seed: int


@dataclass
class ConvergenceGame(Game, Protocol):
    pass


@dataclass
class OnlineGame(Game, Protocol):

    horizon: TimeStep
    current_step: TimeStep = 0
    
    def step(self) -> None:
        # First we sample from the world
        # Then we have the receiver commit to a signaling policy
        # We pass the sample of the world to the receiver and get back one message per receiver
        # We pass the messages to the receivers, they update their beliefs and take an action
        # We pass the actions to the current state of the world and receive the realized state of the world (congestion, travel times, emissions, etc.)
        # We pass that back to both the sender and the receivers.
        # The receiver updates their signaling policy (Not Implemented Yet).
        ...
