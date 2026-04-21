from dataclasses import dataclass
from .datastructures import (
    World, 
    WorldBelief, 
    Singleton, 
    Individual,
    Scenario,
    METRIC_SET,
)
from enum import Enum

from .orders import PartialOrder
from typing import Any, Callable, Mapping, Protocol, TypeAlias

TimeStep: TypeAlias = int
ReceiverType: TypeAlias = str


@dataclass(frozen=True)
class Prior(metaclass=Singleton):
    world: World
    
    def sample(self) -> Scenario:
        pass


@dataclass(frozen=True)
class Preference(PartialOrder):
    elements = METRIC_SET


@dataclass
class SignalPolicy:
    pass

@dataclass
class Sender:
    preference: Preference
    signal_policy: SignalPolicy
    

class Receiver(Individual):
    rtype: ReceiverType
    preference: Preference


class Game:
    sender: Sender
    receiver: list[Receiver]
    
    world: World
    public_prior: Prior
    horizon: TimeStep
    
    current_step: TimeStep = 0
    
    def step(self):
        # First we sample from the world
        # Then we have the receiver commit to a signaling policy
        # We pass the sample of the world to the receiver and get back one message per receiver
        # We pass the messages to the receivers, they update their beliefs and take an action
        # We pass the actions to the current state of the world and receive the realized state of the world (congestion, travel times, emissions, etc.)
        # We pass that back to both the sender and the receivers.
        # The receiver updates their signaling policy (Not Implemented Yet).
        pass
    