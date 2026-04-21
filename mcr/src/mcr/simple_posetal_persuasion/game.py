from dataclasses import dataclass
import random

from typing import Optional, Any
from enum import Enum


@dataclass
class World:
    Theta: list[Enum]
    seed: Optional[int] = None
    
    @property
    def num_states(self):
        return len(self.Theta)
    
    def sample_state(self, seed=None):
        if seed is not None:
            random.seed(seed)
        return random.choice(self.Theta) 

@dataclass
class Sender:
    signal: dict

@dataclass
class Receiver:
    pass
