from __future__ import annotations
from abc import ABC
from pydantic import BaseModel, Field, model_validator
from enum import Enum


Route = Enum("Route", "r1 r2")
Accident = Enum("Accident", "a n")
Lambda = Enum("Lambda", "L1 L2 L3")


class BaseEnv(BaseModel, ABC):
    # Congestion parameters
    alpha_1_a: float
    alpha_1_n: float
    alpha_2: float

    # Probability of accident
    p: float = Field(..., ge=0, le=1)

    # Free-flow travel times
    b_1: float
    b_2: float

    # Population parameter
    pop_lambda: float = Field(..., ge=0, le=1)

    # Total demand
    D: int = Field(..., gt=0)

    # Threshold
    tau: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_b_ordering(self):
        if not (self.b_1 < self.b_2):
            raise ValueError("b_1 must be less than b_2")
        return self

    @model_validator(mode="after")
    def validate_alpha_ordering(self):
        if not (self.alpha_1_a > self.alpha_2 > self.alpha_1_n):
            raise ValueError("alpha_1_a > alpha_2 > alpha_1_n must hold")
        return self

    @model_validator(mode="after")
    def validate_total_demand(self):
        if not (self.D > (self.b_2 - self.b_1) / self.alpha_1_n):
            raise ValueError("Total demand D must be positive")
        return self

    def cost_1_a(self, f_1):
        return self.alpha_1_a * f_1 + self.b_1

    def cost_1_n(self, f_1):
        return self.alpha_1_n * f_1 + self.b_1

    def cost_2(self, f_2):
        return self.alpha_2 * f_2 + self.b_2

    @property
    def cost_diff(self):
        return self.alpha_2 * self.D + self.b_2 - self.b_1

    @property
    def p_top(self):
        return (
            1
            / (self.alpha_1_a - self.alpha_1_n)
            * ((self.cost_diff) / (self.D - self.tau) - self.alpha_2 - self.alpha_1_n)
        )
