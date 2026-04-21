from .base_env import BaseEnv
from pydantic import model_validator

from .base_env import Lambda


class OptimalInformationDesignEnv(BaseEnv):
    @model_validator(mode="after")
    def validate_tau(self):
        ub = self.D - (self.cost_diff) / (self.alpha_1_a + self.alpha_2)
        lb = self.D - (self.cost_diff) / (self.alpha_1_n + self.alpha_2)
        if not (self.alpha_1_a > self.alpha_2 > self.alpha_1_n):
            raise ValueError("alpha_1_a > alpha_2 > alpha_1_n must hold")
        if not (lb < self.tau < ub):
            raise ValueError("tau must be within the valid range")
        return self

    @property
    def pop_lambda_top(self):
        return (
            1
            - (self.cost_diff) / ((self.alpha_1_a + self.alpha_2) * self.D)
            - self.tau / self.D
        )
    
    @property
    def pop_lambda_bottom(self):
        return (
            (self.D - self.tau) * (self.alpha_1_top_theta + self.alpha_2)
            - self.cost_diff
        ) / (self.D * self.p * (self.alpha_1_a + self.alpha_2))

    @property
    def p_top(self):
        return (
            1
            / (self.alpha_1_a - self.alpha_1_n)
            * ((self.cost_diff) / (self.D - self.tau) - self.alpha_2 - self.alpha_1_n)
        )

    @property
    def regime(self) -> Lambda:
        if 0 <= self.pop_lambda < self.pop_lambda_bottom:
            return Lambda.L1
        elif self.pop_lambda_bottom <= self.pop_lambda < self.pop_lambda_top:
            return Lambda.L2
        elif self.pop_lambda_top <= self.pop_lambda <= 1:
            return Lambda.L3
        else:
            raise ValueError("Invalid population parameter lambda")

    @property
    def alpha_1_top_theta(self):
        return self.p * self.alpha_1_a + (1 - self.p) * self.alpha_1_n
