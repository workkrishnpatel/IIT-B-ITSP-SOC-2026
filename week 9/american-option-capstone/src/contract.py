"""
One consistent contract object shared by every pricer in this project.

Every method (binomial, NN, RL) prices the SAME OptionContract, so comparisons
are never contaminated by one method silently using different parameters than
another (this is the #1 thing the Week 9 PDF warns about).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionContract:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    steps: int = 100
    option_type: str = "put"

    def validate(self):
        assert self.S0 > 0, "S0 must be positive"
        assert self.K > 0, "K must be positive"
        assert self.T > 0, "T must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert self.steps >= 1, "steps must be a positive integer"
        assert self.option_type in {"put", "call"}, "option_type must be 'put' or 'call'"
        return self
