"""
CRR (Cox-Ross-Rubinstein) binomial American put pricer.

This is your existing american_puts.py logic -- verified correct
(American >= European, converges smoothly with more steps) -- wrapped
in the Pricer/PricingResult interface the Week 9 PDF recommends so the
binomial, NN, and RL methods all share one calling convention.
"""

import math
import numpy as np

from ..contract import OptionContract


def crr_put_price(S0: float, K: float, T: float, r: float, sigma: float,
                   steps: int, american: bool) -> float:
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")

    steps = int(steps)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r * dt)

    if not (0.0 < p < 1.0):
        raise ValueError("invalid risk-neutral probability: increase step size or check inputs")

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)

    for i in range(steps - 1, -1, -1):
        value = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])
        if american:
            j = np.arange(i + 1)
            stock = S0 * (u ** j) * (d ** (i - j))
            exercise = np.maximum(K - stock, 0.0)
            value = np.maximum(value, exercise)

    return float(value[0])


def crr_put_with_boundary(S0: float, K: float, T: float, r: float, sigma: float, steps: int):
    """Same recursion as crr_put_price(american=True), but also records the
    highest spot at each time step where immediate exercise beats continuation
    -- this is the exercise boundary used for the exercise-region plot."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)
    boundary = []

    for i in range(steps - 1, -1, -1):
        continuation = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])

        j = np.arange(i + 1)
        stock = S0 * (u ** j) * (d ** (i - j))
        exercise = np.maximum(K - stock, 0.0)

        exercise_now = exercise > continuation + 1e-10
        if np.any(exercise_now):
            boundary_stock = float(np.max(stock[exercise_now]))
            boundary.append((i * dt, boundary_stock))

        value = np.maximum(continuation, exercise)

    boundary.reverse()
    return float(value[0]), boundary


class PricingResult:
    def __init__(self, price, metadata=None):
        self.price = float(price)
        self.metadata = metadata or {}


class Pricer:
    name = "base"

    def price(self, contract: OptionContract) -> PricingResult:
        raise NotImplementedError


class BinomialAmericanPutPricer(Pricer):
    name = "crr_binomial_american_put"

    def price(self, contract: OptionContract) -> PricingResult:
        contract.validate()
        price, boundary = crr_put_with_boundary(
            contract.S0, contract.K, contract.T, contract.r, contract.sigma, contract.steps
        )
        return PricingResult(price, {"exercise_boundary": boundary})
