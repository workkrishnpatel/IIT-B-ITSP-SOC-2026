"""
American put optimal-stopping environment. State = [time_fraction,
time_to_expiry, moneyness] -- 3 features, matching week8_dqn_online.keras's
input shape. No future prices or future payoffs ever enter the state
(no lookahead leakage).
"""

import math
import numpy as np


class AmericanPutEnv:
    HOLD = 0
    EXERCISE = 1

    def __init__(self, S0, K, T, r, sigma, steps, seed=None):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.rng = np.random.default_rng(seed)

        self.dt = T / steps
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-r * self.dt)

        self.reset()

    def _state(self):
        time_fraction = self.step / self.steps
        time_to_expiry = 1.0 - time_fraction
        moneyness = self.spot / self.K
        return np.array([time_fraction, time_to_expiry, moneyness], dtype=np.float32)

    def reset(self):
        self.step = 0
        self.spot = self.S0
        self.done = False
        return self._state()

    def step_env(self, action):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")

        payoff = max(self.K - self.spot, 0.0)

        if action == self.EXERCISE:
            self.done = True
            return self._state(), payoff, True, {"reason": "exercise"}

        if action != self.HOLD:
            raise ValueError("action must be 0=hold or 1=exercise")

        if self.rng.random() < self.p:
            self.spot *= self.u
        else:
            self.spot *= self.d

        self.step += 1
        if self.step >= self.steps:
            self.done = True
            terminal_payoff = max(self.K - self.spot, 0.0)
            return self._state(), terminal_payoff, True, {"reason": "expiry"}

        return self._state(), 0.0, False, {"reason": "hold"}
