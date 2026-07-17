"""Terminal / exercise payoff functions. These are deliberately tiny and
tested directly, because every other pricer's correctness depends on them."""

import numpy as np


def put_payoff(S, K):
    """max(K - S, 0)"""
    return np.maximum(np.asarray(K, dtype=float) - np.asarray(S, dtype=float), 0.0)


def call_payoff(S, K):
    """max(S - K, 0)"""
    return np.maximum(np.asarray(S, dtype=float) - np.asarray(K, dtype=float), 0.0)


def payoff(S, K, option_type="put"):
    if option_type == "put":
        return put_payoff(S, K)
    if option_type == "call":
        return call_payoff(S, K)
    raise ValueError(f"Unknown option_type: {option_type}")
