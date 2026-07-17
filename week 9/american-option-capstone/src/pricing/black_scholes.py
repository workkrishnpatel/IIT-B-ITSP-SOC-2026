"""Black-Scholes closed-form European price. Used only as background context
(Weeks 1-2 of the course) -- it does NOT handle American early exercise, which
is exactly why the binomial model exists. No scipy needed: norm.cdf is
implemented with math.erf so this file has zero extra dependencies."""

import math


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put(S0, K, T, r, sigma):
    if T <= 0:
        return max(K - S0, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)


def black_scholes_call(S0, K, T, r, sigma):
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
