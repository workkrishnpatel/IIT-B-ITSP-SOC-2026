"""
The canonical contract grid: the ONE set of contracts that binomial, NN, and
RL are all evaluated against. Covers ITM/ATM/OTM puts, short/long maturities,
and low/high volatility -- per the PDF's "grid design rules".
"""

import itertools
import pandas as pd


def make_contract_grid(steps: int = 100) -> pd.DataFrame:
    spots = [70, 80, 90, 100, 110, 120, 130]
    strikes = [100]
    maturities = [0.25, 0.5, 1.0, 2.0]
    rates = [0.02, 0.05]
    sigmas = [0.15, 0.25, 0.40]

    rows = []
    for S0, K, T, r, sigma in itertools.product(spots, strikes, maturities, rates, sigmas):
        rows.append({
            "S0": S0,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "steps": steps,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_contract_grid()
    print(f"Generated {len(df)} contracts")
    print(df.head())
