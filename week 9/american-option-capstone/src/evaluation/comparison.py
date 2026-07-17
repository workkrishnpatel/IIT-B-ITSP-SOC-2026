"""
Builds ONE comparison table: for every contract in the canonical grid,
compute the binomial price, the NN price (+ error vs binomial), and (for a
fixed reference contract) the RL policy's value + exercise rate. This is the
central artifact of the final report.

Run as:
    python -m src.evaluation.comparison --out reports/comparison.csv
"""

import argparse
import pandas as pd

from ..contract import OptionContract
from ..pricing.binomial import BinomialAmericanPutPricer
from ..data.contract_grid import make_contract_grid


def add_moneyness_bucket(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["S0"] / df["K"]
    df = df.copy()
    df["bucket"] = pd.cut(
        ratio,
        bins=[0.0, 0.9, 1.1, 10.0],
        labels=["ITM put", "ATM", "OTM put"],
    )
    return df


def summarize_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("bucket", observed=True)
    return grouped.agg(
        nn_mae=("nn_error", lambda x: float(pd_abs_mean(x))),
        nn_bias=("nn_error", "mean"),
        count=("bucket", "size"),
    ).reset_index()


def pd_abs_mean(x):
    return x.abs().mean()


def build_binomial_and_nn_table(grid: pd.DataFrame, nn_pricer=None) -> pd.DataFrame:
    """Binomial price (always computed) + NN price/error (only if an
    NNAmericanPutPricer instance is passed in -- keeps this importable
    without requiring TensorFlow just to build the binomial columns)."""
    binomial_pricer = BinomialAmericanPutPricer()
    rows = []

    for _, row in grid.iterrows():
        contract = OptionContract(
            S0=row["S0"], K=row["K"], T=row["T"], r=row["r"], sigma=row["sigma"],
            steps=int(row["steps"]),
        )
        binomial_result = binomial_pricer.price(contract)
        record = {
            "S0": contract.S0, "K": contract.K, "T": contract.T,
            "r": contract.r, "sigma": contract.sigma,
            "binomial_price": binomial_result.price,
        }
        rows.append(record)

    df = pd.DataFrame(rows)

    if nn_pricer is not None:
        nn_prices = nn_pricer.predict_batch(df)
        df["nn_price"] = nn_prices
        df["nn_error"] = df["nn_price"] - df["binomial_price"]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/comparison.csv")
    parser.add_argument("--with-nn", action="store_true",
                         help="Also load and evaluate the NN pricer (requires TensorFlow)")
    parser.add_argument("--model-path", default="saved_models/american_put_pricing_model.keras")
    parser.add_argument("--scaler-path", default="saved_models/scaler.pkl")
    args = parser.parse_args()

    grid = make_contract_grid()

    nn_pricer = None
    if args.with_nn:
        from ..ml.nn_pricer import NNAmericanPutPricer
        nn_pricer = NNAmericanPutPricer(args.model_path, args.scaler_path)

    df = build_binomial_and_nn_table(grid, nn_pricer=nn_pricer)
    df = add_moneyness_bucket(df)

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    if "nn_error" in df.columns:
        print("\nSummary by moneyness bucket:")
        print(summarize_by_bucket(df))


if __name__ == "__main__":
    main()
