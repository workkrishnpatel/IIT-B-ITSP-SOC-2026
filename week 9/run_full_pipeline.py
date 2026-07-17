"""
Runs the full comparison pipeline in one go: binomial vs NN pricing comparison
+ plot, then RL policy evaluation + baselines + exercise-region plot.

Run from the repo root:
    python run_full_pipeline.py
"""

import pandas as pd

from src.data.contract_grid import make_contract_grid
from src.evaluation.comparison import build_binomial_and_nn_table, add_moneyness_bucket, summarize_by_bucket
from src.evaluation.plots import plot_predicted_vs_benchmark, plot_exercise_region, plot_exercise_boundary
from src.pricing.binomial import crr_put_with_boundary
from src.ml.nn_pricer import NNAmericanPutPricer
from src.rl.policy import (
    load_dqn, make_learned_policy, build_policy_comparison, policy_grid,
    always_hold_policy, immediate_exercise_policy, random_policy,
)


def main():
    print("=" * 70)
    print("STEP A: Binomial vs NN comparison over the canonical contract grid")
    print("=" * 70)

    grid = make_contract_grid()
    nn_pricer = NNAmericanPutPricer(
        "saved_models/american_put_pricing_model.keras",
        "saved_models/scaler.pkl",
    )
    comparison_df = build_binomial_and_nn_table(grid, nn_pricer=nn_pricer)
    comparison_df = add_moneyness_bucket(comparison_df)
    comparison_df.to_csv("reports/comparison.csv", index=False)
    print(f"Saved reports/comparison.csv ({len(comparison_df)} rows)")

    print("\nNN error summary by moneyness bucket:")
    print(summarize_by_bucket(comparison_df))

    print("\nOverall NN vs binomial error:")
    print(f"  MAE:  {comparison_df['nn_error'].abs().mean():.4f}")
    print(f"  Bias: {comparison_df['nn_error'].mean():.4f}")
    print(f"  Max:  {comparison_df['nn_error'].abs().max():.4f}")

    plot_predicted_vs_benchmark(comparison_df)

    print("\n" + "=" * 70)
    print("STEP B: Binomial exercise boundary plot (reference contract)")
    print("=" * 70)
    ref_contract = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=100)
    _, boundary = crr_put_with_boundary(**ref_contract)
    plot_exercise_boundary(boundary)

    print("\n" + "=" * 70)
    print("STEP C: RL policy evaluation vs baselines (5000 episodes each)")
    print("=" * 70)

    model = load_dqn("saved_models/week8_dqn_online.keras")
    learned_policy = make_learned_policy(model)
    env_kwargs = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=100)

    policy_comparison_df = build_policy_comparison(env_kwargs, {
        "always_hold": always_hold_policy,
        "immediate_exercise": immediate_exercise_policy,
        "random": random_policy,
        "learned_dqn": learned_policy,
    }, episodes=5000)

    policy_comparison_df.to_csv("reports/policy_comparison.csv", index=False)
    print(policy_comparison_df.to_string(index=False))
    print("Saved reports/policy_comparison.csv")

    binomial_price = comparison_df.loc[
        (comparison_df["S0"] == 100) & (comparison_df["sigma"] == 0.25)
        & (comparison_df["T"] == 1.0) & (comparison_df["r"] == 0.05),
        "binomial_price"
    ]
    if len(binomial_price):
        bp = float(binomial_price.iloc[0])
        learned_value = float(policy_comparison_df.loc[
            policy_comparison_df["policy"] == "learned_dqn", "value"
        ].iloc[0])
        print(f"\nBinomial benchmark (same contract): {bp:.4f}")
        print(f"DQN learned value:                  {learned_value:.4f}")
        print(f"Difference (DQN - binomial):        {learned_value - bp:.4f}")

    print("\n" + "=" * 70)
    print("STEP D: RL exercise-region plot")
    print("=" * 70)
    grid_actions = policy_grid(learned_policy, steps=100)
    plot_exercise_region(grid_actions)

    print("\nAll done. Check reports/comparison.csv, reports/policy_comparison.csv,")
    print("and reports/figures/ for the three plots.")


if __name__ == "__main__":
    main()
