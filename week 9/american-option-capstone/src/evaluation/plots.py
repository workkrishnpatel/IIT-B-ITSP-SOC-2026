"""
The three required visualizations from the Week 9 PDF:
  1. Predicted-vs-benchmark scatter (validates the NN pricer)
  2. RL exercise-region heatmap over (time, moneyness)
  3. Binomial exercise boundary curve (context for #2)
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_vs_benchmark(df, output_path="reports/figures/nn_vs_binomial.png"):
    x = df["binomial_price"]
    y = df["nn_price"]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.6)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="y = x (perfect match)")
    plt.xlabel("Binomial benchmark price")
    plt.ylabel("NN predicted price")
    plt.title("NN price vs binomial benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_exercise_region(grid: np.ndarray, output_path="reports/figures/exercise_region.png",
                          money_min=0.5, money_max=1.5):
    plt.figure(figsize=(8, 5))
    plt.imshow(
        grid, aspect="auto", origin="lower",
        extent=[money_min, money_max, 0, 1], cmap="RdBu_r",
    )
    plt.colorbar(label="Action (0=hold, 1=exercise)")
    plt.xlabel("Moneyness (S/K)")
    plt.ylabel("Time fraction (t/T)")
    plt.title("RL Policy Exercise Region: American Put")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="S=K (ATM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_exercise_boundary(boundary, output_path="reports/figures/exercise_boundary.png"):
    """boundary: list of (time, boundary_spot) tuples from crr_put_with_boundary."""
    if not boundary:
        print("No boundary points to plot (option may never optimally exercise early).")
        return
    times, spots = zip(*boundary)
    plt.figure(figsize=(7, 4))
    plt.plot(times, spots, marker=".", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Exercise boundary spot")
    plt.title("Binomial American Put Exercise Boundary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")
