"""
Loads your trained week8_dqn_online.keras (confirmed Double DQN) and turns it into a usable policy:
greedy action selection, Monte Carlo policy evaluation (with std error),
baseline policies for comparison, and an exercise-region grid for plotting.
"""

import numpy as np
from tensorflow import keras

from .env import AmericanPutEnv


def load_dqn(model_path: str):
    return keras.models.load_model(model_path)


def greedy_action(model, state) -> int:
    q_values = model(np.expand_dims(state, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values))


def make_learned_policy(model):
    """Returns a policy_fn(state) -> action, closing over the loaded model."""
    def policy_fn(state):
        return greedy_action(model, state)
    return policy_fn


# --- baseline policies (for comparison table) ---

def always_hold_policy(state):
    return AmericanPutEnv.HOLD


def immediate_exercise_policy(state):
    return AmericanPutEnv.EXERCISE


def random_policy(state):
    return np.random.randint(0, 2)


# --- evaluation ---

def evaluate_policy(env_kwargs: dict, policy_fn, episodes: int = 5000, seed_offset: int = 0) -> dict:
    discounted_rewards = []
    exercise_steps = []

    for seed in range(episodes):
        env = AmericanPutEnv(seed=seed + seed_offset, **env_kwargs)
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = policy_fn(state)
            state, reward, done, info = env.step_env(action)
            if done:
                discounted_rewards.append((env.discount ** step) * reward)
                if info["reason"] == "exercise":
                    exercise_steps.append(step)
            step += 1

    discounted_rewards = np.array(discounted_rewards)
    return {
        "value": float(np.mean(discounted_rewards)),
        "std_error": float(np.std(discounted_rewards) / np.sqrt(episodes)),
        "exercise_rate": len(exercise_steps) / episodes,
        "avg_exercise_step": float(np.mean(exercise_steps)) if exercise_steps else None,
    }


def build_policy_comparison(env_kwargs: dict, policies: dict, episodes: int = 5000) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for name, policy_fn in policies.items():
        result = evaluate_policy(env_kwargs, policy_fn, episodes=episodes)
        rows.append({
            "policy": name,
            "value": result["value"],
            "std_error": result["std_error"],
            "exercise_rate": result["exercise_rate"],
            "avg_exercise_step": result["avg_exercise_step"],
        })
    return pd.DataFrame(rows).sort_values("value", ascending=False)


# --- exercise-region grid, for plotting ---

def policy_grid(policy_fn, steps=100, money_min=0.5, money_max=1.5, n_money=100):
    grid = []
    for step in range(steps + 1):
        row = []
        time_fraction = step / steps
        time_to_expiry = 1.0 - time_fraction
        for m in np.linspace(money_min, money_max, n_money):
            state = np.array([time_fraction, time_to_expiry, m], dtype=np.float32)
            row.append(policy_fn(state))
        grid.append(row)
    return np.array(grid)


def boundary_agreement(policy_fn, boundary_by_step: dict, steps=100, K=100):
    """Fraction of grid points where the RL policy's exercise decision agrees
    with the binomial exercise boundary at the same (time, moneyness)."""
    checks = []
    money_grid = np.linspace(0.6, 1.4, 81)

    for step in range(steps):
        boundary_spot = boundary_by_step.get(step)
        if boundary_spot is None:
            continue
        for m in money_grid:
            S = m * K
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            policy_exercise = policy_fn(state) == 1
            binomial_exercise = S <= boundary_spot
            checks.append(policy_exercise == binomial_exercise)

    return float(np.mean(checks)) if checks else None
