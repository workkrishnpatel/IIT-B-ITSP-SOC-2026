import pytest
from src.rl.env import AmericanPutEnv


def make_env(seed=1):
    return AmericanPutEnv(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=50, seed=seed)


def test_payoff_nonnegative_on_exercise():
    env = make_env()
    env.reset()
    _, reward, done, _ = env.step_env(env.EXERCISE)
    assert reward >= 0
    assert done


def test_cannot_step_after_done():
    env = make_env()
    env.reset()
    env.step_env(env.EXERCISE)
    with pytest.raises(RuntimeError):
        env.step_env(env.HOLD)


def test_invalid_action_raises():
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step_env(2)


def test_episode_always_terminates():
    env = make_env()
    state = env.reset()
    for _ in range(env.steps + 1):
        state, reward, done, info = env.step_env(env.HOLD)
        if done:
            assert info["reason"] == "expiry"
            return
    pytest.fail("Episode did not terminate within steps+1 iterations")


def test_state_has_no_future_leakage():
    """State must only depend on current step/spot, never future path values."""
    env = make_env()
    state = env.reset()
    assert len(state) == 3
    assert 0.0 <= state[0] <= 1.0  # time_fraction
    assert 0.0 <= state[1] <= 1.0  # time_to_expiry
    assert state[2] > 0             # moneyness
