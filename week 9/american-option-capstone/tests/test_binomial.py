"""
Fixed version of your test_american_put.py. Bugs that were present and fixed:
  1. `from american_puts import crr_puts_price` -> function is actually named
     crr_put_price (no 's'). This import error would have crashed the whole file.
  2. `import numpy as no` -> typo, and numpy wasn't even used directly.
  3. Functions didn't start with `test_`, so pytest silently never ran them.
  4. Functions used `return True/False` instead of `assert` -- pytest does NOT
     fail a test just because it returns False, so these tests could never fail.
  5. put_falls_on_spot_rises() had swapped variable names (`high_spot` was
     assigned the LOW spot's price and vice versa) AND mixed american=False
     with american=True in the same comparison -- not an apples-to-apples check.
"""

import pytest

from src.pricing.binomial import crr_put_price, crr_put_with_boundary


def test_american_put_at_least_european():
    args = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=500)
    euro = crr_put_price(**args, american=False)
    amer = crr_put_price(**args, american=True)
    assert amer >= euro


def test_put_falls_as_spot_rises():
    price_low_spot = crr_put_price(80, 100, 1.0, 0.05, 0.25, 500, american=True)
    price_high_spot = crr_put_price(120, 100, 1.0, 0.05, 0.25, 500, american=True)
    assert price_low_spot > price_high_spot


def test_more_volatility_is_not_cheaper():
    low_vol = crr_put_price(120, 100, 1.0, 0.05, 0.15, 500, american=True)
    high_vol = crr_put_price(120, 100, 1.0, 0.05, 0.35, 500, american=True)
    assert high_vol >= low_vol


def test_zero_maturity_equals_intrinsic_value():
    price = crr_put_price(90, 100, 1e-12, 0.05, 0.25, 10, american=True)
    assert price == pytest.approx(10.0, abs=1e-2)


def test_price_never_negative():
    price = crr_put_price(150, 100, 1.0, 0.05, 0.25, 200, american=True)
    assert price >= 0.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        crr_put_price(-10, 100, 1.0, 0.05, 0.25, 100, american=True)
    with pytest.raises(ValueError):
        crr_put_price(100, 100, 1.0, 0.05, -0.25, 100, american=True)


def test_convergence_is_monotone_improving():
    """More steps should move the price closer to the finest-grid estimate."""
    args = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, american=True)
    reference = crr_put_price(**args, steps=2000)
    coarse_gap = abs(crr_put_price(**args, steps=25) - reference)
    fine_gap = abs(crr_put_price(**args, steps=500) - reference)
    assert fine_gap < coarse_gap


def test_boundary_rises_toward_strike_near_expiry():
    """Sanity check on the exercise boundary shape: it should be below K and
    generally increase (get closer to K) as time approaches expiry."""
    _, boundary = crr_put_with_boundary(100, 100, 1.0, 0.05, 0.25, 200)
    assert len(boundary) > 0
    times, spots = zip(*boundary)
    assert all(s < 100 for s in spots)
    # last boundary point (closest to expiry) should be higher than the first
    assert spots[-1] > spots[0]


def convergence_table(S0, K, T, r, sigma):
    """Not a test -- a helper you can call manually to print convergence."""
    rows = []
    for steps in [25, 50, 100, 200, 500, 1000]:
        price = crr_put_price(S0, K, T, r, sigma, steps, american=True)
        rows.append((steps, price))
    return rows


if __name__ == "__main__":
    for steps, price in convergence_table(100, 100, 1.0, 0.05, 0.25):
        print(f"{steps:4d} steps -> {price:.6f}")
