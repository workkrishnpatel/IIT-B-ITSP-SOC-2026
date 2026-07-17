"""
Wraps your existing trained NN pricer (american_put_pricing_model.keras +
scaler.pkl) in the same Pricer interface as the binomial model, and adds the
financial sanity checks the PDF requires: intrinsic-value bound and spot
monotonicity. A neural network is not automatically a pricing model -- it
only becomes one after it survives these checks.
"""

import joblib
import numpy as np
from tensorflow import keras

from ..contract import OptionContract
from ..pricing.binomial import Pricer, PricingResult

FEATURE_ORDER = ["S0", "K", "T", "r", "sigma"]  # must match scaler.feature_names_in_ order


class NNAmericanPutPricer(Pricer):
    name = "nn_pricer"

    def __init__(self, model_path: str, scaler_path: str):
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def price(self, contract: OptionContract) -> PricingResult:
        contract.validate()
        x = np.array([[contract.S0, contract.K, contract.T, contract.r, contract.sigma]], dtype=np.float64)
        x_scaled = self._scale(x)
        pred = float(self.model.predict(x_scaled, verbose=0)[0, 0])
        return PricingResult(pred, {})

    def predict_batch(self, df) -> np.ndarray:
        """Vectorized prediction for a DataFrame with columns S0,K,T,r,sigma."""
        X = df[FEATURE_ORDER].to_numpy(dtype=np.float64)
        X_scaled = self._scale(X)
        return self.model.predict(X_scaled, verbose=0).reshape(-1)


# --- Financial sanity checks (run these before trusting the NN as a pricer) ---

def intrinsic_put_value(S, K):
    return np.maximum(np.asarray(K, dtype=float) - np.asarray(S, dtype=float), 0.0)


def count_intrinsic_violations(df, price_col="nn_price", tol=1e-8):
    """American put price must be >= max(K-S, 0). Rows below that are bugs
    or extrapolation failures, not just noise."""
    intrinsic = intrinsic_put_value(df["S0"].to_numpy(), df["K"].to_numpy())
    violation_mask = df[price_col].to_numpy() + tol < intrinsic
    return df[violation_mask]


def put_spot_monotonicity_check(predict_fn, K=100, T=1.0, r=0.05, sigma=0.25,
                                 lo=60, hi=140, n=41):
    """Put price should fall (or stay flat) as spot rises, holding everything
    else fixed. Returns any (spot_i, spot_i+1) pairs that violate this."""
    spots = np.linspace(lo, hi, n)
    prices = [predict_fn(S, K, T, r, sigma) for S in spots]
    violations = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1] + 1e-6:
            violations.append((spots[i - 1], spots[i], prices[i - 1], prices[i]))
    return violations
