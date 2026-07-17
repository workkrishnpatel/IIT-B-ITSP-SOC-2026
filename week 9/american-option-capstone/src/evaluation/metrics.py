"""Regression-style metrics comparing any model's price to the binomial
benchmark price. Use several metrics together -- each sees a different
kind of error (PDF: 'do not report only training loss')."""

import numpy as np


def pricing_metrics(y_true, y_pred, eps=1e-8) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_bias": float(np.mean(err)),
        "median_abs_error": float(np.median(np.abs(err))),
        "mean_relative_error": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps))),
    }
