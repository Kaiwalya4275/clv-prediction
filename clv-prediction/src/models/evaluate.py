"""
Evaluation metrics for the CLV Prediction project.

Centralizing metrics here ensures the single-stage and two-stage models
are judged identically -- critical for a fair comparison between them.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """
    R2 always increases (or stays flat) as you add features, even
    useless ones. Adjusted R2 penalizes that, so it's the fairer metric
    when comparing models with different numbers of features.
    """
    if n_samples - n_features - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE is undefined when actual value is 0 (division by zero). Since
    43.9% of our target values are exactly 0 (customers who churned),
    standard MAPE cannot be computed over the full set. We report MAPE
    only over the subset with y_true > 0, and separately report what
    fraction of the data that excludes.
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_regression(y_true, y_pred, n_features: int) -> dict:
    """Compute the full metric suite for a regression prediction."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, len(y_true), n_features)
    mape = mape_nonzero(y_true, y_pred)
    pct_nonzero = float((y_true > 0).mean() * 100)

    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4),
        "MAPE_on_nonzero_only": round(mape, 2),
        "pct_nonzero_actuals": round(pct_nonzero, 2),
    }
