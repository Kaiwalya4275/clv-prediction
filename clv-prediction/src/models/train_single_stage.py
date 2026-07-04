"""
Single-stage regression baseline for CLV prediction.

This treats the problem as one regression task: predict future_revenue
directly, including the zeros. We know from Phase 5 that 43.9% of the
target is exactly zero -- this baseline exists specifically so we can
show, with numbers, why that hurts a single-stage model and why the
two-stage approach (train_two_stage.py) is justified.

Target is log1p-transformed before fitting, since revenue is heavily
right-skewed (Phase 4 finding) -- this keeps large-spend outliers from
dominating the loss function, and predictions are inverse-transformed
(expm1) before scoring so metrics are reported in real £ terms.
"""

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .evaluate import evaluate_regression


def get_candidate_models(random_seed: int = 42) -> dict:
    """Return the set of algorithms we compare, wrapped for log1p targets."""
    base_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_seed, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=random_seed, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=random_seed, verbose=-1
        ),
    }
    return {
        name: TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
        for name, model in base_models.items()
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series, models: dict,
                           n_splits: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """
    Manually run K-fold CV for each model, averaging our full metric
    suite across folds (rather than a single train/test split, which
    could be lucky/unlucky given we only have ~3300 customers).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    results = []

    for name, model in models.items():
        fold_metrics = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = np.clip(preds, 0, None)  # revenue can't be negative

            fold_metrics.append(evaluate_regression(y_val.values, preds, n_features=X.shape[1]))

        avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_metrics["model"] = name
        results.append(avg_metrics)

    results_df = pd.DataFrame(results).set_index("model")
    return results_df.sort_values("RMSE")
