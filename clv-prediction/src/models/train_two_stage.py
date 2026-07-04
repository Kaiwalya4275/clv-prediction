"""
Two-stage ("hurdle") model for zero-inflated CLV prediction.

Stage 1 (classifier): P(customer generates any revenue in the holdout window)
Stage 2 (regressor):  E[revenue | customer generates revenue], trained only
                       on the non-zero subset, log1p-transformed as before.

Final prediction: P(return) * E[revenue | return]

This directly addresses the 43.9% zero-inflation found in Phase 5,
which a single-stage regressor has to model as pure noise.
"""

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier, XGBRegressor

from .evaluate import evaluate_regression


def get_stage1_classifiers(random_seed: int = 42) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=random_seed, n_jobs=-1
        ),
        "XGBoostClassifier": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=random_seed, verbosity=0, eval_metric="logloss"
        ),
    }


def get_stage2_regressor(random_seed: int = 42):
    """Best-performing single-stage algorithm, reused for stage 2."""
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=random_seed, verbosity=0
    )
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def cross_validate_stage1(X: pd.DataFrame, y_binary: pd.Series, classifiers: dict,
                           n_splits: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """Compare classifiers for the 'will this customer return?' sub-problem."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    results = []

    for name, clf in classifiers.items():
        accs, precs, recs, aucs = [], [], [], []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]

            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            proba = clf.predict_proba(X_val)[:, 1]

            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds))
            recs.append(recall_score(y_val, preds))
            aucs.append(roc_auc_score(y_val, proba))

        results.append({
            "model": name,
            "Accuracy": np.mean(accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "ROC_AUC": np.mean(aucs),
        })

    return pd.DataFrame(results).set_index("model").sort_values("ROC_AUC", ascending=False)


def cross_validate_two_stage(X: pd.DataFrame, y_revenue: pd.Series,
                              classifier, n_splits: int = 5, random_seed: int = 42) -> dict:
    """
    Run the full two-stage pipeline under K-fold CV:
    fit stage 1 + stage 2 on train fold, combine predictions on val fold,
    evaluate combined prediction against true future_revenue.
    """
    y_binary = (y_revenue > 0).astype(int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_rev_train, y_rev_val = y_revenue.iloc[train_idx], y_revenue.iloc[val_idx]
        y_bin_train = y_binary.iloc[train_idx]

        # Stage 1: probability of returning
        clf = classifier
        clf.fit(X_train, y_bin_train)
        p_return = clf.predict_proba(X_val)[:, 1]

        # Stage 2: regressor trained ONLY on customers who returned
        returned_mask = y_bin_train == 1
        stage2 = get_stage2_regressor(random_seed)
        stage2.fit(X_train.loc[returned_mask], y_rev_train.loc[returned_mask])
        expected_spend = stage2.predict(X_val)
        expected_spend = np.clip(expected_spend, 0, None)

        # Combine
        combined_pred = p_return * expected_spend
        fold_metrics.append(evaluate_regression(y_rev_val.values, combined_pred, n_features=X.shape[1]))

    avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
    return avg_metrics
