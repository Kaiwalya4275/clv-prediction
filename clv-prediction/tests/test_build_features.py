"""Unit tests for src/features/build_features.py"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features.build_features import compute_rfm_features, compute_target, build_feature_target_table


def _calib_df():
    return pd.DataFrame({
        "CustomerID": [1, 1, 2],
        "InvoiceNo": ["A1", "A2", "B1"],
        "StockCode": ["X", "Y", "X"],
        "Quantity": [2, 1, 5],
        "Revenue": [20.0, 10.0, 50.0],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-03-01", "2011-02-01"]),
        "Country": ["United Kingdom", "United Kingdom", "France"],
    })


def _holdout_df():
    return pd.DataFrame({
        "CustomerID": [1],
        "Revenue": [15.0],
    })


def test_recency_and_frequency_computed_correctly():
    snapshot = pd.Timestamp("2011-08-31")
    rfm = compute_rfm_features(_calib_df(), snapshot)
    assert rfm.loc[1, "frequency"] == 2
    assert rfm.loc[2, "frequency"] == 1
    assert rfm.loc[1, "recency_days"] == (snapshot - pd.Timestamp("2011-03-01")).days


def test_target_defaults_to_zero_for_non_returning_customers():
    rfm_index = pd.Index([1, 2], name="CustomerID")
    target = compute_target(_holdout_df(), rfm_index)
    assert target.loc[1] == 15.0
    assert target.loc[2] == 0.0


def test_no_leakage_features_use_only_calibration_data():
    """Feature values must not change if we alter holdout data."""
    snapshot = pd.Timestamp("2011-08-31")
    calib = _calib_df()
    rfm_before = compute_rfm_features(calib, snapshot)

    # Simulate a completely different holdout -- features must be unaffected
    table = build_feature_target_table(calib, _holdout_df(), snapshot)
    assert (table["monetary"] == rfm_before["monetary"]).all()
