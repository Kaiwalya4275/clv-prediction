"""
Feature engineering for the CLV Prediction project.

CRITICAL RULE: every feature function here must only ever be called on
the CALIBRATION period dataframe. Passing holdout-period data into any
of these functions would leak future information into the model.
"""

import pandas as pd


def compute_rfm_features(calib_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute core RFM features per customer from the calibration period.

    Parameters
    ----------
    calib_df : pd.DataFrame
        Transactions restricted to the calibration period ONLY.
    snapshot_date : pd.Timestamp
        The date the calibration period ends (used to compute Recency
        and tenure relative to "today" in this simulated scenario).
    """
    grouped = calib_df.groupby("CustomerID")

    rfm = grouped.agg(
        last_purchase_date=("InvoiceDate", "max"),
        first_purchase_date=("InvoiceDate", "min"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
        total_quantity=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
    )

    rfm["recency_days"] = (snapshot_date - rfm["last_purchase_date"]).dt.days
    rfm["tenure_days"] = (snapshot_date - rfm["first_purchase_date"]).dt.days
    rfm = rfm.drop(columns=["last_purchase_date", "first_purchase_date"])

    return rfm


def compute_additional_features(calib_df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features motivated by EDA findings:
    is_one_time_buyer, is_uk, avg_order_value.
    """
    rfm = rfm.copy()

    # Average order value: distinguishes "few big orders" from "many small orders"
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

    # One-time buyer flag: ~25%+ of customers fell into this bucket in EDA
    rfm["is_one_time_buyer"] = (rfm["frequency"] == 1).astype(int)

    # Dominant country per customer (mode) -> simplified to is_uk given 82.8% UK concentration
    dominant_country = calib_df.groupby("CustomerID")["Country"].agg(
        lambda x: x.mode().iloc[0]
    )
    rfm["is_uk"] = (dominant_country == "United Kingdom").astype(int)

    return rfm


def compute_target(holdout_df: pd.DataFrame, customer_index: pd.Index) -> pd.Series:
    """
    Compute the prediction target: total revenue per customer during the
    holdout period. Customers with no holdout purchases correctly get 0
    (this is a real, meaningful outcome -- not missing data).
    """
    holdout_revenue = holdout_df.groupby("CustomerID")["Revenue"].sum()
    target = holdout_revenue.reindex(customer_index, fill_value=0.0)
    target.name = "future_revenue"
    return target


def build_feature_target_table(
    calib_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Full pipeline: build RFM + engineered features from calibration data,
    and attach the future_revenue target from holdout data.

    Only customers present in the calibration period are included --
    customers appearing for the first time in the holdout period are a
    cold-start problem and out of scope here.
    """
    rfm = compute_rfm_features(calib_df, snapshot_date)
    rfm = compute_additional_features(calib_df, rfm)
    target = compute_target(holdout_df, rfm.index)

    table = rfm.join(target)
    return table
