"""
Data cleaning pipeline for the CLV Prediction project.

Each cleaning step is a separate function that logs how many rows it
removes. This creates an auditable trail -- important because CLV
numbers are only trustworthy if every exclusion is documented and
justified, not silently applied.

Design decision (confirmed with stakeholder): cancellations are
DROPPED entirely (Option A), not netted against original purchases.
"""

import pandas as pd

NON_PRODUCT_CODES = {"POST", "D", "M", "MANUAL", "BANK CHARGES", "DOT", "CRUK"}


def _log(step: str, before: int, after: int) -> None:
    print(f"[{step}] removed {before - after:,} rows ({before:,} -> {after:,})")


def drop_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = ~df["InvoiceNo"].astype(str).str.startswith("C")
    out = df.loc[mask].copy()
    _log("drop_cancellations", before, len(out))
    return out


def drop_non_product_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = ~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
    out = df.loc[mask].copy()
    _log("drop_non_product_rows", before, len(out))
    return out


def filter_valid_quantity_price(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    out = df.loc[mask].copy()
    _log("filter_valid_quantity_price", before, len(out))
    return out


def drop_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.dropna(subset=["CustomerID"]).copy()
    _log("drop_missing_customer_id", before, len(out))
    return out


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(keep="first").copy()
    _log("drop_duplicate_rows", before, len(out))
    return out


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


def add_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline in order, with row-count logging
    at every step.
    """
    print(f"Starting rows: {len(df):,}\n")
    df = drop_cancellations(df)
    df = drop_non_product_rows(df)
    df = filter_valid_quantity_price(df)
    df = drop_missing_customer_id(df)
    df = drop_duplicate_rows(df)
    df = fix_dtypes(df)
    df = add_revenue_column(df)
    print(f"\nFinal rows: {len(df):,}")
    return df
