"""
Data quality diagnostics for the CLV Prediction project.

These functions only DIAGNOSE issues -- they do not modify the data.
Cleaning decisions are made explicitly in Phase 3, based on what these
functions report.
"""

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtypes, missing counts, and missing % per column."""
    profile = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df) * 100).round(2),
        "n_unique": df.nunique(),
    })
    return profile


def flag_cancellations(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for cancellation invoices (InvoiceNo starts with 'C')."""
    return df["InvoiceNo"].astype(str).str.startswith("C")


def flag_non_product_rows(df: pd.DataFrame) -> pd.Series:
    """
    Flag rows that are not real product sales (e.g. postage, manual
    adjustments, bank charges). These StockCodes are known non-product
    codes in the Online Retail dataset.
    """
    non_product_codes = {"POST", "D", "M", "MANUAL", "BANK CHARGES", "DOT", "CRUK"}
    return df["StockCode"].astype(str).str.upper().isin(non_product_codes)


def quality_report(df: pd.DataFrame) -> dict:
    """
    Compute a full data quality report as a dictionary of summary stats.
    """
    cancellations = flag_cancellations(df)
    non_product = flag_non_product_rows(df)
    negative_qty_not_cancel = (df["Quantity"] < 0) & (~cancellations)
    non_positive_price = df["UnitPrice"] <= 0
    missing_customer = df["CustomerID"].isna()
    duplicate_rows = df.duplicated(keep=False)

    report = {
        "total_rows": len(df),
        "cancellation_rows": int(cancellations.sum()),
        "cancellation_pct": round(cancellations.mean() * 100, 2),
        "non_product_rows": int(non_product.sum()),
        "negative_qty_not_cancellation": int(negative_qty_not_cancel.sum()),
        "non_positive_price_rows": int(non_positive_price.sum()),
        "missing_customer_id_rows": int(missing_customer.sum()),
        "missing_customer_id_pct": round(missing_customer.mean() * 100, 2),
        "duplicate_rows": int(duplicate_rows.sum()),
        "revenue_from_missing_customer": round(
            (df.loc[missing_customer, "Quantity"] * df.loc[missing_customer, "UnitPrice"]).sum(), 2
        ),
    }
    return report
