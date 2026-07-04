"""Unit tests for src/data/clean_data.py"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.clean_data import (
    drop_cancellations, drop_non_product_rows, filter_valid_quantity_price,
    drop_missing_customer_id, drop_duplicate_rows, add_revenue_column
)


def _sample_df():
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536379", "536366", "536366"],
        "StockCode": ["85123A", "D", "POST", "85123A"],
        "Description": ["ITEM A", "Discount", "POSTAGE", "ITEM A"],
        "Quantity": [6, -1, 1, 6],
        "InvoiceDate": pd.to_datetime(["2011-01-01"] * 4),
        "UnitPrice": [2.55, 27.5, 18.0, 2.55],
        "CustomerID": [17850.0, 17850.0, None, 17850.0],
        "Country": ["United Kingdom"] * 4,
    })


def test_drop_cancellations_removes_c_invoices():
    df = _sample_df()
    out = drop_cancellations(df)
    assert not out["InvoiceNo"].astype(str).str.startswith("C").any()
    assert len(out) == 3


def test_drop_non_product_rows_removes_known_codes():
    df = _sample_df()
    out = drop_non_product_rows(df)
    assert "POST" not in out["StockCode"].values
    assert "D" not in out["StockCode"].values


def test_filter_valid_quantity_price_keeps_positive_only():
    df = _sample_df()
    out = filter_valid_quantity_price(df)
    assert (out["Quantity"] > 0).all()
    assert (out["UnitPrice"] > 0).all()


def test_drop_missing_customer_id_removes_nulls():
    df = _sample_df()
    out = drop_missing_customer_id(df)
    assert out["CustomerID"].isna().sum() == 0
    assert len(out) == 3


def test_drop_duplicate_rows_removes_exact_dupes():
    df = pd.concat([_sample_df(), _sample_df().iloc[[0]]], ignore_index=True)
    out = drop_duplicate_rows(df)
    assert len(out) == len(_sample_df())


def test_add_revenue_column_computes_correctly():
    df = _sample_df()
    out = add_revenue_column(df)
    assert (out["Revenue"] == out["Quantity"] * out["UnitPrice"]).all()
