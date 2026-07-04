"""
Data loading utilities for the CLV Prediction project.

Keeping loading logic here (rather than in the notebook) means every
script/notebook that needs the raw data loads it identically -- same
encoding, same dtypes, same date parsing. This avoids subtle bugs where
two parts of the project parse dates differently.
"""

import pandas as pd


def load_raw_transactions(filepath: str, encoding: str = "latin1") -> pd.DataFrame:
    """
    Load the raw Online Retail transactions file.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.
    encoding : str
        File encoding. The Online Retail dataset requires 'latin1'
        because it contains non-UTF-8 characters (e.g. in product
        descriptions with special characters).

    Returns
    -------
    pd.DataFrame
        Raw transactions with InvoiceDate parsed as datetime.
    """
    df = pd.read_csv(filepath, encoding=encoding)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df
