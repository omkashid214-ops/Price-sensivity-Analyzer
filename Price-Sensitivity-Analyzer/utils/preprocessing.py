"""
preprocessing.py
-----------------
Data loading, cleaning, and feature-engineering utilities shared by the
training script and the Streamlit app.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "Price",
    "Marketing_Spend",
    "Competitor_Price",
    "Discount",
    "Season",
    "Holiday",
    "Category",
]
TARGET_COLUMN = "Demand"
CATEGORICAL_COLUMNS = ["Season", "Category"]


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame as read from disk.
    """
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and enforce sane value ranges.

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame with no nulls and non-negative numeric fields.
    """
    df = df.copy()
    df = df.dropna()
    numeric_cols = ["Price", "Cost_Price", "Competitor_Price", "Discount",
                     "Marketing_Spend", "Demand"]
    for col in numeric_cols:
        if col in df.columns:
            df = df[df[col] >= 0]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def encode_categoricals(
    df: pd.DataFrame, encoders: dict | None = None
) -> Tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns.

    Args:
        df: DataFrame containing the categorical columns.
        encoders: Optional dict of pre-fit LabelEncoders (used at inference
            time to keep encodings consistent with training).

    Returns:
        A tuple of (encoded DataFrame, dict of fitted LabelEncoders).
    """
    df = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if col in encoders:
            le = encoders[col]
            # handle unseen categories gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


def build_feature_matrix(
    df: pd.DataFrame, encoders: dict | None = None
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Build the model-ready feature matrix and target vector.

    Args:
        df: Cleaned raw DataFrame containing FEATURE_COLUMNS and TARGET_COLUMN.
        encoders: Optional pre-fit encoders (for inference).

    Returns:
        Tuple of (X features, y target, fitted encoders dict).
    """
    df = df.copy()
    encoded_df, encoders = encode_categoricals(df[FEATURE_COLUMNS], encoders)
    X = encoded_df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    return X, y, encoders


def compute_revenue_profit(
    price: np.ndarray, demand: np.ndarray, cost_price: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute revenue and profit given price, predicted demand, and cost.

    Args:
        price: Array of unit prices.
        demand: Array of predicted units sold.
        cost_price: Array of unit cost prices.

    Returns:
        Tuple of (revenue array, profit array).
    """
    revenue = price * demand
    profit = revenue - (cost_price * demand)
    return revenue, profit
