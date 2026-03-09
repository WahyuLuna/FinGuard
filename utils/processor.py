"""
utils/processor.py
Feature engineering & normalization pipeline for the ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import streamlit as st


# Features used by the ML engine
NUMERIC_FEATURES = [
    "TransactionAmount",
    "AccountBalance",
    "TransactionDuration",
    "LoginAttempts",
    "CustomerAge",
    "TxHour",
    "TxDayOfWeek",
    "IsWeekend",
    "IsNightTx",
]

CATEGORICAL_FEATURES = ["Channel", "TransactionType", "CustomerOccupation"]


@st.cache_data(show_spinner="Engineering features...")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features on top of the cleaned dataframe.
    Returns an enriched copy.
    """
    df = df.copy()

    # ── Amount-to-balance ratio (spending intensity) ──────────────────────────
    df["AmountBalanceRatio"] = (
        df["TransactionAmount"] / (df["AccountBalance"] + 1)
    ).clip(upper=10)

    # ── Transaction velocity proxy: duration per unit amount ─────────────────
    df["SpeedScore"] = (
        df["TransactionAmount"] / (df["TransactionDuration"] + 1)
    )

    # ── High login flag ───────────────────────────────────────────────────────
    df["HighLoginFlag"] = (df["LoginAttempts"] > 3).astype(int)

    # ── Abnormal duration flags ───────────────────────────────────────────────
    df["VeryFastTx"] = (df["TransactionDuration"] <= 4).astype(int)
    df["VerySlowTx"] = (df["TransactionDuration"] >= 290).astype(int)

    return df


@st.cache_data(show_spinner="Normalizing features...")
def normalize_features(df: pd.DataFrame):
    """
    Apply RobustScaler to numeric features (resistant to extreme outliers).
    Returns:
        X_scaled  – numpy array ready for sklearn models
        feature_cols – list of column names used
        scaler    – fitted RobustScaler instance (for inverse transform)
    """
    df = engineer_features(df)

    extra_numeric = ["AmountBalanceRatio", "SpeedScore",
                     "HighLoginFlag", "VeryFastTx", "VerySlowTx"]
    feature_cols  = NUMERIC_FEATURES + extra_numeric

    # Encode categoricals → integer codes, appended to feature matrix
    encoders = {}
    encoded_cols = []
    for cat in CATEGORICAL_FEATURES:
        if cat in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[cat].astype(str))
            col_name = f"{cat}_enc"
            df[col_name] = encoded
            encoders[cat] = le
            encoded_cols.append(col_name)

    all_cols  = feature_cols + encoded_cols
    X_raw     = df[all_cols].fillna(0).values

    scaler    = RobustScaler()
    X_scaled  = scaler.fit_transform(X_raw)

    return X_scaled, all_cols, scaler, encoders
