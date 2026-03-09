"""
utils/data_loader.py
Handles data ingestion, cleaning, and type conversion for FinGuard.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

DTYPE_MAP = {
    "TransactionAmount":   "float64",
    "AccountBalance":      "float64",
    "TransactionDuration": "int32",
    "LoginAttempts":       "int32",
    "CustomerAge":         "int32",
}


@st.cache_data(show_spinner="Loading & cleaning dataset...")
def load_and_clean_data(source=None):
    if source is None:
        base   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source = os.path.join(base, "data", "transactions.csv")

    if not os.path.exists(source):
        st.error(f"Data file not found: {source}\nRun: python utils/generate_data.py")
        st.stop()

    df = pd.read_csv(source, low_memory=False)
    df.columns = df.columns.str.strip()

    # Missing values
    df["MerchantID"] = df["MerchantID"].fillna("UNKNOWN")
    df["DeviceID"]   = df["DeviceID"].fillna("UNKNOWN")
    df["IP Address"] = df["IP Address"].fillna("0.0.0.0")
    df["Location"]   = df["Location"].fillna("Unknown")

    # Dedup
    df = df.drop_duplicates(subset=["TransactionID"])

    # Date parsing
    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"], format="%m/%d/%Y %H:%M", errors="coerce"
    )

    # Numeric types
    for col, dtype in DTYPE_MAP.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    df = df.dropna(subset=["TransactionAmount", "AccountBalance",
                            "TransactionDuration", "LoginAttempts"])

    # Temporal features
    df["TxHour"]      = df["TransactionDate"].dt.hour
    df["TxDayOfWeek"] = df["TransactionDate"].dt.dayofweek
    df["TxMonth"]     = df["TransactionDate"].dt.month
    df["IsWeekend"]   = df["TxDayOfWeek"].isin([5, 6]).astype(int)
    df["IsNightTx"]   = df["TxHour"].between(0, 5).astype(int)

    # Preserve raw columns
    df["TransactionAmount_raw"] = df["TransactionAmount"]
    df["AccountBalance_raw"]    = df["AccountBalance"]

    return df.reset_index(drop=True)


def get_summary_stats(df):
    return {
        "total_transactions": len(df),
        "unique_accounts":    df["AccountID"].nunique(),
        "total_volume":       df["TransactionAmount"].sum(),
        "avg_amount":         df["TransactionAmount"].mean(),
        "date_range_start":   df["TransactionDate"].min(),
        "date_range_end":     df["TransactionDate"].max(),
        "channels":           df["Channel"].value_counts().to_dict(),
        "tx_types":           df["TransactionType"].value_counts().to_dict(),
    }
