"""
core/anomaly_logic.py
Rule-based Security Analysis Layer:
  1. Identity Theft  – multi-device / multi-IP per account in a rolling window
  2. Brute Force     – LoginAttempts threshold
  3. Impossible Travel – sequential transactions from geographically distant cities
"""

import pandas as pd
import numpy as np
import streamlit as st


# ── Known city coordinates (for impossible-travel calc) ──────────────────────
CITY_COORDS = {
    "New York": (40.71, -74.01), "Los Angeles": (34.05, -118.24),
    "Chicago": (41.88, -87.63),  "Houston": (29.76, -95.37),
    "Phoenix": (33.45, -112.07), "Philadelphia": (39.95, -75.17),
    "San Antonio": (29.42, -98.49), "San Diego": (32.72, -117.15),
    "Dallas": (32.78, -96.80),   "San Jose": (37.34, -121.89),
    "Austin": (30.27, -97.74),   "Jacksonville": (30.33, -81.66),
    "Fort Worth": (32.75, -97.33), "Columbus": (39.96, -82.99),
    "Charlotte": (35.23, -80.84), "Indianapolis": (39.77, -86.16),
    "San Francisco": (37.77, -122.42), "Seattle": (47.61, -122.33),
    "Denver": (39.74, -104.98),  "Nashville": (36.17, -86.78),
    "Oklahoma City": (35.47, -97.52), "El Paso": (31.76, -106.49),
    "Washington": (38.91, -77.04), "Las Vegas": (36.17, -115.14),
    "Louisville": (38.25, -85.76), "Memphis": (35.15, -90.05),
    "Portland": (45.52, -122.68), "Baltimore": (39.29, -76.61),
    "Milwaukee": (43.04, -87.91), "Albuquerque": (35.08, -106.65),
    "Tucson": (32.22, -110.97),  "Fresno": (36.74, -119.77),
    "Sacramento": (38.58, -121.49), "Kansas City": (39.10, -94.58),
    "Mesa": (33.42, -111.83),    "Atlanta": (33.75, -84.39),
    "Omaha": (41.26, -95.94),    "Colorado Springs": (38.83, -104.82),
    "Raleigh": (35.78, -78.64),  "Miami": (25.77, -80.19),
    # International (injected as anomalies)
    "Tokyo": (35.69, 139.69),    "London": (51.51, -0.13),
    "Lagos": (6.45, 3.40),       "Sydney": (-33.87, 151.21),
    "Moscow": (55.76, 37.62),    "Dubai": (25.20, 55.27),
}

MAX_SPEED_KMH     = 900    # max realistic travel speed (commercial flight)
WINDOW_HOURS      = 1      # rolling window for multi-device / multi-IP check
MAX_DEVICES       = 2      # flag if > this many unique devices per account/window
MAX_IPS           = 3      # flag if > this many unique IPs per account/window


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# ── 1. Identity Theft: Multi-Device / Multi-IP ───────────────────────────────
@st.cache_data(show_spinner="🔍 Analysing device & IP patterns...")
def detect_multi_device_ip(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, look back WINDOW_HOURS on the same AccountID.
    Flag if more than MAX_DEVICES unique DeviceIDs or MAX_IPS unique IPs seen.
    """
    df_sorted = df.sort_values(["AccountID", "TransactionDate"]).copy()
    df_sorted["MultiDeviceFlag"] = False
    df_sorted["MultiIPFlag"]     = False

    for acct, grp in df_sorted.groupby("AccountID"):
        grp = grp.sort_values("TransactionDate")
        dates   = grp["TransactionDate"].values
        devices = grp["DeviceID"].values
        ips     = grp["IP Address"].values
        idxs    = grp.index.values

        window_ns = np.timedelta64(int(WINDOW_HOURS * 3600), "s")

        for i in range(len(grp)):
            mask       = (dates[i] - dates) <= window_ns
            mask[i:]   = mask[i:]  # include current
            uniq_dev   = len(set(devices[mask & (dates <= dates[i])]))
            uniq_ip    = len(set(ips[mask & (dates <= dates[i])]))

            if uniq_dev > MAX_DEVICES:
                df_sorted.at[idxs[i], "MultiDeviceFlag"] = True
            if uniq_ip > MAX_IPS:
                df_sorted.at[idxs[i], "MultiIPFlag"] = True

    return df_sorted[["MultiDeviceFlag", "MultiIPFlag"]]


# ── 2. Brute Force: High Login Attempts ──────────────────────────────────────
def detect_brute_force(df: pd.DataFrame) -> pd.Series:
    """Flag transactions where LoginAttempts > 3."""
    return (df["LoginAttempts"] > 3).rename("HighLoginFlag_Sec")


# ── 3. Impossible Travel ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="🌍 Checking impossible travel patterns...")
def detect_impossible_travel(df: pd.DataFrame) -> pd.Series:
    """
    For consecutive transactions per account, check if required travel speed
    between locations exceeds MAX_SPEED_KMH (impossible by any transport).
    """
    df_sorted = df.sort_values(["AccountID", "TransactionDate"]).copy()
    df_sorted["ImpossibleTravelFlag"] = False

    for acct, grp in df_sorted.groupby("AccountID"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("TransactionDate")
        rows = grp.itertuples()
        prev = next(rows)

        for curr in rows:
            loc1 = getattr(prev, "Location", None)
            loc2 = getattr(curr, "Location", None)

            if loc1 and loc2 and loc1 != loc2:
                c1 = CITY_COORDS.get(loc1)
                c2 = CITY_COORDS.get(loc2)

                if c1 and c2:
                    dist_km   = _haversine(*c1, *c2)
                    t1        = pd.Timestamp(getattr(prev, "TransactionDate"))
                    t2        = pd.Timestamp(getattr(curr, "TransactionDate"))
                    delta_hrs = max((t2 - t1).total_seconds() / 3600, 1e-3)
                    speed     = dist_km / delta_hrs

                    if speed > MAX_SPEED_KMH:
                        df_sorted.at[curr.Index, "ImpossibleTravelFlag"] = True

            prev = curr

    return df_sorted["ImpossibleTravelFlag"]


# ── Master Security Scorer ────────────────────────────────────────────────────
def run_security_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all security checks and returns a unified security DataFrame
    with individual flags + composite SecurityRisk score (0–100).
    """
    # Run checks
    device_ip_flags    = detect_multi_device_ip(df)
    brute_force_flag   = detect_brute_force(df)
    travel_flag        = detect_impossible_travel(df)

    sec_df = pd.DataFrame(index=df.index)
    sec_df["MultiDeviceFlag"]      = device_ip_flags["MultiDeviceFlag"]
    sec_df["MultiIPFlag"]          = device_ip_flags["MultiIPFlag"]
    sec_df["HighLoginFlag_Sec"]    = brute_force_flag.reindex(df.index)
    sec_df["ImpossibleTravelFlag"] = travel_flag.reindex(df.index)

    # Weighted security risk score
    sec_df["SecurityRisk"] = (
        sec_df["MultiDeviceFlag"].astype(int)      * 35 +
        sec_df["MultiIPFlag"].astype(int)           * 25 +
        sec_df["HighLoginFlag_Sec"].astype(int)     * 20 +
        sec_df["ImpossibleTravelFlag"].astype(int)  * 40
    ).clip(0, 100).astype(float)

    # Risk tier label
    def tier(s):
        if s >= 75: return "🔴 HIGH"
        if s >= 45: return "🟡 MEDIUM"
        return "🟢 LOW"

    sec_df["SecurityRiskTier"] = sec_df["SecurityRisk"].apply(tier)

    return sec_df
