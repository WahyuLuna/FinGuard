"""
core/model_engine.py
Unsupervised ML Engine: Isolation Forest + K-Means Clustering
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import warnings
warnings.filterwarnings("ignore")

IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 200
IF_RANDOM_STATE  = 42
KMEANS_K         = 8
KMEANS_RANDOM    = 42


@st.cache_data(show_spinner="🤖 Running Isolation Forest...")
def run_isolation_forest(X_scaled: np.ndarray, df_index):
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    labels = pd.Series(model.predict(X_scaled), index=df_index, name="IF_Label")
    raw    = pd.Series(model.score_samples(X_scaled), index=df_index, name="IF_Score")
    r_min, r_max = raw.min(), raw.max()
    risk = pd.Series(
        100 * (1 - (raw - r_min) / (r_max - r_min + 1e-9)),
        index=df_index, name="IF_Risk",
    )
    return labels, raw, risk


@st.cache_data(show_spinner="🤖 Running K-Means Clustering...")
def run_kmeans(X_scaled: np.ndarray, df_index):
    km = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_RANDOM,
                n_init=10, max_iter=300)
    km.fit(X_scaled)
    cluster_ids = pd.Series(km.labels_, index=df_index, name="KM_Cluster")
    centroids   = km.cluster_centers_
    _, dists    = pairwise_distances_argmin_min(X_scaled, centroids)
    distances   = pd.Series(dists, index=df_index, name="KM_Distance")
    thresh_series = cluster_ids.map(
        distances.groupby(cluster_ids).quantile(0.95)
    )
    km_anomaly = pd.Series(dists > thresh_series.values, index=df_index, name="KM_Anomaly")
    d_min, d_max = distances.min(), distances.max()
    km_risk = pd.Series(
        100 * (distances - d_min) / (d_max - d_min + 1e-9),
        index=df_index, name="KM_Risk",
    )
    return cluster_ids, distances, km_anomaly, km_risk


def compute_combined_risk(if_risk, km_risk, security_risk):
    """
    Ensemble: IF 50% + KM 25% + Security 25%.
    Normalise each signal to 0-100 before blending.
    """
    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9) * 100

    combined = (
        0.50 * _norm(if_risk) +
        0.25 * _norm(km_risk) +
        0.25 * security_risk.reindex(if_risk.index).fillna(0)
    ).clip(0, 100)
    return pd.Series(combined, index=if_risk.index, name="CombinedRisk")


def label_risk_tier(score: float) -> str:
    if score >= 70: return "🔴 HIGH"
    if score >= 40: return "🟡 MEDIUM"
    return "🟢 LOW"


def enrich_dataframe(df, if_labels, if_risk, cluster_ids,
                     km_anomaly, km_risk, security_df):
    df = df.copy()
    df["IF_Label"]   = if_labels
    df["IF_Risk"]    = if_risk.round(2)
    df["KM_Cluster"] = cluster_ids
    df["KM_Anomaly"] = km_anomaly
    df["KM_Risk"]    = km_risk.round(2)

    sec_risk = security_df["SecurityRisk"].reindex(df.index).fillna(0)
    df["SecurityRisk"] = sec_risk.round(2)

    for col in ["MultiDeviceFlag", "MultiIPFlag", "HighLoginFlag_Sec",
                "ImpossibleTravelFlag", "SecurityRiskTier"]:
        if col in security_df.columns:
            default = False if "Flag" in col else "🟢 LOW"
            df[col] = security_df[col].reindex(df.index).fillna(default)

    df["CombinedRisk"] = compute_combined_risk(if_risk, km_risk, sec_risk).round(2)
    df["RiskTier"]     = df["CombinedRisk"].apply(label_risk_tier)

    # Flag as anomaly: top 8% combined risk OR IF flagged OR KM flagged
    threshold = df["CombinedRisk"].quantile(0.92)
    df["IsAnomaly"] = (
        (df["IF_Label"] == -1) |
        df["KM_Anomaly"] |
        (df["CombinedRisk"] >= threshold)
    )
    return df
