"""
pages/01_Realtime_Monitor.py
Live transaction feed with real-time anomaly flagging & heatmaps.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="FinGuard | Realtime Monitor", page_icon="📡", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "css", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader  import load_and_clean_data
from utils.processor    import normalize_features
from core.model_engine  import enrich_dataframe
from core.anomaly_logic import run_security_analysis

# ── Helper ────────────────────────────────────────────────────────────────────
def section_header(title, icon=""):
    return f"""<div class="section-header">
      <div class="section-dot"></div><h2>{icon} {title}</h2>
    </div>"""

def metric_card(label, value, sub="", variant="default"):
    cls = {"danger":"danger","warning":"warning","neutral":"neutral"}.get(variant,"")
    return f"""<div class="metric-card {cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
      {"<div class='metric-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

# ── Load enriched data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def get_enriched():
    df = load_and_clean_data()
    X, cols, scaler, enc = normalize_features(df)
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    import warnings; warnings.filterwarnings("ignore")

    ifo = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    ifo.fit(X)
    labels  = pd.Series(ifo.predict(X), index=df.index)
    raw     = pd.Series(ifo.score_samples(X), index=df.index)
    r_min, r_max = raw.min(), raw.max()
    if_risk = pd.Series(100*(1-(raw-r_min)/(r_max-r_min+1e-9)), index=df.index, name="IF_Risk")

    km = KMeans(n_clusters=8, random_state=42, n_init=10); km.fit(X)
    cluster_ids = pd.Series(km.labels_, index=df.index, name="KM_Cluster")
    _, dists    = pairwise_distances_argmin_min(X, km.cluster_centers_)
    distances   = pd.Series(dists, index=df.index)
    thresh      = cluster_ids.map(distances.groupby(cluster_ids).quantile(0.95))
    km_anom     = pd.Series(dists > thresh.values, index=df.index, name="KM_Anomaly")
    d_min, d_max = distances.min(), distances.max()
    km_risk     = pd.Series(100*(distances-d_min)/(d_max-d_min+1e-9), index=df.index, name="KM_Risk")

    sec_df   = run_security_analysis(df)
    enriched = enrich_dataframe(df, labels, if_risk, cluster_ids, km_anom, km_risk, sec_df)
    return enriched

df = get_enriched()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <h1 style='font-family:"Space Mono",monospace;font-size:1.4rem;
             color:#e6f1ff;margin:0;'>📡 Realtime Monitor</h1>
  <p style='color:#8892a4;font-size:0.85rem;margin-top:4px;'>
    Simulated live transaction feed · auto-refresh every 5s · anomaly flagging in real-time
  </p>
</div>""", unsafe_allow_html=True)

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Monitor Settings")
    auto_refresh  = st.toggle("Auto Refresh (5s)", value=False)
    show_only_anom= st.toggle("Show Anomalies Only", value=False)
    risk_filter   = st.select_slider("Min Risk Score", options=[0,10,20,30,40,50,60,70,80,90], value=0)
    channel_filter= st.multiselect("Channels", df["Channel"].unique().tolist(),
                                    default=df["Channel"].unique().tolist())
    tx_type_filter= st.multiselect("Transaction Types", df["TransactionType"].unique().tolist(),
                                    default=df["TransactionType"].unique().tolist())

# ── KPI Row ────────────────────────────────────────────────────────────────────
recent = df.sort_values("TransactionDate").tail(500)
anom_recent = recent["IsAnomaly"].sum()
high_recent = (recent["RiskTier"] == "🔴 HIGH").sum()
avg_risk    = recent["CombinedRisk"].mean()

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(metric_card("LAST 500 TXN", "500", "most recent batch", "neutral"), unsafe_allow_html=True)
with c2: st.markdown(metric_card("ANOMALIES", f"{anom_recent}", f"{anom_recent/5:.1f}% rate", "danger"), unsafe_allow_html=True)
with c3: st.markdown(metric_card("HIGH RISK", f"{high_recent}", "score ≥ 70", "danger"), unsafe_allow_html=True)
with c4: st.markdown(metric_card("AVG RISK SCORE", f"{avg_risk:.1f}", "0–100 scale", "warning"), unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Live Feed Simulation ──────────────────────────────────────────────────────
st.markdown(section_header("LIVE TRANSACTION FEED", "🔴"), unsafe_allow_html=True)

# Filter
feed_df = df.copy()
if show_only_anom:    feed_df = feed_df[feed_df["IsAnomaly"]]
if risk_filter > 0:   feed_df = feed_df[feed_df["CombinedRisk"] >= risk_filter]
if channel_filter:    feed_df = feed_df[feed_df["Channel"].isin(channel_filter)]
if tx_type_filter:    feed_df = feed_df[feed_df["TransactionType"].isin(tx_type_filter)]

# Simulate "live" by sampling last N rows with a random offset stored in session_state
if "feed_offset" not in st.session_state:
    st.session_state["feed_offset"] = 0

if auto_refresh:
    time.sleep(0.1)
    st.session_state["feed_offset"] = (st.session_state["feed_offset"] + 20) % max(len(feed_df)-100, 1)
    st.rerun()

offset   = st.session_state["feed_offset"]
live_df  = feed_df.sort_values("TransactionDate").iloc[offset:offset+100].copy()

live_df["Status"] = live_df["RiskTier"]
live_df["TransactionDate"] = live_df["TransactionDate"].dt.strftime("%H:%M:%S")
live_df["TransactionAmount"] = live_df["TransactionAmount"].apply(lambda x: f"${x:,.2f}")
live_df["CombinedRisk"]  = live_df["CombinedRisk"].apply(lambda x: f"{x:.1f}")

display_cols = ["TransactionID","AccountID","TransactionAmount","TransactionDate",
                "TransactionType","Channel","Location","CombinedRisk","Status"]
live_display = live_df[display_cols].rename(columns={
    "TransactionID":"Tx ID","AccountID":"Account","TransactionAmount":"Amount",
    "TransactionDate":"Time","TransactionType":"Type",
    "CombinedRisk":"Risk Score","Status":"Risk Tier",
})

st.dataframe(
    live_display, use_container_width=True, hide_index=True, height=350,
    column_config={
        "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
    },
)

col_refresh, col_next = st.columns([1, 5])
with col_refresh:
    if st.button("⟳ Next Batch"):
        st.session_state["feed_offset"] = (offset + 100) % max(len(feed_df)-100, 1)
        st.rerun()

st.markdown("<br/>", unsafe_allow_html=True)

# ── Heatmap: Hour × DayOfWeek ─────────────────────────────────────────────────
st.markdown(section_header("ANOMALY HEATMAP · HOUR vs DAY OF WEEK", "🗓️"), unsafe_allow_html=True)

hm_col1, hm_col2 = st.columns(2)

with hm_col1:
    pivot = (df[df["IsAnomaly"]]
             .groupby(["TxDayOfWeek","TxHour"])
             .size()
             .unstack(fill_value=0))
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot.index = [days[i] for i in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale=[[0,"#060d1a"],[0.3,"#1a3a5c"],[0.6,"#64ffda"],[1,"#ff4b4b"]],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Anomalies: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#8892a4", size=9), thickness=10),
    ))
    fig.update_layout(
        title=dict(text="Anomaly Frequency by Hour & Day",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", tickfont=dict(size=9), tickangle=-45),
        yaxis=dict(color="#8892a4", tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=40), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with hm_col2:
    # Amount heatmap per channel x type
    pivot2 = (df.groupby(["Channel","TransactionType"])["CombinedRisk"]
              .mean().unstack(fill_value=0).round(1))

    fig2 = go.Figure(go.Heatmap(
        z=pivot2.values, x=pivot2.columns.tolist(), y=pivot2.index.tolist(),
        colorscale=[[0,"#060d1a"],[0.4,"#1a3a5c"],[0.7,"#64ffda"],[1,"#ff4b4b"]],
        text=pivot2.values.round(1),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#e6f1ff"),
        hovertemplate="Channel: %{y}<br>Type: %{x}<br>Avg Risk: %{z:.1f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#8892a4", size=9), thickness=10),
    ))
    fig2.update_layout(
        title=dict(text="Avg Risk Score · Channel × Transaction Type",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#8892a4", tickfont=dict(size=10)),
        yaxis=dict(color="#8892a4", tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=10), height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Scatter: Amount vs Duration colored by Risk ───────────────────────────────
st.markdown(section_header("TRANSACTION SCATTER · AMOUNT vs DURATION", "⚡"), unsafe_allow_html=True)

sample = df.sample(min(3000, len(df)), random_state=42).copy()
fig3 = go.Figure()

for tier, color in [("🟢 LOW","#64ffda"),("🟡 MEDIUM","#ffb347"),("🔴 HIGH","#ff4b4b")]:
    sub = sample[sample["RiskTier"] == tier]
    if len(sub) == 0: continue
    fig3.add_trace(go.Scatter(
        x=sub["TransactionDuration"], y=sub["TransactionAmount"],
        mode="markers",
        name=tier.split()[-1],
        marker=dict(color=color, size=4, opacity=0.6,
                    line=dict(width=0)),
        hovertemplate=(
            "Account: %{customdata[0]}<br>"
            "Amount: $%{y:,.2f}<br>"
            "Duration: %{x}s<br>"
            "Risk: %{customdata[1]:.1f}<extra></extra>"
        ),
        customdata=sub[["AccountID","CombinedRisk"]].values,
    ))

fig3.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Transaction Duration (s)", color="#4a5568",
               gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
    yaxis=dict(title="Transaction Amount ($)", color="#4a5568",
               gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
    legend=dict(font=dict(color="#8892a4", size=10), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=0, r=0, t=10, b=0), height=320,
    font=dict(family="DM Sans"),
)
st.plotly_chart(fig3, use_container_width=True)
