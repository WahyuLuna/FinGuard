"""
app.py — FinGuard Anomaly Detection System
Main Dashboard Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="FinGuard | Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ─────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "css", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from utils.data_loader   import load_and_clean_data, get_summary_stats
from utils.processor     import normalize_features
from core.model_engine   import run_isolation_forest, run_kmeans, enrich_dataframe
from core.anomaly_logic  import run_security_analysis

# ── Helper: HTML components ──────────────────────────────────────────────────
def metric_card(label, value, sub="", variant="default"):
    cls = {"danger":"danger","warning":"warning","neutral":"neutral"}.get(variant,"")
    val_cls = cls
    return f"""
    <div class="metric-card {cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value {val_cls}">{value}</div>
      {"<div class='metric-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

def section_header(title, icon=""):
    return f"""
    <div class="section-header">
      <div class="section-dot"></div>
      <h2>{icon} {title}</h2>
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem;'>
      <div style='font-family:"Space Mono",monospace;font-size:1.1rem;
                  color:#64ffda;font-weight:700;letter-spacing:0.05em;'>
        🛡️ FINGUARD
      </div>
      <div style='font-size:0.7rem;color:#8892a4;letter-spacing:0.15em;
                  text-transform:uppercase;margin-top:4px;'>
        Anomaly Detection v1.0
      </div>
    </div>
    <hr style='border-color:rgba(100,255,218,0.12);margin-bottom:1.5rem;'/>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Data Source**")
    data_source = st.radio("", ["Synthetic Dataset (50k)", "Upload CSV"],
                           label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload transactions CSV",
                                          type=["csv"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(100,255,218,0.12);margin:1.2rem 0;'/>",
                unsafe_allow_html=True)

    st.markdown("**⚙️ Model Settings**")
    contamination = st.slider("IF Contamination %", 1, 15, 5, 1,
                               help="Expected anomaly ratio for Isolation Forest")
    n_clusters    = st.slider("K-Means Clusters",   4, 16,  8, 1)

    st.markdown("<hr style='border-color:rgba(100,255,218,0.12);margin:1.2rem 0;'/>",
                unsafe_allow_html=True)

    run_btn = st.button("▶  RUN ANALYSIS", type="primary", use_container_width=True)

    st.markdown("""
    <div style='margin-top:2rem;font-size:0.68rem;color:#4a5568;
                font-family:"Space Mono",monospace;line-height:1.7;'>
      PAGES<br/>
      — Realtime Monitor<br/>
      — Analytics Deep Dive<br/>
      — Security Logs
    </div>""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(source_path=None):
    return load_and_clean_data(source_path)

if uploaded_file:
    import tempfile, pathlib
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uploaded_file.read()); tmp.close()
    df_raw = get_data(tmp.name)
else:
    df_raw = get_data()

# ── Run ML Pipeline ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🤖 Running ML pipeline…")
def run_pipeline(data_hash, _df, contam, k):
    X, cols, scaler, enc = normalize_features(_df)
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    import warnings; warnings.filterwarnings("ignore")

    # Isolation Forest
    ifo = IsolationForest(n_estimators=200, contamination=contam/100,
                          random_state=42, n_jobs=-1)
    ifo.fit(X)
    labels = pd.Series(ifo.predict(X), index=_df.index, name="IF_Label")
    raw    = pd.Series(ifo.score_samples(X), index=_df.index)
    r_min, r_max = raw.min(), raw.max()
    if_risk = pd.Series(100*(1-(raw-r_min)/(r_max-r_min+1e-9)), index=_df.index, name="IF_Risk")

    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10); km.fit(X)
    cluster_ids = pd.Series(km.labels_, index=_df.index, name="KM_Cluster")
    _, dists = pairwise_distances_argmin_min(X, km.cluster_centers_)
    distances = pd.Series(dists, index=_df.index)
    thresh = cluster_ids.map(distances.groupby(cluster_ids).quantile(0.95))
    km_anom = pd.Series(dists > thresh.values, index=_df.index, name="KM_Anomaly")
    d_min, d_max = distances.min(), distances.max()
    km_risk = pd.Series(100*(distances-d_min)/(d_max-d_min+1e-9), index=_df.index, name="KM_Risk")

    sec_df   = run_security_analysis(_df)
    enriched = enrich_dataframe(_df, labels, if_risk, cluster_ids, km_anom, km_risk, sec_df)
    return enriched

data_hash = str(len(df_raw)) + str(contamination) + str(n_clusters)

if "enriched_df" not in st.session_state or run_btn:
    with st.spinner("⏳ Analysing 50,000 transactions…"):
        st.session_state["enriched_df"] = run_pipeline(data_hash, df_raw, contamination, n_clusters)

df = st.session_state["enriched_df"]
stats = get_summary_stats(df_raw)

# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:0.25rem;'>
  <h1 style='font-family:"Space Mono",monospace;font-size:1.6rem;
             color:#e6f1ff;margin:0;letter-spacing:0.03em;'>
    🛡️ FinGuard
  </h1>
  <span style='font-size:0.75rem;color:#8892a4;font-family:"Space Mono",monospace;
               letter-spacing:0.15em;text-transform:uppercase;'>
    Anomaly Detection System
  </span>
</div>
<p style='color:#8892a4;font-size:0.88rem;margin-bottom:1.5rem;'>
  Real-time unsupervised fraud & anomaly detection · Isolation Forest + K-Means + Security Rules
</p>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# KPI CARDS — Row 1
# ════════════════════════════════════════════════════════════════════════════════
total      = len(df)
anomalies  = int(df["IsAnomaly"].sum())
high_risk  = int((df["RiskTier"] == "🔴 HIGH").sum())
med_risk   = int((df["RiskTier"] == "🟡 MEDIUM").sum())
brute_force= int(df["HighLoginFlag_Sec"].sum())
imp_travel = int(df["ImpossibleTravelFlag"].sum())
health_pct = round((1 - anomalies / max(total, 1)) * 100, 2)

c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, "TOTAL TXN",       f"{total:,}",       f"across {df['AccountID'].nunique():,} accounts", "default"),
    (c2, "ANOMALIES",       f"{anomalies:,}",    f"{anomalies/total*100:.1f}% of total",          "danger"),
    (c3, "HIGH RISK",       f"{high_risk:,}",    "combined score ≥ 70",                            "danger"),
    (c4, "MEDIUM RISK",     f"{med_risk:,}",     "combined score 40–70",                           "warning"),
    (c5, "SECURITY FLAGS",  f"{brute_force+imp_travel:,}", f"{brute_force} brute · {imp_travel} travel", "warning"),
    (c6, "SYSTEM HEALTH",   f"{health_pct}%",    "normal transactions",                            "neutral"),
]
for col, label, value, sub, variant in cards:
    with col:
        st.markdown(metric_card(label, value, sub, variant), unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# CHARTS — Row 2
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("RISK DISTRIBUTION OVERVIEW", "📊"), unsafe_allow_html=True)

chart_col1, chart_col2, chart_col3 = st.columns([1.2, 1.2, 1])

# ── Chart 1: Risk Score Distribution ─────────────────────────────────────────
with chart_col1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["CombinedRisk"],
        nbinsx=60,
        marker=dict(
            color=df["CombinedRisk"].values,
            colorscale=[[0,"#1a3a5c"],[0.4,"#64ffda"],[0.7,"#ffb347"],[1,"#ff4b4b"]],
            line=dict(width=0),
        ),
        hovertemplate="Risk: %{x:.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=40, line_dash="dash", line_color="#ffb347", line_width=1,
                  annotation_text="MEDIUM", annotation_font_color="#ffb347",
                  annotation_font_size=10)
    fig.add_vline(x=70, line_dash="dash", line_color="#ff4b4b", line_width=1,
                  annotation_text="HIGH", annotation_font_color="#ff4b4b",
                  annotation_font_size=10)
    fig.update_layout(
        title=dict(text="Combined Risk Score Distribution", font=dict(size=12, color="#8892a4",
                   family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Risk Score (0–100)", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
        yaxis=dict(title="Transactions", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0), height=280,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Chart 2: Anomalies by Channel ────────────────────────────────────────────
with chart_col2:
    channel_anom = df[df["IsAnomaly"]].groupby("Channel").size().reset_index(name="Anomalies")
    channel_total = df.groupby("Channel").size().reset_index(name="Total")
    channel_df = channel_anom.merge(channel_total, on="Channel")
    channel_df["Rate%"] = (channel_df["Anomalies"] / channel_df["Total"] * 100).round(1)
    channel_df = channel_df.sort_values("Anomalies", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=channel_df["Anomalies"], y=channel_df["Channel"],
        orientation="h",
        marker=dict(
            color=channel_df["Rate%"],
            colorscale=[[0,"#1a3a5c"],[0.5,"#64ffda"],[1,"#ff4b4b"]],
            line=dict(width=0),
        ),
        text=channel_df["Rate%"].apply(lambda x: f"{x}%"),
        textposition="outside", textfont=dict(color="#8892a4", size=10),
        hovertemplate="%{y}<br>Anomalies: %{x:,}<br>Rate: %{text}<extra></extra>",
    ))
    fig2.update_layout(
        title=dict(text="Anomalies by Channel", font=dict(size=12, color="#8892a4",
                   family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
        yaxis=dict(color="#8892a4", tickfont=dict(size=10)),
        margin=dict(l=0, r=40, t=40, b=0), height=280,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: Risk Tier Donut ───────────────────────────────────────────────
with chart_col3:
    tier_counts = df["RiskTier"].value_counts()
    labels_map  = {"🔴 HIGH": "HIGH", "🟡 MEDIUM": "MEDIUM", "🟢 LOW": "LOW"}
    colors_map  = {"🔴 HIGH": "#ff4b4b", "🟡 MEDIUM": "#ffb347", "🟢 LOW": "#64ffda"}

    tiers  = [t for t in ["🔴 HIGH","🟡 MEDIUM","🟢 LOW"] if t in tier_counts]
    values = [tier_counts[t] for t in tiers]
    colors = [colors_map[t] for t in tiers]
    lbls   = [labels_map[t] for t in tiers]

    fig3 = go.Figure(go.Pie(
        labels=lbls, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#060d1a", width=2)),
        textinfo="percent", textfont=dict(size=10, color="#e6f1ff",
                                          family="Space Mono"),
        hovertemplate="%{label}<br>%{value:,} transactions<br>%{percent}<extra></extra>",
    ))
    fig3.add_annotation(
        text=f"<b>{total:,}</b>",
        x=0.5, y=0.55, showarrow=False,
        font=dict(size=18, color="#e6f1ff", family="Space Mono"),
    )
    fig3.add_annotation(
        text="TXN", x=0.5, y=0.38, showarrow=False,
        font=dict(size=10, color="#8892a4", family="Space Mono"),
    )
    fig3.update_layout(
        title=dict(text="Risk Tier Breakdown", font=dict(size=12, color="#8892a4",
                   family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#8892a4", size=10, family="DM Sans"),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1),
        margin=dict(l=0, r=0, t=40, b=0), height=280,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# CHARTS — Row 3: Timeline + Transaction Type
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("TEMPORAL & BEHAVIORAL ANALYSIS", "📈"), unsafe_allow_html=True)

tc1, tc2 = st.columns([2, 1])

with tc1:
    # Anomaly timeline by month
    timeline = df.groupby(["TxMonth", "IsAnomaly"]).size().reset_index(name="count")
    months   = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    normal_m = timeline[~timeline["IsAnomaly"]].set_index("TxMonth")["count"]
    anom_m   = timeline[timeline["IsAnomaly"]].set_index("TxMonth")["count"]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=[months[m-1] for m in normal_m.index], y=normal_m.values,
        name="Normal", marker_color="rgba(100,255,218,0.25)",
        marker_line=dict(width=0),
        hovertemplate="%{x}<br>Normal: %{y:,}<extra></extra>",
    ))
    fig4.add_trace(go.Bar(
        x=[months[m-1] for m in anom_m.index], y=anom_m.values,
        name="Anomaly", marker_color="#ff4b4b",
        marker_line=dict(width=0),
        hovertemplate="%{x}<br>Anomalies: %{y:,}<extra></extra>",
    ))
    fig4.update_layout(
        barmode="stack",
        title=dict(text="Monthly Transaction Volume with Anomaly Overlay",
                   font=dict(size=12, color="#8892a4", family="Space Mono"),
                   x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        yaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        legend=dict(font=dict(color="#8892a4", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=260,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig4, use_container_width=True)

with tc2:
    # Anomaly rate by transaction type
    tx_stats = df.groupby("TransactionType").agg(
        Total=("IsAnomaly","count"), Anomalies=("IsAnomaly","sum")
    ).reset_index()
    tx_stats["Rate"] = (tx_stats["Anomalies"] / tx_stats["Total"] * 100).round(1)

    fig5 = go.Figure(go.Bar(
        x=tx_stats["TransactionType"], y=tx_stats["Rate"],
        marker=dict(
            color=tx_stats["Rate"],
            colorscale=[[0,"#1a3a5c"],[0.5,"#64ffda"],[1,"#ff4b4b"]],
            line=dict(width=0),
        ),
        text=tx_stats["Rate"].apply(lambda x: f"{x}%"),
        textposition="outside", textfont=dict(color="#8892a4", size=10),
        hovertemplate="%{x}<br>Anomaly Rate: %{y:.1f}%<extra></extra>",
    ))
    fig5.update_layout(
        title=dict(text="Anomaly Rate by Type",
                   font=dict(size=12, color="#8892a4", family="Space Mono"),
                   x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", tickfont=dict(size=10)),
        yaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.05)",
                   tickfont=dict(size=10), title="Rate (%)"),
        margin=dict(l=0, r=0, t=40, b=10), height=260,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# HIGH-RISK ANOMALY TABLE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("HIGH-RISK ANOMALY ALERTS", "⚠️"), unsafe_allow_html=True)

top_n = st.select_slider("Show top N anomalies", options=[25, 50, 100, 250, 500], value=50)

anom_df = (
    df[df["IsAnomaly"]]
    .sort_values("CombinedRisk", ascending=False)
    .head(top_n)[[
        "TransactionID", "AccountID", "TransactionAmount",
        "TransactionDate", "TransactionType", "Channel",
        "Location", "LoginAttempts", "CombinedRisk", "RiskTier",
        "HighLoginFlag_Sec", "ImpossibleTravelFlag",
    ]]
    .copy()
)

anom_df["TransactionDate"] = anom_df["TransactionDate"].dt.strftime("%Y-%m-%d %H:%M")
anom_df["TransactionAmount"] = anom_df["TransactionAmount"].apply(lambda x: f"${x:,.2f}")
anom_df["CombinedRisk"]      = anom_df["CombinedRisk"].apply(lambda x: f"{x:.1f}")
anom_df.columns = [
    "Tx ID", "Account", "Amount", "Date", "Type", "Channel",
    "Location", "Login Attempts", "Risk Score", "Risk Tier",
    "Brute Force", "Impossible Travel",
]

st.dataframe(
    anom_df,
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "Risk Score":        st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
        "Brute Force":       st.column_config.CheckboxColumn("Brute Force"),
        "Impossible Travel": st.column_config.CheckboxColumn("Impossible Travel"),
    },
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(100,255,218,0.08);margin-top:3rem;'/>
<div style='text-align:center;font-family:"Space Mono",monospace;
            font-size:0.65rem;color:#2d3748;padding:1rem 0;letter-spacing:0.1em;'>
  FINGUARD v1.0 · ISOLATION FOREST + K-MEANS + SECURITY RULES · UNSUPERVISED ML
</div>
""", unsafe_allow_html=True)
