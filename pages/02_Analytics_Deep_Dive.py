"""
pages/02_Analytics_Deep_Dive.py
Deep analytics: cluster explorer, feature importance, distribution breakdowns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="FinGuard | Analytics", page_icon="📊", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "css", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader  import load_and_clean_data
from utils.processor    import normalize_features, engineer_features
from core.model_engine  import enrich_dataframe
from core.anomaly_logic import run_security_analysis

def section_header(title, icon=""):
    return f"""<div class="section-header">
      <div class="section-dot"></div><h2>{icon} {title}</h2></div>"""

def metric_card(label, value, sub="", variant="default"):
    cls = {"danger":"danger","warning":"warning","neutral":"neutral"}.get(variant,"")
    return f"""<div class="metric-card {cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
      {"<div class='metric-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

@st.cache_data(show_spinner="Running ML pipeline…")
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
    return enriched, X, cols

df, X_scaled, feature_cols = get_enriched()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <h1 style='font-family:"Space Mono",monospace;font-size:1.4rem;color:#e6f1ff;margin:0;'>
    📊 Analytics Deep Dive
  </h1>
  <p style='color:#8892a4;font-size:0.85rem;margin-top:4px;'>
    Cluster explorer · Feature distributions · Behavioral profiling · ML model insights
  </p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Analysis Filters")
    selected_cluster = st.selectbox("Focus Cluster", ["All"] + [f"Cluster {i}" for i in range(8)])
    selected_occ     = st.multiselect("Occupation", df["CustomerOccupation"].unique().tolist(),
                                       default=df["CustomerOccupation"].unique().tolist())
    age_range        = st.slider("Customer Age Range", 18, 80, (18, 80))
    amount_range     = st.slider("Transaction Amount ($)",
                                  int(df["TransactionAmount"].min()),
                                  min(int(df["TransactionAmount"].max()), 50000),
                                  (0, 10000))

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df.copy()
if selected_cluster != "All":
    cid = int(selected_cluster.split()[-1])
    fdf = fdf[fdf["KM_Cluster"] == cid]
if selected_occ:
    fdf = fdf[fdf["CustomerOccupation"].isin(selected_occ)]
fdf = fdf[(fdf["CustomerAge"] >= age_range[0]) & (fdf["CustomerAge"] <= age_range[1])]
fdf = fdf[(fdf["TransactionAmount"] >= amount_range[0]) & (fdf["TransactionAmount"] <= amount_range[1])]

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
anom_rate = fdf["IsAnomaly"].mean()*100
with c1: st.markdown(metric_card("FILTERED TXN",  f"{len(fdf):,}",     f"{len(fdf)/len(df)*100:.1f}% of total","neutral"), unsafe_allow_html=True)
with c2: st.markdown(metric_card("ANOMALY RATE",  f"{anom_rate:.1f}%", f"{fdf['IsAnomaly'].sum():,} flagged",  "danger"),  unsafe_allow_html=True)
with c3: st.markdown(metric_card("AVG AMOUNT",    f"${fdf['TransactionAmount'].mean():,.0f}", "mean transaction", "default"), unsafe_allow_html=True)
with c4: st.markdown(metric_card("AVG RISK",      f"{fdf['CombinedRisk'].mean():.1f}",       "combined score",   "warning"), unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: Cluster Explorer
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("K-MEANS CLUSTER EXPLORER", "🔵"), unsafe_allow_html=True)

cl1, cl2 = st.columns([2, 1])

with cl1:
    # Scatter: IF Risk vs KM Risk, colored by cluster
    sample = fdf.sample(min(4000, len(fdf)), random_state=42)
    colors = px.colors.qualitative.Set1[:8]

    fig = go.Figure()
    for cid in sorted(sample["KM_Cluster"].unique()):
        sub = sample[sample["KM_Cluster"] == cid]
        color = colors[cid % len(colors)]
        fig.add_trace(go.Scatter(
            x=sub["IF_Risk"], y=sub["KM_Risk"],
            mode="markers",
            name=f"C{cid}",
            marker=dict(color=color, size=4, opacity=0.55, line=dict(width=0)),
            hovertemplate=(
                f"Cluster {cid}<br>"
                "IF Risk: %{x:.1f}<br>"
                "KM Risk: %{y:.1f}<br>"
                "Amount: $%{customdata:.0f}<extra></extra>"
            ),
            customdata=sub["TransactionAmount"].values,
        ))

    # Highlight anomalies
    anom = sample[sample["IsAnomaly"]]
    fig.add_trace(go.Scatter(
        x=anom["IF_Risk"], y=anom["KM_Risk"], mode="markers",
        name="⚠ Anomaly",
        marker=dict(color="rgba(255,75,75,0.8)", size=6,
                    symbol="x", line=dict(color="#ff4b4b", width=1)),
        hovertemplate="ANOMALY<br>IF: %{x:.1f} KM: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="IF Risk vs KM Risk · Colored by Cluster",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Isolation Forest Risk", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        yaxis=dict(title="K-Means Distance Risk", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        legend=dict(font=dict(color="#8892a4", size=9), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=40), height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

with cl2:
    # Cluster profile table
    cluster_profile = df.groupby("KM_Cluster").agg(
        Size        = ("TransactionID","count"),
        Anomalies   = ("IsAnomaly","sum"),
        AvgRisk     = ("CombinedRisk","mean"),
        AvgAmount   = ("TransactionAmount","mean"),
        AvgLogins   = ("LoginAttempts","mean"),
    ).reset_index()
    cluster_profile["AnomalyRate%"] = (cluster_profile["Anomalies"]/cluster_profile["Size"]*100).round(1)
    cluster_profile["AvgRisk"]      = cluster_profile["AvgRisk"].round(1)
    cluster_profile["AvgAmount"]    = cluster_profile["AvgAmount"].round(0).astype(int)
    cluster_profile["AvgLogins"]    = cluster_profile["AvgLogins"].round(2)
    cluster_profile = cluster_profile.rename(columns={"KM_Cluster":"Cluster"})

    st.markdown("<div style='font-size:0.75rem;color:#8892a4;font-family:Space Mono;margin-bottom:0.5rem;letter-spacing:0.1em;'>CLUSTER PROFILES</div>", unsafe_allow_html=True)
    st.dataframe(
        cluster_profile[["Cluster","Size","AnomalyRate%","AvgRisk","AvgAmount"]],
        use_container_width=True, hide_index=True, height=340,
        column_config={
            "AvgRisk": st.column_config.ProgressColumn("Avg Risk", min_value=0, max_value=100, format="%.1f"),
            "AnomalyRate%": st.column_config.NumberColumn("Anom %", format="%.1f%%"),
        },
    )

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2: Feature Distributions
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("FEATURE DISTRIBUTION · NORMAL vs ANOMALY", "📉"), unsafe_allow_html=True)

feat_choice = st.selectbox(
    "Select feature to compare",
    ["TransactionAmount","TransactionDuration","LoginAttempts",
     "AccountBalance","CustomerAge","TxHour"],
    label_visibility="collapsed",
)

normal_vals = fdf[~fdf["IsAnomaly"]][feat_choice].dropna()
anom_vals   = fdf[fdf["IsAnomaly"]][feat_choice].dropna()

fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=normal_vals, name="Normal",
    marker_color="rgba(100,255,218,0.35)",
    marker_line=dict(width=0), nbinsx=50,
    hovertemplate="Value: %{x}<br>Count: %{y}<extra>Normal</extra>",
))
fig2.add_trace(go.Histogram(
    x=anom_vals, name="Anomaly",
    marker_color="rgba(255,75,75,0.6)",
    marker_line=dict(width=0), nbinsx=50,
    hovertemplate="Value: %{x}<br>Count: %{y}<extra>Anomaly</extra>",
))
fig2.update_layout(
    barmode="overlay",
    title=dict(text=f"{feat_choice} Distribution — Normal vs Anomaly",
               font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title=feat_choice, color="#4a5568",
               gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
    yaxis=dict(title="Count", color="#4a5568",
               gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
    legend=dict(font=dict(color="#8892a4", size=10), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=0, r=0, t=40, b=0), height=280,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Stats comparison ──────────────────────────────────────────────────────────
sc1, sc2, sc3 = st.columns(3)
stats_pairs = [
    ("Mean (Normal)",   f"{normal_vals.mean():.2f}",  f"{anom_vals.mean():.2f}",   "Mean (Anomaly)"),
    ("Median (Normal)", f"{normal_vals.median():.2f}", f"{anom_vals.median():.2f}", "Median (Anomaly)"),
    ("Std (Normal)",    f"{normal_vals.std():.2f}",    f"{anom_vals.std():.2f}",    "Std (Anomaly)"),
]
for col, (ln, vn, va, la) in zip([sc1,sc2,sc3], stats_pairs):
    with col:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(100,255,218,0.1);
                    border-radius:8px;padding:0.8rem 1rem;text-align:center;'>
          <div style='font-size:0.65rem;color:#8892a4;font-family:Space Mono;
                      text-transform:uppercase;letter-spacing:0.1em;'>{ln}</div>
          <div style='font-size:1.1rem;color:#64ffda;font-family:Space Mono;font-weight:700;'>{vn}</div>
          <div style='font-size:0.65rem;color:#ff4b4b;font-family:Space Mono;margin-top:4px;'>{la}: {va}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3: Behavioral Profiling
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("BEHAVIORAL PROFILING BY OCCUPATION", "👤"), unsafe_allow_html=True)

bp1, bp2 = st.columns(2)

with bp1:
    occ_stats = df.groupby("CustomerOccupation").agg(
        TotalTxn    = ("TransactionID","count"),
        Anomalies   = ("IsAnomaly","sum"),
        AvgAmount   = ("TransactionAmount","mean"),
        AvgRisk     = ("CombinedRisk","mean"),
    ).reset_index()
    occ_stats["AnomalyRate"] = (occ_stats["Anomalies"]/occ_stats["TotalTxn"]*100).round(2)
    occ_stats = occ_stats.sort_values("AnomalyRate", ascending=True)

    fig3 = go.Figure(go.Bar(
        x=occ_stats["AnomalyRate"], y=occ_stats["CustomerOccupation"],
        orientation="h",
        marker=dict(
            color=occ_stats["AnomalyRate"],
            colorscale=[[0,"#1a3a5c"],[0.5,"#64ffda"],[1,"#ff4b4b"]],
            line=dict(width=0),
        ),
        text=occ_stats["AnomalyRate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="#8892a4", size=9),
        hovertemplate="%{y}<br>Anomaly Rate: %{x:.2f}%<extra></extra>",
    ))
    fig3.update_layout(
        title=dict(text="Anomaly Rate by Occupation",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=10), title="Anomaly Rate (%)"),
        yaxis=dict(color="#8892a4", tickfont=dict(size=10)),
        margin=dict(l=0, r=50, t=40, b=0), height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

with bp2:
    # Box plot: risk score by occupation
    box_sample = df.sample(min(5000, len(df)), random_state=42)
    fig4 = go.Figure()
    for occ in sorted(box_sample["CustomerOccupation"].unique()):
        vals = box_sample[box_sample["CustomerOccupation"]==occ]["CombinedRisk"]
        fig4.add_trace(go.Box(
            y=vals, name=occ,
            marker_color="#64ffda",
            line_color="rgba(100,255,218,0.6)",
            fillcolor="rgba(100,255,218,0.08)",
            boxmean=True,
            hovertemplate=f"{occ}<br>Risk: %{{y:.1f}}<extra></extra>",
        ))
    fig4.update_layout(
        title=dict(text="Risk Score Distribution by Occupation",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#8892a4", tickfont=dict(size=9), tickangle=-30),
        yaxis=dict(title="Combined Risk Score", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=40), height=300,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: Feature Importance (pseudo, based on anomaly mean delta)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("FEATURE IMPORTANCE · ANOMALY SIGNAL STRENGTH", "🧬"), unsafe_allow_html=True)

num_features = ["TransactionAmount","TransactionDuration","LoginAttempts",
                "AccountBalance","CustomerAge","TxHour","AmountBalanceRatio","SpeedScore"]

from utils.processor import engineer_features
df_eng = engineer_features(df)

importance = []
for feat in num_features:
    if feat not in df_eng.columns: continue
    mn = df_eng[~df_eng["IsAnomaly"]][feat].mean()
    ma = df_eng[df_eng["IsAnomaly"]][feat].mean()
    sd = df_eng[feat].std()
    delta = abs(ma - mn) / (sd + 1e-9)
    importance.append({"Feature": feat, "Signal": round(delta, 3),
                        "Normal Mean": round(mn,2), "Anomaly Mean": round(ma,2)})

imp_df = pd.DataFrame(importance).sort_values("Signal", ascending=True)

fig5 = go.Figure(go.Bar(
    x=imp_df["Signal"], y=imp_df["Feature"],
    orientation="h",
    marker=dict(
        color=imp_df["Signal"],
        colorscale=[[0,"#1a3a5c"],[0.5,"#64ffda"],[1,"#ff4b4b"]],
        line=dict(width=0),
    ),
    text=imp_df["Signal"].apply(lambda x: f"{x:.3f}"),
    textposition="outside", textfont=dict(color="#8892a4", size=9),
    hovertemplate="%{y}<br>Signal Strength: %{x:.3f}<extra></extra>",
))
fig5.update_layout(
    title=dict(text="Anomaly Signal Strength (|Δmean| / σ) per Feature",
               font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Signal Strength", color="#4a5568",
               gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
    yaxis=dict(color="#8892a4", tickfont=dict(size=10)),
    margin=dict(l=0, r=60, t=40, b=0), height=300,
)
st.plotly_chart(fig5, use_container_width=True)
