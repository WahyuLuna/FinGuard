"""
pages/03_Security_Logs.py
Security audit log: identity theft, brute force, impossible travel deep-dives.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="FinGuard | Security Logs", page_icon="🔐", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "css", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader  import load_and_clean_data
from utils.processor    import normalize_features
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

@st.cache_data(show_spinner="Loading security data…")
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <h1 style='font-family:"Space Mono",monospace;font-size:1.4rem;color:#e6f1ff;margin:0;'>
    🔐 Security Logs
  </h1>
  <p style='color:#8892a4;font-size:0.85rem;margin-top:4px;'>
    Identity theft · Brute force detection · Impossible travel · Account-level audit
  </p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔒 Security Filters")
    flag_filter = st.multiselect(
        "Show flag types",
        ["Brute Force","Impossible Travel","Multi Device","Multi IP"],
        default=["Brute Force","Impossible Travel","Multi Device","Multi IP"],
    )
    date_start = st.date_input("From date", value=df["TransactionDate"].min().date())
    date_end   = st.date_input("To date",   value=df["TransactionDate"].max().date())
    min_risk   = st.slider("Minimum Security Risk", 0, 100, 0)
    export_btn = st.button("📥 Export Flagged CSV", use_container_width=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
brute      = int(df["HighLoginFlag_Sec"].sum())
impossible = int(df["ImpossibleTravelFlag"].sum())
multi_dev  = int(df["MultiDeviceFlag"].sum())
multi_ip   = int(df["MultiIPFlag"].sum())
sec_high   = int((df["SecurityRisk"] >= 75).sum())

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(metric_card("BRUTE FORCE",      f"{brute:,}",      f"LoginAttempts > 3",            "danger"),  unsafe_allow_html=True)
with c2: st.markdown(metric_card("IMPOSSIBLE TRAVEL",f"{impossible:,}", "speed > 900 km/h",              "danger"),  unsafe_allow_html=True)
with c3: st.markdown(metric_card("MULTI-DEVICE",     f"{multi_dev:,}",  "> 2 devices / 1h window",       "warning"), unsafe_allow_html=True)
with c4: st.markdown(metric_card("MULTI-IP",         f"{multi_ip:,}",   "> 3 IPs / 1h window",           "warning"), unsafe_allow_html=True)
with c5: st.markdown(metric_card("HIGH SEC RISK",    f"{sec_high:,}",   "security score ≥ 75",           "danger"),  unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: Security Flag Summary Charts
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("SECURITY FLAG OVERVIEW", "🚨"), unsafe_allow_html=True)

ch1, ch2 = st.columns(2)

with ch1:
    # Flag counts over time (monthly)
    monthly_flags = pd.DataFrame({
        "Month":        df["TxMonth"],
        "BruteForce":   df["HighLoginFlag_Sec"].astype(int),
        "ImpTravel":    df["ImpossibleTravelFlag"].astype(int),
    }).groupby("Month").sum().reset_index()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[months[m-1] for m in monthly_flags["Month"]], y=monthly_flags["BruteForce"],
        name="Brute Force", mode="lines+markers",
        line=dict(color="#ff4b4b", width=2),
        marker=dict(size=6, color="#ff4b4b"),
        hovertemplate="%{x}<br>Brute Force: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[months[m-1] for m in monthly_flags["Month"]], y=monthly_flags["ImpTravel"],
        name="Impossible Travel", mode="lines+markers",
        line=dict(color="#ffb347", width=2),
        marker=dict(size=6, color="#ffb347"),
        hovertemplate="%{x}<br>Impossible Travel: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Security Flags Over Time (Monthly)",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        yaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10),
                   title="Flag Count"),
        legend=dict(font=dict(color="#8892a4", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    # Brute force login attempts distribution
    bf_df    = df[df["HighLoginFlag_Sec"]]
    login_counts = bf_df["LoginAttempts"].value_counts().sort_index().reset_index()
    login_counts.columns = ["LoginAttempts","Count"]

    fig2 = go.Figure(go.Bar(
        x=login_counts["LoginAttempts"], y=login_counts["Count"],
        marker=dict(
            color=login_counts["LoginAttempts"],
            colorscale=[[0,"#ffb347"],[1,"#ff4b4b"]],
            line=dict(width=0),
        ),
        hovertemplate="Attempts: %{x}<br>Transactions: %{y}<extra></extra>",
    ))
    fig2.update_layout(
        title=dict(text="Brute Force — Login Attempts Distribution",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Login Attempts", color="#4a5568", tickfont=dict(size=10)),
        yaxis=dict(title="Count", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0), height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2: Impossible Travel Deep Dive
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("IMPOSSIBLE TRAVEL INCIDENTS", "✈️"), unsafe_allow_html=True)

travel_df = df[df["ImpossibleTravelFlag"]].copy()

if len(travel_df) > 0:
    tr1, tr2 = st.columns([2,1])

    with tr1:
        # Location frequency bar
        loc_counts = travel_df["Location"].value_counts().head(15).reset_index()
        loc_counts.columns = ["Location","Count"]
        loc_counts = loc_counts.sort_values("Count", ascending=True)

        fig3 = go.Figure(go.Bar(
            x=loc_counts["Count"], y=loc_counts["Location"],
            orientation="h",
            marker=dict(color="#ffb347", opacity=0.8, line=dict(width=0)),
            hovertemplate="%{y}<br>Incidents: %{x}<extra></extra>",
        ))
        fig3.update_layout(
            title=dict(text="Impossible Travel Incidents by Location",
                       font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#4a5568", tickfont=dict(size=10)),
            yaxis=dict(color="#8892a4", tickfont=dict(size=10)),
            margin=dict(l=0, r=0, t=40, b=0), height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tr2:
        st.markdown("<div style='font-size:0.75rem;color:#8892a4;font-family:Space Mono;margin-bottom:0.5rem;letter-spacing:0.1em;'>TOP FLAGGED ACCOUNTS</div>", unsafe_allow_html=True)
        top_accts = (travel_df.groupby("AccountID")
                     .agg(Incidents=("TransactionID","count"),
                          AvgAmount=("TransactionAmount","mean"),
                          Locations=("Location","nunique"))
                     .sort_values("Incidents", ascending=False)
                     .head(10).reset_index())
        top_accts["AvgAmount"] = top_accts["AvgAmount"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_accts, use_container_width=True, hide_index=True, height=310)
else:
    st.info("No impossible travel incidents found in current dataset.")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3: Account-Level Security Audit
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("ACCOUNT SECURITY AUDIT", "🔍"), unsafe_allow_html=True)

# Build account risk summary
acct_summary = df.groupby("AccountID").agg(
    TotalTxn         = ("TransactionID","count"),
    Anomalies        = ("IsAnomaly","sum"),
    MaxSecurityRisk  = ("SecurityRisk","max"),
    AvgCombinedRisk  = ("CombinedRisk","mean"),
    BruteForceEvents = ("HighLoginFlag_Sec","sum"),
    TravelEvents     = ("ImpossibleTravelFlag","sum"),
    UniqueDevices    = ("DeviceID","nunique"),
    UniqueIPs        = ("IP Address","nunique"),
    UniqueLocations  = ("Location","nunique"),
    TotalVolume      = ("TransactionAmount","sum"),
).reset_index()

acct_summary["AnomalyRate%"]     = (acct_summary["Anomalies"]/acct_summary["TotalTxn"]*100).round(1)
acct_summary["AvgCombinedRisk"]  = acct_summary["AvgCombinedRisk"].round(1)
acct_summary["MaxSecurityRisk"]  = acct_summary["MaxSecurityRisk"].round(1)
acct_summary["TotalVolume"]      = acct_summary["TotalVolume"].round(0).astype(int)

# Filter: only accounts with at least one security event
sec_accts = acct_summary[
    (acct_summary["BruteForceEvents"] > 0) |
    (acct_summary["TravelEvents"]     > 0) |
    (acct_summary["MaxSecurityRisk"]  >= min_risk)
].sort_values("MaxSecurityRisk", ascending=False)

st.markdown(f"<div style='font-size:0.78rem;color:#8892a4;margin-bottom:0.75rem;'>"
            f"Showing <b style='color:#64ffda;'>{len(sec_accts):,}</b> accounts with security flags"
            f"</div>", unsafe_allow_html=True)

# Export
if export_btn and len(sec_accts) > 0:
    csv = sec_accts.to_csv(index=False)
    st.download_button(
        "📥 Download CSV", csv, "finguard_security_flags.csv", "text/csv",
        use_container_width=True,
    )

st.dataframe(
    sec_accts.head(200),
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "MaxSecurityRisk":  st.column_config.ProgressColumn("Max Sec Risk", min_value=0, max_value=100, format="%.1f"),
        "AvgCombinedRisk":  st.column_config.ProgressColumn("Avg Risk",     min_value=0, max_value=100, format="%.1f"),
        "AnomalyRate%":     st.column_config.NumberColumn("Anom%", format="%.1f%%"),
        "TotalVolume":      st.column_config.NumberColumn("Volume ($)", format="$%d"),
        "BruteForceEvents": st.column_config.NumberColumn("BruteForce 🔑"),
        "TravelEvents":     st.column_config.NumberColumn("Travel ✈️"),
    },
)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: Account Deep Dive
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("ACCOUNT TRANSACTION TIMELINE", "🧵"), unsafe_allow_html=True)

top_flagged = sec_accts.head(20)["AccountID"].tolist() if len(sec_accts) > 0 else df["AccountID"].unique()[:20].tolist()
sel_account = st.selectbox("Select account to inspect", top_flagged)

acct_df = df[df["AccountID"] == sel_account].sort_values("TransactionDate").copy()

if len(acct_df) > 0:
    info1, info2, info3, info4 = st.columns(4)
    with info1: st.markdown(metric_card("TRANSACTIONS",  f"{len(acct_df)}",  "",                               "neutral"), unsafe_allow_html=True)
    with info2: st.markdown(metric_card("ANOMALIES",     f"{acct_df['IsAnomaly'].sum()}", "",                  "danger"),  unsafe_allow_html=True)
    with info3: st.markdown(metric_card("UNIQUE DEVICES",f"{acct_df['DeviceID'].nunique()}", "",               "warning"), unsafe_allow_html=True)
    with info4: st.markdown(metric_card("UNIQUE IPs",    f"{acct_df['IP Address'].nunique()}", "",             "warning"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Timeline chart
    fig4 = go.Figure()
    normal_a = acct_df[~acct_df["IsAnomaly"]]
    anom_a   = acct_df[acct_df["IsAnomaly"]]

    fig4.add_trace(go.Scatter(
        x=normal_a["TransactionDate"], y=normal_a["TransactionAmount"],
        mode="markers", name="Normal",
        marker=dict(color="rgba(100,255,218,0.6)", size=8, symbol="circle"),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra>Normal</extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=anom_a["TransactionDate"], y=anom_a["TransactionAmount"],
        mode="markers", name="⚠ Anomaly",
        marker=dict(color="#ff4b4b", size=12, symbol="star",
                    line=dict(color="#ff4b4b", width=1)),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<br>Risk: %{customdata:.1f}<extra>ANOMALY</extra>",
        customdata=anom_a["CombinedRisk"].values,
    ))

    fig4.update_layout(
        title=dict(text=f"Transaction Timeline — {sel_account}",
                   font=dict(size=12, color="#8892a4", family="Space Mono"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a5568", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        yaxis=dict(title="Amount ($)", color="#4a5568",
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
        legend=dict(font=dict(color="#8892a4", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=300,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Raw transaction log
    with st.expander(f"📋 Full transaction log for {sel_account}"):
        log_df = acct_df[[
            "TransactionID","TransactionDate","TransactionAmount",
            "TransactionType","Channel","Location","DeviceID",
            "IP Address","LoginAttempts","CombinedRisk","RiskTier",
            "HighLoginFlag_Sec","ImpossibleTravelFlag",
        ]].copy()
        log_df["TransactionDate"]   = log_df["TransactionDate"].dt.strftime("%Y-%m-%d %H:%M")
        log_df["TransactionAmount"] = log_df["TransactionAmount"].apply(lambda x: f"${x:,.2f}")
        log_df["CombinedRisk"]      = log_df["CombinedRisk"].apply(lambda x: f"{x:.1f}")
        st.dataframe(log_df, use_container_width=True, hide_index=True,
                     column_config={
                         "HighLoginFlag_Sec":    st.column_config.CheckboxColumn("BruteForce"),
                         "ImpossibleTravelFlag": st.column_config.CheckboxColumn("ImpTravel"),
                     })
