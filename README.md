# 🛡️ FinGuard — Anomaly Detection System v1.0

Aplikasi deteksi anomali transaksi perbankan berbasis **Unsupervised Machine Learning** dibangun dengan Streamlit.

---

## 📁 Struktur Project

```
finguard_app/
├── app.py                        # 🏠 Main Dashboard
├── pages/
│   ├── 01_Realtime_Monitor.py    # 📡 Live feed + heatmaps
│   ├── 02_Analytics_Deep_Dive.py # 📊 Cluster explorer + profiling
│   └── 03_Security_Logs.py       # 🔐 Security audit + account timeline
├── core/
│   ├── model_engine.py           # 🤖 Isolation Forest + K-Means
│   └── anomaly_logic.py          # 🔍 Security rules engine
├── utils/
│   ├── data_loader.py            # 📂 Ingestion + cleaning
│   ├── processor.py              # ⚙️  Feature engineering + normalization
│   └── generate_data.py          # 🎲 Synthetic data generator (50k rows)
├── assets/css/style.css          # 🎨 Dark Navy Theme
├── data/transactions.csv         # 📊 Generated dataset
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (50,000 transactions)
python utils/generate_data.py

# 3. Launch the app
streamlit run app.py
```

---

## 🤖 ML Engine

| Model | Purpose | Config |
|-------|---------|--------|
| **Isolation Forest** | Global outlier detection | contamination=5%, n_estimators=200 |
| **K-Means Clustering** | Distance-based anomaly | k=8 clusters, 95th-percentile threshold |
| **Security Rules** | Rule-based fraud signals | Brute force, impossible travel, multi-device |
| **Ensemble Score** | Combined risk 0–100 | IF 50% + KM 25% + Security 25% |

---

## 📊 Pages

### 🏠 Main Dashboard (`app.py`)
- 6 KPI cards (total txn, anomalies, risk tiers, health)
- Risk distribution histogram + donut chart
- Anomaly by channel + monthly timeline
- Top N anomaly alert table

### 📡 Realtime Monitor
- Live transaction feed (simulated, batch scroll)
- Anomaly heatmap (Hour × Day of Week)
- Risk heatmap (Channel × Transaction Type)
- Amount vs Duration scatter colored by risk tier

### 📊 Analytics Deep Dive
- K-Means cluster explorer (IF Risk vs KM Risk scatter)
- Cluster profile table (size, anomaly rate, avg risk)
- Feature distribution: Normal vs Anomaly overlay
- Behavioral profiling by occupation
- Feature importance (anomaly signal strength)

### 🔐 Security Logs
- Security flag KPIs (brute force, travel, multi-device, multi-IP)
- Flag timeline (monthly trend)
- Impossible travel location breakdown
- Account-level security audit table
- Individual account transaction timeline + export CSV

---

## 🎨 Theme
Dark Navy · Space Mono (monospace) · DM Sans (body) · Teal `#64ffda` accents
