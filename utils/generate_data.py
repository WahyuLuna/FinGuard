"""
generate_data.py
Generates a synthetic dataset of 50,000 transactions based on the FinGuard schema.
Run once: python utils/generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N = 50_000

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "San Francisco", "Seattle", "Denver", "Nashville",
    "Oklahoma City", "El Paso", "Washington", "Las Vegas", "Louisville",
    "Memphis", "Portland", "Baltimore", "Milwaukee", "Albuquerque",
    "Tucson", "Fresno", "Sacramento", "Kansas City", "Mesa",
    "Atlanta", "Omaha", "Colorado Springs", "Raleigh", "Miami",
]

OCCUPATIONS = ["Doctor", "Engineer", "Teacher", "Student", "Manager",
                "Retired", "Self-Employed", "Government", "IT Professional", "Other"]

CHANNELS = ["ATM", "Online", "Mobile", "Branch", "POS"]

TRANSACTION_TYPES = ["Debit", "Credit", "Transfer", "Payment"]

MERCHANTS = [f"M{str(i).zfill(3)}" for i in range(1, 101)]
DEVICES   = [f"D{str(i).zfill(6)}" for i in range(1, 1001)]
ACCOUNTS  = [f"AC{str(i).zfill(5)}" for i in range(1, 5001)]


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def generate_date():
    start = datetime(2023, 1, 1)
    end   = datetime(2023, 12, 31)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def build_normal_record(tx_id):
    account   = random.choice(ACCOUNTS)
    amount    = round(np.random.lognormal(mean=5.0, sigma=1.2), 2)
    date      = generate_date()
    tx_type   = random.choice(TRANSACTION_TYPES)
    location  = random.choice(CITIES)
    device    = random.choice(DEVICES)
    ip        = random_ip()
    merchant  = random.choice(MERCHANTS) if tx_type in ["Debit", "Payment"] else ""
    channel   = random.choice(CHANNELS)
    age       = random.randint(18, 80)
    occ       = random.choice(OCCUPATIONS)
    duration  = random.randint(10, 300)
    logins    = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.55, 0.20, 0.15, 0.05, 0.04, 0.01])
    balance   = round(random.uniform(500, 50000), 2)

    return {
        "TransactionID": f"TX{str(tx_id).zfill(6)}",
        "AccountID": account,
        "TransactionAmount": amount,
        "TransactionDate": date.strftime("%m/%d/%Y %H:%M"),
        "TransactionType": tx_type,
        "Location": location,
        "DeviceID": device,
        "IP Address": ip,
        "MerchantID": merchant,
        "Channel": channel,
        "CustomerAge": age,
        "CustomerOccupation": occ,
        "TransactionDuration": duration,
        "LoginAttempts": int(logins),
        "AccountBalance": balance,
    }


def inject_anomalies(df, anomaly_ratio=0.05):
    """Inject ~5% anomalous patterns into the dataset."""
    n_anomalies = int(len(df) * anomaly_ratio)
    indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    for idx in indices:
        pattern = random.choice(["high_amount", "brute_force", "fast_tx",
                                  "multi_device", "impossible_travel"])

        if pattern == "high_amount":
            df.at[idx, "TransactionAmount"] = round(random.uniform(8000, 50000), 2)

        elif pattern == "brute_force":
            df.at[idx, "LoginAttempts"] = random.randint(5, 15)

        elif pattern == "fast_tx":
            df.at[idx, "TransactionDuration"] = random.randint(1, 4)

        elif pattern == "multi_device":
            # Same account, new random device & IP
            df.at[idx, "DeviceID"] = f"D{str(random.randint(9000, 9999)).zfill(6)}"
            df.at[idx, "IP Address"] = random_ip()

        elif pattern == "impossible_travel":
            df.at[idx, "Location"] = random.choice(
                ["Tokyo", "London", "Lagos", "Sydney", "Moscow", "Dubai"]
            )

    return df


if __name__ == "__main__":
    print("⏳ Generating 50,000 transaction records...")
    records = [build_normal_record(i + 1) for i in range(N)]
    df = pd.DataFrame(records)
    df = inject_anomalies(df)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset saved → {out_path}  ({len(df):,} rows)")
