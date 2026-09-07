import math

import pandas as pd

from db import engine

NA = float("nan")

ONE_HOUR = pd.Timedelta(hours=1)
ONE_DAY = pd.Timedelta(days=1)
THIRTY_DAYS = pd.Timedelta(days=30)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


df = pd.read_sql("""
    SELECT t.*, a.typical_amount, a.home_lat, a.home_lon
    FROM transactions t
    JOIN accounts a ON a.account_id = t.account_id
    ORDER BY t.account_id, t.ts
""", engine, parse_dates=["ts"])

history = {}
rows = []

for txn in df.itertuples():
    hist = history.setdefault(txn.account_id, [])

    # Everything below reads ONLY from `hist`, which holds this account's
    # prior transactions and nothing from `txn` itself. `txn` is appended
    # at the very bottom, after the features are already built.

    # Narrowing windows: each filters the one before it, so we scan the
    # full history once and the smaller windows only scan what's left.
    recent_30d = [h for h in hist if h.ts >= txn.ts - THIRTY_DAYS]
    recent_24h = [h for h in recent_30d if h.ts >= txn.ts - ONE_DAY]
    recent_1h = [h for h in recent_24h if h.ts >= txn.ts - ONE_HOUR]

    if hist:
        last = hist[-1]
        prior_amounts = [h.amount for h in hist]

        secs_since_last = (txn.ts - last.ts).total_seconds()
        dist_from_last_km = haversine_km(last.lat, last.lon, txn.lat, txn.lon)

        hours_elapsed = secs_since_last / 3600.0
        implied_speed_kmh = dist_from_last_km / hours_elapsed if hours_elapsed > 0 else NA

        amount_vs_hist_mean = txn.amount / (sum(prior_amounts) / len(prior_amounts))
        amount_vs_hist_max = txn.amount / max(prior_amounts)
    else:
        # First transaction this account has ever made: genuinely unknown,
        # not zero. NaN says "no information", 0 would be a claim.
        secs_since_last = NA
        dist_from_last_km = NA
        implied_speed_kmh = NA
        amount_vs_hist_mean = NA
        amount_vs_hist_max = NA

    merchant_use_count = sum(1 for h in hist if h.merchant_id == txn.merchant_id)

    feats = {
        # identifiers and label - NOT features, dropped before training
        "txn_id": txn.txn_id,
        "account_id": txn.account_id,
        "ts": txn.ts,
        "is_fraud": txn.is_fraud,
        "fraud_type": txn.fraud_type,

        # raw from the transaction itself - known at decision time
        "amount": txn.amount,
        "card_present": txn.card_present,
        "hour": txn.ts.hour,

        # amount vs this account's normal -> account_takeover
        "amount_ratio": txn.amount / txn.typical_amount,
        "amount_vs_hist_mean": amount_vs_hist_mean,
        "amount_vs_hist_max": amount_vs_hist_max,

        # velocity -> card_testing
        "secs_since_last": secs_since_last,
        "txn_count_1h": len(recent_1h),
        "txn_count_24h": len(recent_24h),
        "distinct_merchants_1h": len({h.merchant_id for h in recent_1h}),

        # location -> impossible_travel
        "dist_from_last_km": dist_from_last_km,
        "implied_speed_kmh": implied_speed_kmh,
        "dist_from_home_km": haversine_km(txn.home_lat, txn.home_lon, txn.lat, txn.lon),

        # merchant familiarity -> account_takeover
        "merchant_seen_before": 1 if merchant_use_count > 0 else 0,
        "merchant_use_count": merchant_use_count,
        "distinct_merchants_30d": len({h.merchant_id for h in recent_30d}),
    }
    rows.append(feats)

    # Only now does this transaction become part of the history.
    hist.append(txn)

features = pd.DataFrame(rows)
features.to_csv("features.csv", index=False)

print("shape:", features.shape)
print("fraud rows:", int(features["is_fraud"].sum()))
print(features.head())
