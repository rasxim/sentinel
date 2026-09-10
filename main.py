import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from models import Account, Decision, Transaction
from db import SessionLocal

app = FastAPI()

MODEL_PATH = Path(__file__).parent / "model.joblib"
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURE_NAMES = bundle["features"]
print("loaded model with features:", FEATURE_NAMES)

# Thresholds come from the sweep in cost_curve.py, not from taste. 0.5 is a
# statistical midpoint; these are where expected cost is actually lowest given
# $200 per missed fraud and $50 per false decline.
REVIEW_THRESHOLD = 0.05
DECLINE_THRESHOLD = 0.90

ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)
THIRTY_DAYS = timedelta(days=30)
NA = float("nan")


#validating transaction data
class TransactionIn(BaseModel):
    txn_id : str
    account_id : str
    merchant_id : str
    amount : float
    ts : datetime
    lat : float
    lon : float
    card_present : bool


def haversine_km(lat1, lon1, lat2, lon2):
    # Must stay identical to the copy in features.py. test_parity.py is what
    # catches it if these two ever drift apart.
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_history(session, account_id, before_ts):
    """This account's prior transactions, oldest first.

    `ts < before_ts` is the as-of cutoff. The database would happily return
    transactions from after this one; including them would leak the future
    into the features. No lower bound, because amount_vs_hist_max and
    merchant_use_count read all of history offline - bounding it here would
    silently break parity.
    """
    return (session.query(Transaction)
            .filter(Transaction.account_id == account_id,
                    Transaction.ts < before_ts)
            .order_by(Transaction.ts)
            .all())


def compute_features(txn, typical_amount, home_lat, home_lon, history):
    """Online twin of the offline loop in features.py.

    Same 16 features, different implementation: features.py walks 189k rows in
    one pandas pass, this handles a single transaction against a list pulled
    from SQLite. They must agree exactly - test_parity.py asserts it.
    """
    recent_30d = [h for h in history if h.ts >= txn.ts - THIRTY_DAYS]
    recent_24h = [h for h in recent_30d if h.ts >= txn.ts - ONE_DAY]
    recent_1h = [h for h in recent_24h if h.ts >= txn.ts - ONE_HOUR]

    if history:
        last = history[-1]
        prior_amounts = [h.amount for h in history]

        secs_since_last = (txn.ts - last.ts).total_seconds()
        dist_from_last_km = haversine_km(last.lat, last.lon, txn.lat, txn.lon)

        hours_elapsed = secs_since_last / 3600.0
        implied_speed_kmh = dist_from_last_km / hours_elapsed if hours_elapsed > 0 else NA

        amount_vs_hist_mean = txn.amount / (sum(prior_amounts) / len(prior_amounts))
        amount_vs_hist_max = txn.amount / max(prior_amounts)
    else:
        # First transaction this account has ever made. NaN means "unknown",
        # which is honest; 0 would be a claim we cannot support.
        secs_since_last = NA
        dist_from_last_km = NA
        implied_speed_kmh = NA
        amount_vs_hist_mean = NA
        amount_vs_hist_max = NA

    merchant_use_count = sum(1 for h in history if h.merchant_id == txn.merchant_id)

    return {
        "amount": txn.amount,
        "card_present": int(txn.card_present),
        "hour": txn.ts.hour,

        "amount_ratio": txn.amount / typical_amount,
        "amount_vs_hist_mean": amount_vs_hist_mean,
        "amount_vs_hist_max": amount_vs_hist_max,

        "secs_since_last": secs_since_last,
        "txn_count_1h": len(recent_1h),
        "txn_count_24h": len(recent_24h),
        "distinct_merchants_1h": len({h.merchant_id for h in recent_1h}),

        "dist_from_last_km": dist_from_last_km,
        "implied_speed_kmh": implied_speed_kmh,
        "dist_from_home_km": haversine_km(home_lat, home_lon, txn.lat, txn.lon),

        "merchant_seen_before": 1 if merchant_use_count > 0 else 0,
        "merchant_use_count": merchant_use_count,
        "distinct_merchants_30d": len({h.merchant_id for h in recent_30d}),
    }


@app.get("/health")
def root():
    return {"status": "ok"}


@app.post("/score")
def score_transaction(txn: TransactionIn):
    started = perf_counter()
    session = SessionLocal()

    account = session.get(Account, txn.account_id)
    if account is None:
        session.close()
        raise HTTPException(status_code=404, detail=f"account {txn.account_id} not found")

    # Copy the scalars out while the session is alive. commit() expires ORM
    # objects and close() detaches them, so reading account.* after the write
    # raises DetachedInstanceError.
    typical_amount = account.typical_amount
    home_lat, home_lon = account.home_lat, account.home_lon

    history = load_history(session, txn.account_id, txn.ts)
    feats = compute_features(txn, typical_amount, home_lat, home_lon, history)

    # XGBoost matches columns by POSITION, not name. Building the row in
    # FEATURE_NAMES order is what keeps serving aligned with training - get it
    # wrong and you get confident nonsense, with no error and no warning.
    X = pd.DataFrame([[feats[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    fraud_probability = float(MODEL.predict_proba(X)[0, 1])

    if fraud_probability >= DECLINE_THRESHOLD:
        decision = "DECLINE"
    elif fraud_probability >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    record = Decision(
        txn_id=txn.txn_id,
        account_id=txn.account_id,
        amount=txn.amount,
        typical_amount=typical_amount,
        ratio=feats["amount_ratio"],
        fraud_probability=fraud_probability,
        decision=decision,
        ts=txn.ts,
        scored_at=datetime.now(),
    )
    session.add(record)
    session.commit()
    session.close()

    return {
        "decision": decision,
        "fraud_probability": round(fraud_probability, 6),
        "history_used": len(history),
        "latency_ms": round((perf_counter() - started) * 1000, 1),
    }
