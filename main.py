import bisect
import math
import threading
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from datetime import datetime, timedelta, timezone
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
DECLINE_THRESHOLD = 0.85

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

    @field_validator("ts")
    @classmethod
    def naive_utc(cls, v):
        # Stored history is naive. Browsers send ISO strings with a trailing "Z",
        # which parse as timezone-aware, and comparing aware to naive raises
        # TypeError. Normalise to naive UTC so both sides compare cleanly.
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v


@dataclass(frozen=True)
class Seen:
    """One past transaction, as the feature code needs it.

    Plain values rather than ORM objects, so nothing here can become detached
    from a closed session.
    """
    txn_id: str
    ts: datetime
    amount: float
    merchant_id: str
    lat: float
    lon: float


class LiveHistory:
    """Per-account transaction history held in memory.

    The first time an account is scored, its full history is loaded from
    SQLite. After that, every scored transaction is added here, so a burst
    of made-up transactions builds on itself - the third charge sees the
    first two. Scored transactions never go into the `transactions` table,
    which stays the untouched training set.

    `before()` enforces the same as-of cutoff as the offline pipeline:
    only transactions strictly earlier than the one being scored.
    """

    def __init__(self):
        self._by_account = {}
        self._ids = {}
        # Sync endpoints run in a thread pool, so requests can overlap.
        self._lock = threading.Lock()

    def _warm(self, session, account_id):
        rows = (session.query(Transaction)
                .filter(Transaction.account_id == account_id)
                .order_by(Transaction.ts)
                .all())
        self._by_account[account_id] = [
            Seen(r.txn_id, r.ts, r.amount, r.merchant_id, r.lat, r.lon) for r in rows
        ]
        self._ids[account_id] = {r.txn_id for r in rows}

    def before(self, session, account_id, ts):
        with self._lock:
            if account_id not in self._by_account:
                self._warm(session, account_id)
            hist = self._by_account[account_id]
            cut = bisect.bisect_left(hist, ts, key=lambda h: h.ts)
            return hist[:cut]

    def add(self, txn):
        """Record a scored transaction. Returns False for a txn_id already seen,
        so replaying a real historical row does not count it twice."""
        with self._lock:
            hist = self._by_account.get(txn.account_id)
            if hist is None or txn.txn_id in self._ids[txn.account_id]:
                return False
            seen = Seen(txn.txn_id, txn.ts, txn.amount, txn.merchant_id, txn.lat, txn.lon)
            bisect.insort(hist, seen, key=lambda h: h.ts)
            self._ids[txn.account_id].add(txn.txn_id)
            return True

    def reset(self, account_id=None):
        """Forget live additions. The next request reloads from SQLite."""
        with self._lock:
            if account_id is None:
                self._by_account.clear()
                self._ids.clear()
            else:
                self._by_account.pop(account_id, None)
                self._ids.pop(account_id, None)


LIVE = LiveHistory()


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
    """This account's prior transactions, oldest first, straight from SQLite.

    The serving path uses LiveHistory instead; this is the reference it is
    checked against in test_parity.py.

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

    history = LIVE.before(session, txn.account_id, txn.ts)
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

    added = LIVE.add(txn)

    # NaN is not valid JSON, so unknown values (an account's first-ever
    # transaction) go out as null.
    features_out = {
        k: None if isinstance(v, float) and math.isnan(v)
        else round(v, 4) if isinstance(v, float) else v
        for k, v in feats.items()
    }

    return {
        "decision": decision,
        "fraud_probability": round(fraud_probability, 6),
        "history_used": len(history),
        "added_to_history": added,
        "features": features_out,
        "latency_ms": round((perf_counter() - started) * 1000, 1),
    }


@app.post("/reset")
def reset(account_id: str | None = None):
    """Drop live additions so a demo can start again from the stored history."""
    LIVE.reset(account_id)
    return {"reset": account_id or "all accounts"}
