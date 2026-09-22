import bisect
import math
import threading
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from sqlalchemy import func
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
import xgboost as xgb

from models import Account, Decision, Merchant, Transaction
from db import SessionLocal

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_PATH = Path(__file__).parent / "model.joblib"
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURE_NAMES = bundle["features"]
BOOSTER = MODEL.get_booster()
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


def explain(X, top=3):
    """The features that moved this score the most, from the model itself.

    XGBoost's pred_contribs returns each feature's exact SHAP contribution to
    the score in log-odds; positive raised the fraud risk, negative lowered it.
    The contributions plus the bias term sum to the model's raw output, so these
    are the real reasons for this prediction, not a separate approximation.
    """
    contribs = BOOSTER.predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    order = sorted(range(len(FEATURE_NAMES)), key=lambda i: abs(contribs[i]), reverse=True)
    return [{"feature": FEATURE_NAMES[i], "contribution": round(float(contribs[i]), 3)}
            for i in order[:top]]


def km(lat1, lon1, lat2, lon2):
    return round(haversine_km(lat1, lon1, lat2, lon2))


def merchant_out(m, home_lat, home_lon, visits=None):
    out = {"merchant_id": m.merchant_id, "name": m.name, "category": m.category,
           "city": m.city, "lat": m.lat, "lon": m.lon,
           "km_from_home": km(home_lat, home_lon, m.lat, m.lon)}
    if visits is not None:
        out["visits"] = visits
    return out


_DEMO_ACCOUNTS = None


def demo_accounts(session):
    """One clean account per city for the demo, chosen once and cached.

    "Clean" means no planted fraud in its history, so every score a visitor
    sees is the model reacting to what they send and nothing else.
    """
    global _DEMO_ACCOUNTS
    if _DEMO_ACCOUNTS is not None:
        return _DEMO_ACCOUNTS

    fraud_ids = {r[0] for r in session.query(Transaction.account_id)
                 .filter(Transaction.is_fraud == True).distinct()}  # noqa: E712
    counts = dict(session.query(Transaction.account_id, func.count())
                  .group_by(Transaction.account_id).all())

    by_city = {}
    for a in session.query(Account).order_by(Account.account_id):
        if a.account_id not in fraud_ids:
            by_city.setdefault(a.home_city, []).append(a)

    picked = []
    for city in sorted(by_city):
        accts = by_city[city]
        chosen = next((a for a in accts if a.account_id == "ACC00042"), None)
        if chosen is None:
            chosen = min(accts, key=lambda a: abs(counts.get(a.account_id, 0) - 100))
        picked.append({"account_id": chosen.account_id, "home_city": chosen.home_city,
                       "typical_amount": chosen.typical_amount,
                       "n_transactions": counts.get(chosen.account_id, 0)})

    _DEMO_ACCOUNTS = picked
    return picked


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def root():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return {"model": "XGBoost", "n_features": len(FEATURE_NAMES),
            "review_threshold": REVIEW_THRESHOLD, "decline_threshold": DECLINE_THRESHOLD}


@app.get("/accounts")
def list_accounts():
    session = SessionLocal()
    try:
        return demo_accounts(session)
    finally:
        session.close()


@app.get("/accounts/{account_id}")
def account_profile(account_id: str):
    """Everything the console needs to show an account and build transactions for it."""
    session = SessionLocal()
    try:
        a = session.get(Account, account_id)
        if a is None:
            raise HTTPException(status_code=404, detail=f"account {account_id} not found")

        txns = (session.query(Transaction)
                .filter(Transaction.account_id == account_id)
                .order_by(Transaction.ts).all())
        merchants = {m.merchant_id: m for m in session.query(Merchant).all()}

        visits = {}
        for t in txns:
            visits[t.merchant_id] = visits.get(t.merchant_id, 0) + 1

        favourites = [merchant_out(merchants[mid], a.home_lat, a.home_lon, n)
                      for mid, n in sorted(visits.items(), key=lambda kv: -kv[1])[:5]]

        local_new = [merchant_out(m, a.home_lat, a.home_lon)
                     for m in sorted(merchants.values(), key=lambda m: m.merchant_id)
                     if m.city == a.home_city and m.merchant_id not in visits][:5]

        elsewhere, seen_cities = [], set()
        for m in sorted(merchants.values(), key=lambda m: m.merchant_id):
            if m.city != a.home_city and m.city not in seen_cities and m.merchant_id not in visits:
                seen_cities.add(m.city)
                elsewhere.append(merchant_out(m, a.home_lat, a.home_lon))
        elsewhere.sort(key=lambda m: m["km_from_home"])

        recent = [{"ts": t.ts.isoformat(), "amount": t.amount,
                   "merchant": merchants[t.merchant_id].name,
                   "category": merchants[t.merchant_id].category,
                   "city": merchants[t.merchant_id].city,
                   "card_present": bool(t.card_present)}
                  for t in reversed(txns[-5:])]

        return {
            "account_id": a.account_id,
            "home_city": a.home_city,
            "home_lat": a.home_lat,
            "home_lon": a.home_lon,
            "typical_amount": a.typical_amount,
            "n_transactions": len(txns),
            "distinct_merchants": len(visits),
            "first_ts": txns[0].ts.isoformat() if txns else None,
            "last_ts": txns[-1].ts.isoformat() if txns else None,
            "favourites": favourites,
            "local_new": local_new,
            "elsewhere": elsewhere,
            "recent": recent,
        }
    finally:
        session.close()


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
    reasons = explain(X)

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
        "reasons": reasons,
        "latency_ms": round((perf_counter() - started) * 1000, 1),
    }


@app.post("/reset")
def reset(account_id: str | None = None):
    """Drop live additions so a demo can start again from the stored history."""
    LIVE.reset(account_id)
    return {"reset": account_id or "all accounts"}
