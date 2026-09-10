"""Offline/online feature parity check.

features.py computes 16 features by walking 189k rows in one pandas pass.
main.py computes the same 16 for a single transaction against history pulled
from SQLite. Two implementations of the same definitions.

If they drift apart, nothing crashes - the model just quietly gets worse,
because it is being served features that do not match what it was trained on.
This is the test that catches that.

Run:  python test_parity.py
"""

import sys

import pandas as pd

from db import SessionLocal
from models import Account, Transaction
from main import TransactionIn, compute_features, load_history

TOLERANCE = 1e-6      # float arithmetic will not be bit-identical across paths
SAMPLE_PER_GROUP = 50

FEATURES = [
    "amount", "card_present", "hour",
    "amount_ratio", "amount_vs_hist_mean", "amount_vs_hist_max",
    "secs_since_last", "txn_count_1h", "txn_count_24h", "distinct_merchants_1h",
    "dist_from_last_km", "implied_speed_kmh", "dist_from_home_km",
    "merchant_seen_before", "merchant_use_count", "distinct_merchants_30d",
]


def online_features(session, txn_id):
    """Score one transaction through the serving path."""
    row = session.get(Transaction, txn_id)
    account = session.get(Account, row.account_id)

    txn = TransactionIn(
        txn_id=row.txn_id,
        account_id=row.account_id,
        merchant_id=row.merchant_id,
        amount=row.amount,
        ts=row.ts,
        lat=row.lat,
        lon=row.lon,
        card_present=bool(row.card_present),
    )
    history = load_history(session, row.account_id, row.ts)
    return compute_features(txn, account.typical_amount,
                            account.home_lat, account.home_lon, history)


def pick_sample(offline):
    """A stratified sample - every fraud pattern plus ordinary traffic.

    Sampling only legitimate rows would pass trivially, because most of their
    velocity and distance features are zero. The fraud rows are where the
    interesting values live, so they are where drift would actually show up.
    """
    chosen = []
    for group in ["card_testing", "impossible_travel", "account_takeover"]:
        rows = offline[offline.fraud_type == group]
        chosen += list(rows.sample(min(SAMPLE_PER_GROUP, len(rows)), random_state=42).txn_id)

    legit = offline[offline.fraud_type.isna()]
    chosen += list(legit.sample(SAMPLE_PER_GROUP, random_state=42).txn_id)

    # An account's first-ever transaction, where every history feature is NaN.
    chosen.append(offline.iloc[0].txn_id)
    return chosen


def main():
    offline = pd.read_csv("features.csv").set_index("txn_id")
    session = SessionLocal()

    sample = pick_sample(offline.reset_index())
    failures = []

    for txn_id in sample:
        online = online_features(session, txn_id)
        expected = offline.loc[txn_id]

        for name in FEATURES:
            want, got = expected[name], online[name]

            # NaN == NaN is False in float arithmetic, so handle it explicitly.
            if pd.isna(want) and pd.isna(got):
                continue
            if pd.isna(want) != pd.isna(got):
                failures.append(f"{txn_id}.{name}: offline={want} online={got}")
                continue
            if abs(float(want) - float(got)) > TOLERANCE:
                failures.append(f"{txn_id}.{name}: offline={want} online={got}")

    session.close()

    checked = len(sample) * len(FEATURES)
    print(f"compared {checked:,} values across {len(sample)} transactions")

    if failures:
        print(f"\nFAIL - {len(failures)} mismatch(es):")
        for f in failures[:20]:
            print("  " + f)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        sys.exit(1)

    print("PASS - offline and online agree on every feature")


if __name__ == "__main__":
    main()
