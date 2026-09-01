from collections import Counter

from db import SessionLocal
from models import Transaction

session = SessionLocal()
acct = "ACC00042"

rows = (session.query(Transaction)
        .filter_by(account_id=acct)
        .order_by(Transaction.ts)
        .all())

print(f"{acct}: {len(rows)} transactions")
print("first:", rows[0].ts, rows[0].merchant_id, f"${rows[0].amount}")
print("last: ", rows[-1].ts, rows[-1].merchant_id, f"${rows[-1].amount}")
print("amounts: min ${:.2f}  max ${:.2f}".format(
    min(r.amount for r in rows), max(r.amount for r in rows)))

print("\nmerchants by visit count:")
for merchant_id, n in Counter(r.merchant_id for r in rows).most_common(10):
    print(f"  {merchant_id}  {n}")

session.close()