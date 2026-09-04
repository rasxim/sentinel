from sqlalchemy import func

from db import SessionLocal
from models import Transaction, Merchant

session = SessionLocal()

rows = (session.query(Transaction.account_id,
                      func.count(func.distinct(Merchant.city)).label("cities"))
        .join(Merchant, Merchant.merchant_id == Transaction.merchant_id)
        .group_by(Transaction.account_id)
        .all())

multi = sum(1 for _, n in rows if n > 1)
print(f"{multi} of {len(rows)} accounts transacted in more than one city")

session.close()