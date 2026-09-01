import random 
from datetime import datetime, timedelta 

from faker import Faker 
from sqlalchemy import insert 

from db import SessionLocal
from models import Account, Merchant, Transaction

import math

SEED = 42 
random.seed(SEED)
Faker.seed(SEED)
fake = Faker()

CITIES = [
    ("New York",     40.7128,  -74.0060),
    ("Philadelphia", 39.9526,  -75.1652),
    ("Boston",       42.3601,  -71.0589),
    ("Chicago",      41.8781,  -87.6298),
    ("Austin",       30.2672,  -97.7431),
    ("Denver",       39.7392, -104.9903),
    ("Seattle",      47.6062, -122.3321),
    ("Atlanta",      33.7490,  -84.3880),
]

CATEGORIES = ["grocery", "restaurant", "gas", "coffee",
              "pharmacy", "clothing", "electronics", "online_retail"]


N_ACCOUNTS = 2000
N_MERCHANTS = 300
TODAY = datetime(2026, 8, 31)

# Relative likelihood of a purchase in each hour (index 0 = midnight).
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 4, 7, 8, 7, 8, 10,
                11, 9, 8, 8, 9, 11, 12, 10, 7, 5, 3, 2]

# How many purchases an account makes in a day.
DAILY_COUNTS  = [0, 1, 2, 3]
DAILY_WEIGHTS = [0.35, 0.35, 0.20, 0.10]

N_DAYS = 90
FAVORITES = 5
START = TODAY - timedelta(days=N_DAYS)

def make_accounts(n):
    rows=[]
    for i in range(n):
        city, lat, lon = random.choice(CITIES)
        rows.append({
            "account_id": f"ACC{i:05d}",
            "home_city": city,
            "home_lat": lat + random.uniform(-0.08, 0.08),
            "home_lon": lon + random.uniform(-0.08, 0.08),
            "typical_amount": round(random.lognormvariate(3.4, 0.5), 2),
            "opened_at": TODAY - timedelta(days=random.randint(90, 2000))
        }
        )
    return rows

def make_merchants(n):
    rows = []
    for i in range(n):
        city, lat, lon = random.choice(CITIES)
        rows.append({
            "merchant_id": f"MER{i:04d}",
            "name": fake.company(),
            "city": city,
            "category": random.choice(CATEGORIES),
            "lat": lat + random.uniform(-0.12, 0.12),
            "lon": lon + random.uniform(-0.12, 0.12),
        })
    return rows

def make_transactions(accounts, merchants):
    # Group merchants by city so each account shops locally.
    by_city = {}
    for m in merchants:
        by_city.setdefault(m["city"], []).append(m)

    rows = []
    for acct in accounts:
        local = by_city[acct["home_city"]]
        favorites = random.sample(local, min(FAVORITES, len(local)))
        log_typical = math.log(acct["typical_amount"])

        for day in range(N_DAYS):
            count = random.choices(DAILY_COUNTS, weights=DAILY_WEIGHTS)[0]

            for _ in range(count):
                # 80% of the time they go somewhere familiar.
                merchant = (random.choice(favorites)
                            if random.random() < 0.80
                            else random.choice(local))

                hour = random.choices(range(24), weights=HOUR_WEIGHTS)[0]
                ts = START + timedelta(days=day, hours=hour,
                                       minutes=random.randint(0, 59),
                                       seconds=random.randint(0, 59))

                amount = max(1.0, round(random.lognormvariate(log_typical, 0.6), 2))
                card_present = random.random() < 0.70

                if card_present:
                    lat = merchant["lat"] + random.uniform(-0.002, 0.002)
                    lon = merchant["lon"] + random.uniform(-0.002, 0.002)
                else:
                    lat, lon = acct["home_lat"], acct["home_lon"]

                rows.append({
                    "account_id": acct["account_id"],
                    "merchant_id": merchant["merchant_id"],
                    "amount": amount,
                    "ts": ts,
                    "lat": lat,
                    "lon": lon,
                    "card_present": card_present,
                    "is_fraud": False,
                    "fraud_type": None,
                })

    # Sort by time, then number them — so IDs run in chronological order.
    rows.sort(key=lambda r: r["ts"])
    for i, row in enumerate(rows):
        row["txn_id"] = f"TXN{i:08d}"
    return rows

def main():
    session = SessionLocal()

    session.query(Transaction).delete()
    session.query(Merchant).delete()
    session.query(Account).delete()
    session.commit()

    accounts = make_accounts(N_ACCOUNTS)
    merchants = make_merchants(N_MERCHANTS)
    session.execute(insert(Account), accounts)
    session.execute(insert(Merchant), merchants)
    session.commit()

    txns = make_transactions(accounts, merchants)
    for i in range(0, len(txns), 5000):
        session.execute(insert(Transaction), txns[i:i + 5000])
    session.commit()

    print(f"accounts:     {session.query(Account).count()}")
    print(f"merchants:    {session.query(Merchant).count()}")
    print(f"transactions: {session.query(Transaction).count()}")
    session.close()
if __name__ == "__main__":
    main()