import random 
from datetime import datetime, timedelta 

from faker import Faker 
from sqlalchemy import insert 

from db import SessionLocal
from models import Account, Merchant, Transaction

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

def main():
    session = SessionLocal()

    # Wipe first so re-running is safe. Transactions go first —
    # they reference the other two tables.
    session.query(Transaction).delete()
    session.query(Merchant).delete()
    session.query(Account).delete()
    session.commit()

    session.execute(insert(Account), make_accounts(N_ACCOUNTS))
    session.execute(insert(Merchant), make_merchants(N_MERCHANTS))
    session.commit()

    print(f"accounts:  {session.query(Account).count()}")
    print(f"merchants: {session.query(Merchant).count()}")
    session.close()

if __name__ == "__main__":
    main()