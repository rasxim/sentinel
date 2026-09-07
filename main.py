from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date, datetime, time, timedelta
from models import Account,Decision
from db import SessionLocal

app = FastAPI()

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

@app.get("/health")

def root():
    return {"status":"ok"}

@app.post("/score")

def score_transaction(txn: TransactionIn):
    DECLINE_THRESHOLD = 8
    REVIEW_THRESHOLD = 5
    session = SessionLocal()
    account = session.get(Account, txn.account_id)

    if account is None:
        raise HTTPException(status_code=404, detail=f"account {txn.account_id} not found")

    # Copy the value out while the session is alive. After commit() the ORM
    # object is expired, and after close() it is detached - reading from it
    # then raises DetachedInstanceError.
    typical_amount = account.typical_amount

    ratio = txn.amount / typical_amount

    if ratio >= DECLINE_THRESHOLD:
        decision = "DECLINE"
    elif ratio >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    record = Decision(
        txn_id = txn.txn_id,
        account_id=txn.account_id,
        amount=txn.amount,
        typical_amount=typical_amount,
        ratio=ratio,
        decision=decision,
        ts=txn.ts,
        scored_at=datetime.now()
    )
    session.add(record)
    session.commit()
    session.close()

    return {
        "decision": decision,
        "amount": txn.amount,
        "typical_amount": typical_amount,
        "ratio": round(ratio, 2),
    }



   
    
   