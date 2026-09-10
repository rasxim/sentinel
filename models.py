from sqlalchemy import (Column, String, Float, Boolean, DateTime, ForeignKey, Index, Integer)
from db import Base 

#Account table in database
class Account(Base):
    __tablename__ = "accounts"
    account_id = Column(String,primary_key=True)
    home_city = Column(String, nullable = False)
    home_lat       = Column(Float,  nullable=False)
    home_lon       = Column(Float,  nullable=False)
    typical_amount = Column(Float,  nullable=False)
    opened_at      = Column(DateTime, nullable=False)
#Merchant table in database
class Merchant(Base):
    __tablename__ = "merchants"
    merchant_id = Column(String, primary_key=True)
    name        = Column(String, nullable=False)
    city        = Column(String, nullable=False) 
    category    = Column(String, nullable=False)
    lat         = Column(Float,  nullable=False)
    lon         = Column(Float,  nullable=False)
#Transaction table in database
class Transaction(Base):
    __tablename__ = "transactions"
    txn_id       = Column(String, primary_key=True)
    account_id   = Column(String, ForeignKey("accounts.account_id"), nullable=False)
    merchant_id  = Column(String, ForeignKey("merchants.merchant_id"), nullable=False)
    amount       = Column(Float,    nullable=False)
    ts           = Column(DateTime, nullable=False)
    lat          = Column(Float,    nullable=False)
    lon          = Column(Float,    nullable=False)
    card_present = Column(Boolean,  nullable=False)
    is_fraud     = Column(Boolean,  nullable=False, default=False)
    fraud_type   = Column(String,   nullable=True)

class Decision(Base):
    __tablename__ = "decisions"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    txn_id         = Column(String, nullable=False)
    account_id     = Column(String, ForeignKey("accounts.account_id"), nullable=False)
    amount         = Column(Float,  nullable=False)
    typical_amount = Column(Float,  nullable=False)
    ratio          = Column(Float,  nullable=False)
    fraud_probability = Column(Float, nullable=False)
    decision       = Column(String, nullable=False)
    ts             = Column(DateTime, nullable=False)   # when the transaction happened
    scored_at      = Column(DateTime, nullable=False)   # when we decided
#Index statement
Index("ix_transac_account_ts",Transaction.account_id,Transaction.ts)