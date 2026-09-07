import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

SEED = 42
TRAIN_FRACTION = 0.8

# Identifiers and the label - none of these may be fed to the model.
NOT_FEATURES = ["txn_id", "account_id", "ts", "is_fraud", "fraud_type"]


# ---------------------------------------------------------------- load + split
f = pd.read_csv("features.csv", parse_dates=["ts"])
f = f.sort_values("ts").reset_index(drop=True)

cutoff = int(len(f) * TRAIN_FRACTION)
train = f.iloc[:cutoff]
test = f.iloc[cutoff:]

print("=== time-based split ===")
print(f"train: {train.ts.min()} -> {train.ts.max()}  n={len(train):,}  fraud={int(train.is_fraud.sum())}")
print(f"test:  {test.ts.min()} -> {test.ts.max()}  n={len(test):,}  fraud={int(test.is_fraud.sum())}")

X_train = train.drop(columns=NOT_FEATURES)
y_train = train["is_fraud"]
X_test = test.drop(columns=NOT_FEATURES)
y_test = test["is_fraud"]

feature_names = list(X_train.columns)
print(f"\n{len(feature_names)} features: {feature_names}")

# A model that guesses randomly scores roughly the positive rate on PR-AUC.
# Every number below is only meaningful relative to this.
base_rate = y_test.mean()
print(f"\nno-skill PR-AUC (test fraud rate): {base_rate:.5f}")


# ---------------------------------------------------------------- baseline: LR
logreg = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000,
                                 class_weight="balanced",
                                 random_state=SEED)),
])
logreg.fit(X_train, y_train)
lr_probs = logreg.predict_proba(X_test)[:, 1]
lr_prauc = average_precision_score(y_test, lr_probs)

print("\n=== logistic regression (baseline) ===")
print(f"PR-AUC: {lr_prauc:.4f}   ({lr_prauc / base_rate:.1f}x no-skill)")


# ---------------------------------------------------------------- XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="aucpr",
    random_state=SEED,
)
xgb.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_prauc = average_precision_score(y_test, xgb_probs)

print("\n=== xgboost ===")
print(f"PR-AUC: {xgb_prauc:.4f}   ({xgb_prauc / base_rate:.1f}x no-skill)")
print(f"lift over logistic regression: {xgb_prauc / lr_prauc:.2f}x")

importances = sorted(zip(feature_names, xgb.feature_importances_),
                     key=lambda kv: kv[1], reverse=True)
print("\ntop features:")
for name, score in importances[:8]:
    print(f"  {score:.3f}  {name}")


# ---------------------------------------------------------------- shuffle test
# Destroy the relationship between features and labels by shuffling the
# TRAINING labels only, then retrain and score against the real test labels.
# If PR-AUC stays high, the model is reading something it shouldn't be.
rng = np.random.default_rng(SEED)
y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)

xgb_shuf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="aucpr",
    random_state=SEED,
)
xgb_shuf.fit(X_train, y_shuffled)
shuf_probs = xgb_shuf.predict_proba(X_test)[:, 1]
shuf_prauc = average_precision_score(y_test, shuf_probs)

print("\n=== shuffle control ===")
print(f"PR-AUC with shuffled labels: {shuf_prauc:.4f}")
print(f"vs no-skill baseline:        {base_rate:.5f}")
print(f"vs real model:               {xgb_prauc:.4f}")
if shuf_prauc < base_rate * 3:
    print("PASS - collapsed to noise, no leakage detected")
else:
    print("FAIL - still predictive on shuffled labels, something leaks")


# ---------------------------------------------------------------- save
joblib.dump({"model": xgb, "features": feature_names}, "model.joblib")
print("\nsaved model.joblib")
