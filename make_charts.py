"""Generates the figures used in README.md."""

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

SEED = 42
TRAIN_FRACTION = 0.8
NOT_FEATURES = ["txn_id", "account_id", "ts", "is_fraud", "fraud_type"]

INK = "#1f2937"
BLUE = "#2563eb"
GREY = "#9ca3af"
RED = "#dc2626"
GREEN = "#16a34a"
AMBER = "#d97706"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.alpha": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

f = pd.read_csv("features.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
f["group"] = f["fraud_type"].fillna("legitimate")

cutoff = int(len(f) * TRAIN_FRACTION)
train, test = f.iloc[:cutoff], f.iloc[cutoff:]

X_train, y_train = train.drop(columns=NOT_FEATURES + ["group"]), train["is_fraud"]
X_test, y_test = test.drop(columns=NOT_FEATURES + ["group"]), test["is_fraud"]
FEATURES = list(X_train.columns)


# ---------------------------------------------------------------- 1. the data
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

counts = [int((~f.is_fraud.astype(bool)).sum()), int(f.is_fraud.sum())]
axes[0].bar(["legitimate", "fraud"], counts, color=[GREY, RED], width=0.55)
axes[0].set_yscale("log")
axes[0].set_ylabel("transactions (log scale)")
axes[0].set_title("Class balance — fraud is 0.69% of volume", loc="left", fontweight="bold")
for i, c in enumerate(counts):
    axes[0].text(i, c * 1.15, f"{c:,}", ha="center", fontweight="bold")

pats = f[f.is_fraud.astype(bool)].group.value_counts()
axes[1].barh(pats.index[::-1], pats.values[::-1], color=[AMBER, BLUE, RED][:len(pats)][::-1], height=0.55)
axes[1].set_xlabel("transactions")
axes[1].set_title("Planted fraud typologies", loc="left", fontweight="bold")
for i, v in enumerate(pats.values[::-1]):
    axes[1].text(v + 12, i, str(v), va="center", fontweight="bold")

fig.tight_layout()
fig.savefig("docs/chart_data.png", dpi=150)
print("saved docs/chart_data.png")


# ------------------------------------------------- 2. each pattern's signature
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
order = ["legitimate", "card_testing", "impossible_travel", "account_takeover"]
short = ["legit", "card\ntesting", "impossible\ntravel", "account\ntakeover"]
cols = [GREY, RED, BLUE, AMBER]

specs = [
    ("txn_count_1h", "transactions in prior hour", "Velocity → card testing", False),
    ("implied_speed_kmh", "km/h since last transaction", "Implied speed → impossible travel", True),
    ("amount_ratio", "amount ÷ account's typical", "Amount deviation → account takeover", False),
]

for ax, (col, ylab, title, logscale) in zip(axes, specs):
    vals = [f[f.group == g][col].dropna() for g in order]
    medians = [v.median() for v in vals]
    p90 = [v.quantile(0.90) for v in vals]
    x = np.arange(len(order))
    ax.bar(x - 0.19, medians, width=0.36, color=cols, label="median")
    ax.bar(x + 0.19, p90, width=0.36, color=cols, alpha=0.45, label="90th pct")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    if logscale:
        ax.set_yscale("symlog")
    ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("docs/chart_signatures.png", dpi=150)
print("saved docs/chart_signatures.png")


# ------------------------------------- 3. model comparison + leakage control
logreg = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
logreg.fit(X_train, y_train)
lr_p = logreg.predict_proba(X_test)[:, 1]

xgb = joblib.load("model.joblib")["model"]
xgb_p = xgb.predict_proba(X_test)[:, 1]

rng = np.random.default_rng(SEED)
shuf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                     subsample=0.9, colsample_bytree=0.9,
                     eval_metric="aucpr", random_state=SEED)
shuf.fit(X_train, rng.permutation(y_train.values))
shuf_p = shuf.predict_proba(X_test)[:, 1]

base = y_test.mean()
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for probs, name, colour, style in [
    (xgb_p, "XGBoost", BLUE, "-"),
    (lr_p, "Logistic regression (baseline)", GREEN, "-"),
    (shuf_p, "XGBoost, labels shuffled (control)", RED, "-"),
]:
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(rec, prec, style, color=colour, linewidth=2, label=f"{name} — PR-AUC {ap:.4f}")

ax.axhline(base, color=GREY, linestyle="--", linewidth=1.4,
           label=f"no-skill — PR-AUC {base:.4f}")
ax.set_xlabel("recall — share of fraud caught")
ax.set_ylabel("precision — share of alerts that are really fraud")
ax.set_title("Precision–recall, with the label-shuffle control", loc="left", fontweight="bold")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="center left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("docs/chart_pr_curve.png", dpi=150)
print("saved docs/chart_pr_curve.png")


# ---------------------------------------------------------- 4. what it learned
imp = sorted(zip(FEATURES, xgb.feature_importances_), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.barh([k for k, _ in imp], [v for _, v in imp], color=BLUE, height=0.65)
ax.set_xlabel("gain-based importance")
ax.set_title("Feature importance", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig("docs/chart_importance.png", dpi=150)
print("saved docs/chart_importance.png")


# ------------------------------------------------ 5. what the policy does
REVIEW_T, DECLINE_T = 0.05, 0.90
approve = xgb_p < REVIEW_T
review = (xgb_p >= REVIEW_T) & (xgb_p < DECLINE_T)
decline = xgb_p >= DECLINE_T
yv = y_test.to_numpy()

fraud_split = [int((approve & (yv == 1)).sum()), int((review & (yv == 1)).sum()), int((decline & (yv == 1)).sum())]
legit_split = [int((approve & (yv == 0)).sum()), int((review & (yv == 0)).sum()), int((decline & (yv == 0)).sum())]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = ["APPROVE", "REVIEW", "DECLINE"]
cols3 = [GREEN, AMBER, RED]

axes[0].bar(labels, fraud_split, color=cols3, width=0.55)
axes[0].set_title(f"Where the {int(yv.sum())} frauds went", loc="left", fontweight="bold")
axes[0].set_ylabel("transactions")
for i, v in enumerate(fraud_split):
    axes[0].text(i, v + 4, str(v), ha="center", fontweight="bold")

axes[1].bar(labels, legit_split, color=cols3, width=0.55)
axes[1].set_yscale("log")
axes[1].set_title(f"Where the {int((yv==0).sum()):,} legitimate transactions went", loc="left", fontweight="bold")
axes[1].set_ylabel("transactions (log scale)")
for i, v in enumerate(legit_split):
    axes[1].text(i, max(v, 1) * 1.25, f"{v:,}", ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig("docs/chart_policy.png", dpi=150)
print("saved docs/chart_policy.png")

print("\nsummary")
print(f"  logreg  PR-AUC {average_precision_score(y_test, lr_p):.4f}")
print(f"  xgboost PR-AUC {average_precision_score(y_test, xgb_p):.4f}")
print(f"  shuffled       {average_precision_score(y_test, shuf_p):.4f}")
print(f"  fraud  approve/review/decline: {fraud_split}")
print(f"  legit  approve/review/decline: {legit_split}")
