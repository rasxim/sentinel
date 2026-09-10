import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# What each kind of mistake costs us. These are assumptions, not measurements -
# the whole point is that they are a business input, not a statistical one.
COST_MISSED_FRAUD = 200     # we eat the transaction
COST_FALSE_DECLINE = 50     # angry customer, support call, churn risk
COST_REVIEW = 5             # a few minutes of an analyst's time

TRAIN_FRACTION = 0.8


# ---------------------------------------------------------------- test scores
f = pd.read_csv("features.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
test = f.iloc[int(len(f) * TRAIN_FRACTION):].copy()

bundle = joblib.load("model.joblib")
probs = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
y = test["is_fraud"].to_numpy()

n_fraud = int(y.sum())
n_legit = int((y == 0).sum())
print(f"test set: {len(y):,} transactions, {n_fraud} fraud, {n_legit:,} legit\n")


# ------------------------------------------------------- single threshold sweep
thresholds = np.arange(0.01, 1.00, 0.01)
costs, missed_list, fd_list = [], [], []

for t in thresholds:
    flagged = probs >= t
    missed = int(((~flagged) & (y == 1)).sum())        # fraud we let through
    false_declines = int((flagged & (y == 0)).sum())   # legit we blocked
    costs.append(missed * COST_MISSED_FRAUD + false_declines * COST_FALSE_DECLINE)
    missed_list.append(missed)
    fd_list.append(false_declines)

costs = np.array(costs)
best_i = int(costs.argmin())
best_t = thresholds[best_i]

# The naive choice everyone reaches for by default.
naive_i = int(np.argmin(np.abs(thresholds - 0.50)))

print("=== single-threshold cost sweep ===")
print(f"naive 0.50 threshold : cost ${costs[naive_i]:,}  "
      f"(missed {missed_list[naive_i]}, false declines {fd_list[naive_i]})")
print(f"cost-optimal {best_t:.2f}     : cost ${costs[best_i]:,}  "
      f"(missed {missed_list[best_i]}, false declines {fd_list[best_i]})")
reduction = 100 * (costs[naive_i] - costs[best_i]) / costs[naive_i]
print(f"reduction vs 0.50    : {reduction:.1f}%")

# What perfect and do-nothing look like, for scale.
print(f"\nfor scale - approve everything: ${n_fraud * COST_MISSED_FRAUD:,}")
print(f"            decline everything: ${n_legit * COST_FALSE_DECLINE:,}")


# ---------------------------------------------------------------- the plot
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(thresholds, costs, linewidth=2, color="#2563eb", label="total expected cost")
ax.axvline(best_t, color="#16a34a", linestyle="--", linewidth=1.5,
           label=f"cost-optimal = {best_t:.2f}  (${costs[best_i]:,})")
ax.axvline(0.50, color="#dc2626", linestyle=":", linewidth=1.5,
           label=f"naive 0.50  (${costs[naive_i]:,})")
ax.set_xlabel("decision threshold (model probability)")
ax.set_ylabel(f"expected cost  (${COST_MISSED_FRAUD}/missed fraud, ${COST_FALSE_DECLINE}/false decline)")
ax.set_title("Cost-optimal threshold selection")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("docs/cost_curve.png", dpi=150)
print("\nsaved docs/cost_curve.png")


# -------------------------------------------------- two thresholds: the policy
# review_t <= p < decline_t goes to a human. Fraud caught in review costs us
# only the analyst's time; legit sent to review costs the same.
best = None
for review_t in np.arange(0.05, 0.96, 0.05):
    for decline_t in np.arange(review_t + 0.05, 1.00, 0.05):
        approve = probs < review_t
        review = (probs >= review_t) & (probs < decline_t)
        decline = probs >= decline_t

        cost = (
            int((approve & (y == 1)).sum()) * COST_MISSED_FRAUD +   # fraud approved
            int((decline & (y == 0)).sum()) * COST_FALSE_DECLINE +  # legit declined
            int(review.sum()) * COST_REVIEW                          # everything reviewed
        )
        if best is None or cost < best[0]:
            best = (cost, review_t, decline_t, approve, review, decline)

cost, review_t, decline_t, approve, review, decline = best

print("\n=== two-threshold policy ===")
print(f"APPROVE  p < {review_t:.2f}")
print(f"REVIEW   {review_t:.2f} <= p < {decline_t:.2f}")
print(f"DECLINE  p >= {decline_t:.2f}")
print(f"total cost: ${cost:,}   (vs ${costs[naive_i]:,} at a naive 0.50 cutoff)")
print(f"reduction vs 0.50: {100 * (costs[naive_i] - cost) / costs[naive_i]:.1f}%")
print()
print(f"  approved: {int(approve.sum()):>6,}  of which fraud (missed): {int((approve & (y==1)).sum())}")
print(f"  reviewed: {int(review.sum()):>6,}  of which fraud (caught):  {int((review & (y==1)).sum())}")
print(f"  declined: {int(decline.sum()):>6,}  of which legit (wrong):  {int((decline & (y==0)).sum())}")
