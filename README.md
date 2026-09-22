# Sentinel — Real-Time Card Fraud Detection

**[Live demo → sentinel-pm60.onrender.com](https://sentinel-pm60.onrender.com)** — runs on a free
tier, so the first visit after a quiet spell takes about a minute to wake up.

A backend scoring service that evaluates card transactions in real time and routes each one to
`APPROVE`, `REVIEW`, or `DECLINE`. Built end to end: a transaction simulator, a point-in-time
feature pipeline, a model selection process, cost-based threshold selection, and a FastAPI
service that scores a live transaction in about 14 ms.

The focus is on the parts of a fraud system that are easy to get quietly wrong — target
leakage, threshold choice, and training/serving consistency — rather than on squeezing out
accuracy.

| | |
|---|---|
| **Data** | 189,230 transactions · 2,000 accounts · 300 merchants · 90 days |
| **Fraud rate** | 0.66% (1,255 transactions across 3 typologies) |
| **Features** | 16 point-in-time behavioural features |
| **Model** | XGBoost — **PR-AUC 0.895** vs 0.755 logistic-regression baseline, 0.008 no-skill |
| **Leakage control** | label shuffle collapses PR-AUC to **0.012**, the no-skill level |
| **Policy** | cost-optimised thresholds cut modelled loss **52%** vs a default 0.5 cut-off, plus a guardrail rule for amounts beyond the training range |
| **Serving** | ~14 ms median, 17 ms p95 per decision, with an offline/online feature parity test |
| **Validation** | adversarial probes found and fixed a data-generation leak — see below |
| **Console** | browser UI with live scenarios and per-decision SHAP explanations |

---

## Architecture

```
  generate_data.py                                  ┌────────────────┐
  (synthetic world) ──────────────────────────────► │  sentinel.db   │
                                                    │    (SQLite)    │
                          ┌───── account history ───┤                │
                          │                         └────────▲───────┘
                          ▼                                  │
  POST /score      ┌──────────────────────┐                  │
  {transaction} ──►│   FastAPI service    │                  │
                   │  1. load history     │                  │
                   │  2. compute 16 feats │──► model.joblib  │
                   │  3. score            │    (XGBoost)     │
                   │  4. apply policy     │                  │
                   └──────────┬───────────┘─── decision ─────┘
                              ▼
              APPROVE  <0.05    REVIEW  0.05–0.85    DECLINE  ≥0.85
```

A request arrives carrying one transaction and no history. The service loads that account's
prior transactions under a strict as-of cutoff, computes 16 behavioural features live, scores
them, applies a cost-optimised policy, and persists the decision.

---

## 1. The data

The dataset is generated rather than downloaded. That was a deliberate trade: public card-fraud
datasets are PCA-anonymised, with no account identity, no merchant, and no usable timestamp —
which makes point-in-time feature engineering impossible and leaves nothing to verify a leakage
control against. Generating the world gives every transaction an account, a merchant, a
location, a time, **and a known ground-truth label**, which is what makes the rest of this
project testable.

`generate_data.py` builds 2,000 accounts and 300 merchants across 8 US cities, producing
189,230 transactions over 90 days. Each account has a per-account baseline spend, five
preferred merchants it uses 80% of the time, and lognormally distributed amounts. A quarter of
accounts take one multi-day trip to another city — so distance from home is deliberately *not*
a reliable fraud signal, and the model has to learn implied speed instead.

![Class balance and fraud typologies](docs/chart_data.png)

Three fraud typologies are planted, totalling 1,255 transactions:

| pattern | accounts | txns | how it is constructed | intended signal |
|---|---|---|---|---|
| **card testing** | 60 | 731 | 6–20 charges of $0.50–$12 inside 5–40 minutes, at unrelated local merchants, mostly card-not-present | burst rate, merchant diversity |
| **impossible travel** | 80 | 239 | 2–4 card-present charges in a different city, 20–90 minutes after a genuine transaction | distance ÷ elapsed time |
| **account takeover** | 50 | 285 | 3–8 charges at 1.5–8× the account's norm, at never-used merchants in its home city | amount deviation + merchant novelty |

Every pattern is built to **overlap with legitimate behaviour**. Fraud amounts sit inside normal
spending ranges, the impossible-travel city pairs run from blatant (coast to coast) to plausible
(a regional hop), and the 20% non-favourite rule means real customers also shop somewhere new.
Without that overlap a single `WHERE` clause would separate fraud perfectly and the model would
be decorative.

Each pattern leaves a distinct fingerprint in the features:

![Feature signatures by fraud type](docs/chart_signatures.png)

---

## 2. Feature engineering

Sixteen features, each computed **only from an account's prior transactions**:

| group | features | targets |
|---|---|---|
| transaction | `amount`, `card_present`, `hour` | — |
| amount deviation | `amount_ratio`, `amount_vs_hist_mean`, `amount_vs_hist_max` | account takeover |
| velocity | `secs_since_last`, `txn_count_1h`, `txn_count_24h`, `distinct_merchants_1h` | card testing |
| location | `dist_from_last_km`, `implied_speed_kmh`, `dist_from_home_km` | impossible travel |
| merchant familiarity | `merchant_seen_before`, `merchant_use_count`, `distinct_merchants_30d` | account takeover |

Features are ratios and counts relative to each account rather than raw values, so the model
never learns anything about a *specific* customer. It learns that a charge several times an
account's own baseline at a merchant that account has never used is suspicious — a rule that
holds equally for a $12/day customer and a $200/day one. A consequence worth noting: a brand-new
account can be scored immediately, with no retraining.

### Leakage control

The loop computes features from accumulated history, then appends the current transaction — so
a transaction structurally cannot contribute to its own features. The train/test split is by
time, never random: the first 80% of the timeline trains, the last 20% tests.

This is verified, not asserted. Inside a card-testing burst:

| | mean `txn_count_1h` |
|---|---|
| first transaction of a burst | 0.07 |
| later transactions in the same burst | 6.91 |
| ordinary legitimate transaction | 0.05 |

The first charge of a 20-charge burst sees exactly as much history as a normal transaction —
none of its own burst. A leaking implementation would show 7–20 there.

A consequence worth stating plainly: **the model cannot flag the first charge of a burst on
velocity features**, because at that moment nothing distinguishes it. Detection begins at the
second transaction. That is how real fraud detection behaves too.

---

## 3. Model selection

A logistic regression baseline was built first, deliberately, so the gradient-boosted model had
something to beat. Both are evaluated on the final 20% of the timeline — 37,846 transactions,
288 of them fraud (0.76%).

| model | PR-AUC | vs. no-skill |
|---|---|---|
| no-skill (test fraud rate) | 0.0076 | 1× |
| logistic regression | 0.7554 | 99× |
| **XGBoost** | **0.8951** | **118×** |
| XGBoost, labels shuffled | 0.0122 | ≈ no-skill |

![Precision-recall curves](docs/chart_pr_curve.png)

**Accuracy is not reported.** Fraud is 1 in 131 transactions here, so a model that always
answers "not fraud" scores 99.2%. **ROC-AUC is also omitted** — its denominator is dominated by
the 37,558 legitimate rows, so flagging dozens of innocent customers barely moves it. PR-AUC is
the metric that penalises false positives proportionally at this level of imbalance.

**XGBoost beat the baseline by 1.18×.** That is a modest gain, and worth stating honestly: the
signal here is largely linear-separable, so a simple model captures most of it. The gain comes
from interactions a weighted sum cannot express — a large amount is unremarkable at a familiar
merchant and suspicious at a new one, and only a tree can represent that conditionally.

**The red line is the important one.** Shuffling the training labels destroys the relationship
between features and outcome; retraining then collapses PR-AUC from 0.8951 to 0.0122 — effectively
the no-skill line (0.0076). If leakage existed, the model would still find signal there. It finds none.

![Feature importance](docs/chart_importance.png)

`distinct_merchants_1h` dominates because card testing is 58% of all fraud and that feature is
its fingerprint.

---

## 4. Choosing the threshold

The model outputs a probability; turning it into a decision needs a cut-off, and 0.5 is an
arbitrary one — it is the midpoint of a number between 0 and 1, and knows nothing about the
business. The two error types do not cost the same:

- missed fraud ≈ **$200** — the transaction is written off
- false decline ≈ **$50** — support contact, customer friction, churn risk

Sweeping the threshold from 0.01 to 0.99 and computing `(missed × 200) + (false_declines × 50)`:

![Cost curve](docs/cost_curve.png)

| threshold | missed fraud | false declines | cost |
|---|---|---|---|
| 0.50 (default) | 77 | 16 | $16,200 |
| **0.11 (cost-optimal)** | 44 | 60 | **$11,800** |

The optimum sits at 0.11 because missed fraud costs 4× a false decline — it is worth wrongly
declining several customers to prevent one loss. Defaulting to 0.5 costs $4,400 on this test set
for no benefit. Note also that the curve is **asymmetric**: being too aggressive is cheap, being
too lax is expensive.

Extending to two thresholds gives the three-way policy:

![Where transactions go under the policy](docs/chart_policy.png)

```
APPROVE   p < 0.05        37,478 transactions    33 frauds missed
REVIEW    0.05 ≤ p < 0.85    179 transactions    71 frauds caught, 108 false alarms
DECLINE   p ≥ 0.85           189 transactions     5 legitimate customers blocked
```

Total modelled cost **$7,745**, against $16,200 at a fixed 0.5 cut-off — a **52.2% reduction**.
The review queue is 0.47% of volume, a realistic analyst workload, and hard declines are
reserved for near-certainty. A two-way system would have to either block those 179 ambiguous
transactions or let them all through; the middle bucket exists precisely because the model is
legitimately uncertain about them.

Detection rate by pattern:

| pattern | recall at `p ≥ 0.5` | sent to at least REVIEW |
|---|---|---|
| card testing | 94.3% | 96.2% |
| impossible travel | 72.0% | 96.0% |
| account takeover | 56.8% | 79.5% |
| false positive rate | 0.043% | — |

That ordering tracks how much signal each pattern was designed to leave, and account takeover is
hardest because it was deliberately built to overlap with normal spending.

---

## 5. Serving

`POST /score` takes a single transaction and returns a decision. The interesting problem is that
a live request carries **no history** — `txn_count_1h`, `dist_from_last_km` and
`merchant_use_count` are not in the request body and must be reconstructed at decision time.

The service keeps an **in-memory history per account**. The first request for an account loads
its full history from SQLite; after that, every scored transaction is appended in memory, so a
burst builds on itself — the third charge of a card-testing burst sees the first two. Reads use
the same as-of cutoff as training: only transactions strictly earlier than the one being scored.
Scored transactions are never written back to the `transactions` table, which stays the
untouched training set, and `POST /reset` discards live additions.

A made-up four-charge burst against one account, sent over HTTP:

```
charge 1   $3.10   txn_count_1h=0   p=0.514   REVIEW
charge 2   $1.25   txn_count_1h=1   p=0.993   DECLINE
charge 3   $7.80   txn_count_1h=2   p=0.980   DECLINE
charge 4   $2.40   txn_count_1h=3   p=0.982   DECLINE
```

Training computes features in one pandas pass over 189,230 rows. Serving computes them for one
transaction against that in-memory history. **Two implementations of the same sixteen
definitions**, and if they drift apart nothing fails loudly — the model is simply served inputs
that no longer match what it was trained on, and quietly gets worse.

`test_parity.py` guards both halves. It compares every feature value against the offline output
for a stratified sample covering all three fraud typologies and an account's first-ever
transaction, and it checks that the in-memory history returns exactly the rows the SQL query
would:

```
compared 3,216 values across 201 transactions
compared live vs SQL history for 201 transactions, 0 mismatch(es)
PASS - offline and online agree on every feature
```

Transactions are indexed on `(account_id, ts)` — equality column first, range column second —
matching the query used to load an account's history.

---

## Validation: adversarial testing

Aggregate metrics can look healthy while the model has learned the wrong thing, so the service
was also probed by hand with constructed transactions against a single account (typical spend
$23.78, Chicago), changing one variable at a time.

That testing found a real defect. A $2,000 charge — 84× the account's norm — at a merchant it had
never used was **approved**, and the score did not move between $500 and $2,000.

The cause was in the data generator, not the model. Account-takeover fraud had been drawing its
"unfamiliar" merchants from all eight cities, so 63% of it happened far from home. The model had
learned *"large amount + new merchant + far away"* rather than the intended *"large amount + new
merchant"*, and a local takeover slipped through. Card testing had the same flaw in milder form:
its occasional card-present charges jumped across the country between swipes.

Both patterns now draw merchants from the account's home city, and the pipeline was rebuilt from
scratch. The same probes, before and after:

| probe (same account, at home, merchant never used) | before | after |
|---|---|---|
| $150 (6.3× typical) | 0.032 — approve | **0.219 — review** |
| $2,000 (84× typical) | 0.026 — approve | **0.145 — review** |
| $150 at a favourite merchant *(control)* | — | 0.0001 — approve |
| Seattle 30 minutes after a Chicago purchase *(control)* | — | 0.955 — decline |
| Seattle 16 hours later *(plausible flight)* | — | 0.0003 — approve |

Headline PR-AUC fell from 0.961 to 0.895, and account-takeover recall from 72% to 57%. Both
earlier figures were inflated by the leak; the current ones reflect what the features actually
support.

### Amounts outside the training range

A second round of probing found a $20,000 charge — 841× the account's usual spend — **approved**
at one of its regular stores, and at an unfamiliar store the score froze at 0.145 for every amount
from $500 upward.

This is a property of tree models rather than a bug in the pipeline: they cannot extrapolate. The
largest `amount_ratio` in the training data is 12.75×, and every value above that falls into the
same leaves, so $500 and $20,000 are indistinguishable to the model.

The first attempt was a **monotonic constraint** telling XGBoost that a larger amount may never
lower the risk. It made things worse — PR-AUC fell from 0.895 to 0.865 and account-takeover recall
from 57% to 44% — because fraud risk is U-shaped in amount: tiny charges are card testing and
large ones are takeover, and a monotone function cannot represent a U. It was reverted.

What shipped instead is a **guardrail rule** in the service, layered on top of the model:

| `amount_ratio` | decision |
|---|---|
| up to 12.75× (inside the training range) | the model decides |
| above 12.75× | at least REVIEW — the model has no evidence here |
| above 38.3× (3× the training range) | DECLINE |

The rule can only make a decision stricter, never looser, and the console labels any decision the
rule changed and shows what the model alone would have said. It touches **1 of 37,846** test-set
transactions, so the evaluation figures above are unaffected.

The per-decision explanations also surfaced a subtler artifact. On that $20,000 charge, the model
counted "182× its largest purchase so far" as *lowering* the risk. In the generated data, takeover
fraud was capped at about 2.8× an account's largest prior purchase, so the only transactions that
ever far exceeded it were rare legitimate splurges — and the model learned that. It is covered for
extreme amounts by the rule, and listed under limitations below.

---

## Design trade-offs

| decision | why | what it costs |
|---|---|---|
| Synthetic data over a public dataset | keeps account, merchant, time and location, so point-in-time features and a leakage control are possible | accuracy figures describe the pipeline, not real-world fraud |
| SQLite over Postgres | real SQL, zero setup, and it sits behind SQLAlchemy so the swap is a config line | not suitable for concurrent production write load |
| In-memory history per account, loaded from SQLite on first use | no database round trip after the first request, and bursts build live | held in one process: lost on restart and not shared across replicas; years of history would need running aggregates instead of full lists |
| Two feature implementations | serving cannot use a batch pandas pass | requires a parity test to stay honest |
| Logistic regression kept as a baseline | makes the gradient-boosted gain measurable rather than assumed | — |
| A rule for amounts outside the training range | tree models cannot extrapolate, so the model has no evidence past 12.75× usual spend | the review and decline cut-offs for the rule are set by hand from the training range, not learned |

## Limitations

**Because the data is synthetic, the model can only recover patterns I generated — the accuracy
figures bound what the pipeline can do, not what it would do on real card data. What the project
demonstrates is the pipeline itself: point-in-time feature construction, leakage control,
baseline comparison, and cost-based threshold selection.**

Specifically:

- Real fraud is adversarial and shifts as detection improves. These patterns are static.
- Real labels arrive late and incompletely, via chargebacks. These are perfect and immediate.
- The generator's timing and volume constants are hand-tuned estimates, not fitted to real card
  data.
- The cost figures ($200 / $50, plus $5 per human review) are assumptions. They are the right
  *kind* of input for this decision, but the specific values are illustrative.
- Takeover fraud in the generated data only ever happens at merchants the account has never used,
  and never far above the account's largest purchase. So inside the training range, a large charge
  at a *regular* store is trusted — up to 12.75× usual spend it can be approved — and exceeding an
  account's previous maximum reads as a legitimate splurge. Beyond 12.75× the guardrail takes over.
  Planting takeover fraud at familiar merchants and with a wider amount range would teach the model
  both cases directly.

---

## Running it

```bash
pip install -r requirements.txt

python init_db.py          # create tables
python generate_data.py    # 189k synthetic transactions   (~1 min)
python features.py         # build the feature table       (~3 min)
python train.py            # baseline, model, shuffle control
python cost_curve.py       # threshold sweep + cost_curve.png
python test_parity.py      # offline/online feature parity
python make_charts.py      # regenerate README figures

uvicorn main:app
```

Then open **`http://127.0.0.1:8000`** for the scoring console, or `/docs` for the raw API.

The console picks one clean demo account per city and lets you run four scenarios — an everyday
purchase, a card-testing burst, impossible travel, and account takeover — or compose your own
transaction. Each decision expands to show its three strongest reasons, taken from the model's
own SHAP contributions (XGBoost `pred_contribs`), alongside all sixteen feature values. A
simulated clock starts the morning after the account's last real purchase and advances with each
transaction, which is what lets a burst or an impossible trip play out in real time.

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"txn_id":"T1","account_id":"ACC00042","merchant_id":"MER0175",
       "amount":25.00,"ts":"2026-08-31T12:00:00",
       "lat":41.9053,"lon":-87.6595,"card_present":true}'
```

```json
{"decision": "APPROVE", "fraud_probability": 0.000002, "history_used": 109, "latency_ms": 13.7}
```

`features.csv` and `sentinel.db` are build artifacts and are not committed — the steps above
regenerate them.

---

## Stack

Python · FastAPI · SQLAlchemy · SQLite · XGBoost · scikit-learn · pandas · matplotlib
