# Day 1 — 1 Sep 2026

## Built
- SQLAlchemy + SQLite data layer: `accounts`, `merchants`, `transactions`
- Synthetic generator: 2,000 accounts, 300 merchants, 188,281 transactions over 90 days
- Verified the habit is real: account ACC00042's top 5 merchants = 84% of its activity,
  then a hard cliff to 1–2 visits. That cliff is what makes fraud detectable.

## Design decisions I should be able to defend

**Compound index `(account_id, ts)` — order is not cosmetic.**
Every query in this project has one shape: "what did THIS account do in the LAST N hours?"
Index on account first (equality), time second (range). General rule: equality columns
before range columns. Reversed, the index still returns correct results — just slowly.
A silent performance bug, which is the hardest kind to notice.
Verify with `EXPLAIN QUERY PLAN` — want `SEARCH ... USING INDEX`, not `SCAN`.
Didn't index `amount` because nothing ever searches by it. Indexes cost write speed.

**Accounts:merchants = 2000:300, ~37 per city, 5 favourites each.**
The ratio is load-bearing. Too few merchants → everyone shops everywhere → "new merchant"
never fires. Too many → every purchase is at a new merchant → it always fires. Same
feature killed from both directions. Needed enough repetition to form a habit and enough
unfamiliar places for a thief to go.

**80% favourites.**
Fraud is only detectable as a break in a habit. No habit, no signal. The other 20% keeps
it non-robotic, so "unfamiliar merchant" isn't a perfect giveaway that trivialises the
problem. This is the most important line in the generator.

**Amounts: lognormal, centred per account.**
Spending is right-skewed — mostly small, occasionally large, never negative. A normal
distribution would produce negative dollar amounts. Centring on each account's own
typical value is what makes "amount ÷ account average" a real feature: $200 is
unremarkable for one customer and alarming for another.

**Location depends on card_present.**
In person → merchant coordinates + GPS jitter. Online → home coordinates, because there
is no physical location. Consequence: 30% of transactions can never trip the travel-speed
feature. That's realistic, and it's a limitation to state rather than hide.

**`random.seed(42)`.**
Same data every run. When a score changes tomorrow I need to know it was my code, not the
dice. Note: the seed makes the whole *sequence* reproducible — it does NOT rewind between
calls. Calling `make_accounts()` twice returns two different sets of people. Generate
once, reuse the result.

## What broke, and what it taught me

**A success message is not evidence of success.**
`print("Tables created")` fired while zero tables existed, because `models.py` was empty
and `create_all` found an empty registry. Cost me an hour. Fix: print the actual result,
not the intention — `print("created:", list(Base.metadata.tables))` cannot lie.
This is the same failure my fraud model will attempt on day 6, wearing a different hat:
a number that looks like success without being checked against reality.

**Read tracebacks bottom-up.**
Last line = what actually broke. Then scan up for the first file that's mine, ignoring
everything in `site-packages`. Two pieces, five seconds. On a constraint error, the
`[SQL: ...]` line shows exactly what was sent — `INSERT INTO merchants (merchant_id,
name, category, lat, lon)` with no `city` told me instantly that the model had been
updated and the generator hadn't.

**Don't keep a project in OneDrive.**
Sync locks files mid-write; SQLite writes continuously to the db plus journal files.
Result is `database is locked` or corruption, with no clue why. Also: moving a venv
breaks it — absolute paths are baked in. Delete and recreate, never relocate.

**Imports run the file.**
`import models` in `init_db.py` looks unused and Pylance greys it out. It isn't — the
import is what *executes* the class definitions, which is what registers them on `Base`.
Delete it and you get an empty database with no error. Every module runs once and is
cached, so all files share the same `Base` object. That shared-registry idea is why a
second `declarative_base()` anywhere would silently break everything.

## Interview answers I now have

**"Your data is synthetic — doesn't the model just learn the rules you wrote?"**
Partly, and I state that limitation in the README. A model trained on data I generated
can at best recover my generating process; it can't discover a pattern I never planted.
But the generating rules and the detecting rules are different objects. The generator
says "a thief in another city buys at unfamiliar merchants." The model never sees that —
it sees ~15 aggregate features computed at decision time and has to invert a process it
can't observe. The thresholds and interactions that connect the two are nowhere
specified, and I don't know them either.
What the project demonstrates is the pipeline, not the accuracy number: point-in-time
feature construction, leakage control, baseline comparison, cost-based thresholds.

**"Why ML instead of hand-written rules?"**
I built the rules first (day 3) specifically so I could measure the gap. Rules lose for
four reasons: (1) one threshold can't fit all accounts — 5× average means different
things at $12/day and $200/day; (2) the signal lives in interactions, and 15 features
have thousands of combinations I can't hand-enumerate; (3) rules give yes/no, but the
cost-based threshold work needs a continuous probability — the cost curve is impossible
without one; (4) rules give a single operating point, a model gives the whole trade-off
curve.

## Open / for later
- README must state the synthetic-data limitation explicitly
- Generator constants (hour weights, daily counts) are hand-tuned guesses, not fitted to
  real card data — say so
- Try `N_MERCHANTS = 10` some time and watch the "new merchant" feature die, to confirm
  I understand what the ratio controls