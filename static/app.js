"use strict";

// ---------------------------------------------------------------- state
const state = {
  meta: { review_threshold: 0.05, decline_threshold: 0.85 },
  accounts: [],
  profile: null,
  clock: 0,           // simulated time, ms since epoch (UTC)
  seq: 0,
  counts: { APPROVE: 0, REVIEW: 0, DECLINE: 0 },
  latencies: [],
  busy: false,
  advanceMs: 600000,
};
// Identifies this browser tab to the server, which keeps a separate live
// history per visitor so simultaneous demo users never affect each other.
const session = (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2))
  .replace(/-/g, "").slice(0, 12).toUpperCase();

const $ = (id) => document.getElementById(id);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// ---------------------------------------------------------------- formatting
const money = (n) => "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const int = (n) => Math.round(n).toLocaleString("en-US");
const fixed = (n, d = 1) => n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

function clockLabel(ms) {
  return new Date(ms).toLocaleString("en-US", {
    timeZone: "UTC", weekday: "short", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }) + " UTC";
}
function hhmm(ms) {
  return new Date(ms).toLocaleTimeString("en-US", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hour12: false });
}
function shortDate(iso) {
  return new Date(iso + "Z").toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
}
function duration(s) {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  if (s < 172800) return `${fixed(s / 3600, 1)} h`;
  return `${Math.round(s / 86400)} days`;
}
function prob(p) {
  if (p < 0.001) return "<0.001";
  if (p > 0.999) return ">0.999";
  return p.toFixed(3);
}
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// Position on a log-odds scale, as a fraction 0..1. Log-odds spreads out both
// tails - 0.000001 and 0.999999 sit equally far from the middle - and it is the
// scale the model's SHAP contributions are measured in.
const LOGIT_RANGE = 10;
const logPos = (p) => {
  const q = Math.min(1 - 1e-9, Math.max(1e-9, p));
  const z = Math.log(q / (1 - q));
  return Math.min(1, Math.max(0, (z + LOGIT_RANGE) / (2 * LOGIT_RANGE)));
};

// Plain-English reading of one feature value.
function describe(name, v) {
  if (v === null || v === undefined) {
    return {
      secs_since_last: "First transaction on record",
      dist_from_last_km: "No previous location to compare",
      implied_speed_kmh: "No previous location to compare",
      amount_vs_hist_mean: "No purchase history to compare",
      amount_vs_hist_max: "No purchase history to compare",
    }[name] || `${name} unknown`;
  }
  switch (name) {
    case "amount": return `${money(v)} charge`;
    case "card_present": return v ? "Card present, in person" : "Card not present, online";
    case "hour": return `Made at ${String(v).padStart(2, "0")}:00 UTC`;
    case "amount_ratio": return `${fixed(v, v < 1 ? 2 : 1)}× this account's usual spend`;
    case "amount_vs_hist_mean": return `${fixed(v, v < 1 ? 2 : 1)}× its average purchase`;
    case "amount_vs_hist_max": return `${fixed(v, 2)}× its largest purchase so far`;
    case "secs_since_last": return `${duration(v)} after the previous transaction`;
    case "txn_count_1h": return `${plural(v, "transaction")} in the past hour`;
    case "txn_count_24h": return `${plural(v, "transaction")} in the past 24 hours`;
    case "distinct_merchants_1h": return `${plural(v, "different merchant")} in the past hour`;
    case "dist_from_last_km": return `${int(v)} km from the previous transaction`;
    case "implied_speed_kmh": return `Implied travel speed ${int(v)} km/h`;
    case "dist_from_home_km": return `${int(v)} km from home`;
    case "merchant_seen_before": return v ? "Merchant used before" : "First time at this merchant";
    case "merchant_use_count": return v ? `Used this merchant ${plural(v, "time")} before` : "Never used this merchant";
    case "distinct_merchants_30d": return `${plural(v, "merchant")} in the past 30 days`;
    default: return `${name} = ${v}`;
  }
}

// ---------------------------------------------------------------- api
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", "X-Visitor": session },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

function toast(msg) {
  const t = h("div", { class: "toast", role: "status" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 3200);
}

// ---------------------------------------------------------------- scoring
function merchantById(id) {
  const p = state.profile;
  return [...p.favourites, ...p.local_new, ...p.elsewhere].find((m) => m.merchant_id === id);
}

async function score({ merchant, amount, inPerson, advanceMs }) {
  const p = state.profile;
  state.clock += advanceMs;
  const txn = {
    txn_id: `WEB-${session}-${++state.seq}`,
    account_id: p.account_id,
    merchant_id: merchant.merchant_id,
    amount: Math.round(amount * 100) / 100,
    ts: new Date(state.clock).toISOString(),
    // Same convention as the generator: in person uses the store's location,
    // online uses the cardholder's home.
    lat: inPerson ? merchant.lat : p.home_lat,
    lon: inPerson ? merchant.lon : p.home_lon,
    card_present: inPerson,
  };
  const out = await api("/score", { method: "POST", body: txn });
  state.counts[out.decision] += 1;
  state.latencies.push(out.latency_ms);
  addRow(txn, merchant, out);
  renderTally();
  $("clock").textContent = clockLabel(state.clock);
  return out;
}

function addRow(txn, merchant, out) {
  $("empty").hidden = true;
  const ratio = txn.amount / state.profile.typical_amount;
  const d = out.decision.toLowerCase();
  const pos = logPos(out.fraud_probability) * 100;
  const where = txn.card_present ? merchant.city : "Online";

  const zones = h("div", { class: "bar" }, ...zoneSpans(), h("span", { class: "marker", style: `left:${pos}%` }));

  const reasons = h("ul", { class: "reasons" }, out.reasons.map((r) => {
    const up = r.contribution > 0;
    return h("li", {},
      h("span", { class: `dir ${up ? "up" : "down"}`, "aria-label": up ? "raised risk" : "lowered risk" }, up ? "↑" : "↓"),
      h("span", {}, describe(r.feature, out.features[r.feature])),
      h("span", { class: "w", title: "SHAP contribution in log-odds" }, (up ? "+" : "") + r.contribution.toFixed(2)));
  }));

  const title = (s) => s[0] + s.slice(1).toLowerCase();
  const ruleNote = out.rule && h("div", { class: "rule-note" },
    h("strong", {}, `Rule: ${title(out.rule.decision)}. `),
    `This charge is ${fixed(out.rule.amount_ratio, 1)}× usual spend, beyond the ${fixed(out.rule.training_max, 1)}× ` +
    "range the model was trained on. Tree models can't extrapolate past their training data, so a rule " +
    `decides instead. The model alone scored ${prob(out.fraud_probability)} (${out.model_decision.toLowerCase()}).`);

  const top = new Set(out.reasons.map((r) => r.feature));
  const feats = h("dl", { class: "featgrid" }, Object.entries(out.features).map(([k, v]) => [
    h("dt", { class: top.has(k) ? "hl" : null }, k),
    h("dd", { class: top.has(k) ? "hl" : null }, v === null ? "null" : typeof v === "number" ? (Number.isInteger(v) ? v : fixed(v, 2)) : v),
  ]));

  const row = h("li", { class: "row enter" },
    h("button", { class: "row-main", type: "button", "aria-expanded": "false",
      onclick: (e) => {
        const li = e.currentTarget.parentElement;
        const open = li.classList.toggle("open");
        e.currentTarget.setAttribute("aria-expanded", String(open));
      } },
      h("span", { class: "t-time" }, hhmm(state.clock)),
      h("span", { class: "t-merch" },
        h("span", { class: "name" }, merchant.name),
        h("span", { class: "sub" }, `${merchant.category.replace("_", " ")} · ${where}`)),
      h("span", { class: "t-amt" },
        h("span", { class: "amt" }, money(txn.amount)),
        h("span", { class: "ratio" }, `${fixed(ratio, ratio < 1 ? 2 : 1)}× usual`)),
      h("span", { class: "t-risk" },
        h("span", { class: "bar-wrap" }, zones),
        h("span", { class: "p" }, `p ${prob(out.fraud_probability)}`)),
      h("span", { class: "t-dec" },
        h("span", { class: `badge b-${d}` }, title(out.decision)),
        out.rule && h("span", { class: "rule-tag", title: "Decided by the out-of-range amount rule" }, "Rule")),
      h("span", { class: "chev", "aria-hidden": "true" }, "▸")),
    h("div", { class: "detail" },
      h("div", {},
        h("h3", {}, "Why this decision"),
        ruleNote,
        reasons,
        h("p", { class: "detail-note" },
          `Scored against ${plural(out.history_used, "earlier transaction")} in ${out.latency_ms} ms. ` +
          "Weights are the model's own SHAP contributions.")),
      h("div", {},
        h("h3", {}, "All 16 features"),
        feats)));

  $("feed").prepend(row);
}

function zoneSpans() {
  const r = logPos(state.meta.review_threshold) * 100;
  const dcl = logPos(state.meta.decline_threshold) * 100;
  return [
    h("span", { class: "zone z-approve", style: `width:${r}%` }),
    h("span", { class: "zone z-review", style: `width:${dcl - r}%` }),
    h("span", { class: "zone z-decline", style: `width:${100 - dcl}%` }),
  ];
}

function renderTally() {
  $("n-approve").textContent = state.counts.APPROVE;
  $("n-review").textContent = state.counts.REVIEW;
  $("n-decline").textContent = state.counts.DECLINE;
  const l = [...state.latencies].sort((a, b) => a - b);
  $("latency").textContent = l.length ? `${fixed(l[Math.floor(l.length / 2)], 1)} ms` : "—";
}

// ---------------------------------------------------------------- scenarios
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MIN = 60000, HOUR = 3600000;

const SCENARIOS = [
  {
    title: "Everyday purchase",
    desc: "A usual amount at one of this account's regular stores.",
    steps: (p) => [
      { merchant: p.favourites[0], amount: p.typical_amount * 1.05, inPerson: true, advanceMs: 90 * MIN },
    ],
  },
  {
    title: "Card-testing burst",
    desc: "Five tiny online charges at unfamiliar stores, a minute apart.",
    steps: (p) => [3.10, 1.25, 7.80, 2.40, 0.99].map((amount, i) => ({
      merchant: p.local_new[i % p.local_new.length], amount, inPerson: false,
      advanceMs: i === 0 ? 60 * MIN : 70000,
    })),
  },
  {
    title: "Impossible travel",
    desc: "A purchase at home, then one 25 minutes later across the country.",
    steps: (p) => [
      { merchant: p.favourites[1] || p.favourites[0], amount: p.typical_amount, inPerson: true, advanceMs: 60 * MIN },
      { merchant: p.elsewhere[p.elsewhere.length - 1], amount: p.typical_amount * 1.2, inPerson: true, advanceMs: 25 * MIN },
    ],
  },
  {
    title: "Account takeover",
    desc: "Large purchases at local stores this account has never used.",
    steps: (p) => [4.5, 6.2, 7.4].map((mult, i) => ({
      merchant: p.local_new[(i + 1) % p.local_new.length], amount: p.typical_amount * mult,
      inPerson: i === 0, advanceMs: i === 0 ? 90 * MIN : 35 * MIN,
    })),
  },
];

function renderScenarios() {
  const list = $("scenarios");
  list.replaceChildren(...SCENARIOS.map((s, i) => h("li", {},
    h("button", { class: "scenario", type: "button", "data-i": i, onclick: () => runScenario(i) },
      h("span", { class: "idx" }, String(i + 1).padStart(2, "0")),
      h("span", {}, h("span", { class: "title" }, s.title), h("span", { class: "desc" }, s.desc)),
      h("span", { class: "go" }, "Run →")))));
}

function setBusy(busy, activeIndex = -1) {
  state.busy = busy;
  document.querySelectorAll(".scenario").forEach((b, i) => {
    b.disabled = busy;
    b.classList.toggle("running", busy && i === activeIndex);
    b.querySelector(".go").textContent = busy && i === activeIndex ? "Running…" : "Run →";
  });
  $("f-submit").disabled = busy;
}

async function runScenario(i) {
  if (state.busy || !state.profile) return;
  setBusy(true, i);
  try {
    const steps = SCENARIOS[i].steps(state.profile);
    for (let k = 0; k < steps.length; k++) {
      await score(steps[k]);
      if (k < steps.length - 1) await sleep(650);
    }
  } catch (err) {
    toast(`Couldn't score that transaction: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------- compose form
function renderMerchantOptions() {
  const p = state.profile;
  const group = (label, list) => h("optgroup", { label },
    list.map((m) => h("option", { value: m.merchant_id },
      `${m.name} — ${m.category.replace("_", " ")}${m.city !== p.home_city ? ", " + m.city : ""}`)));
  $("f-merchant").replaceChildren(
    group("Regular stores", p.favourites),
    group(`Never used, in ${p.home_city}`, p.local_new),
    group("Other cities", p.elsewhere));
  updateHints();
}

function updateHints() {
  const p = state.profile;
  if (!p) return;
  const m = merchantById($("f-merchant").value);
  if (m) {
    const seen = p.favourites.find((f) => f.merchant_id === m.merchant_id);
    const where = m.city === p.home_city ? m.city : `${m.city}, ${int(m.km_from_home)} km from home`;
    $("f-merchant-hint").textContent = seen ? `${where} · visited ${plural(seen.visits, "time")}` : `${where} · never visited`;
  }
  const amt = parseFloat($("f-amount").value);
  $("f-amount-hint").textContent = Number.isFinite(amt) && amt > 0
    ? `${fixed(amt / p.typical_amount, 1)}× usual spend` : "";
}

async function onCompose(e) {
  e.preventDefault();
  if (state.busy || !state.profile) return;
  const raw = $("f-amount").value.trim().replace(/[$,]/g, "");
  const amount = Number(raw);
  if (!raw || !Number.isFinite(amount) || amount <= 0 || amount > 100000) {
    $("f-error").textContent = "Enter an amount between $0.01 and $100,000.";
    $("f-amount").closest(".money").classList.add("invalid");
    $("f-amount").focus();
    return;
  }
  setBusy(true);
  try {
    await score({
      merchant: merchantById($("f-merchant").value),
      amount,
      inPerson: $("f-present").value === "1",
      advanceMs: state.advanceMs,
    });
  } catch (err) {
    toast(`Couldn't score that transaction: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------- account
async function loadAccount(id) {
  setBusy(true);
  try {
    const [p] = await Promise.all([
      api(`/accounts/${encodeURIComponent(id)}`),
      api(`/reset?account_id=${encodeURIComponent(id)}`, { method: "POST" }),
    ]);
    state.profile = p;
    // Start the demo clock at 09:00 the day after the account's last real
    // purchase. After its whole stored history, and in daytime: legitimate
    // spending is rare at night, so a scenario landing at 2am would be flagged
    // partly for the hour alone, which hides what the scenario is showing.
    const last = new Date(p.last_ts + "Z");
    state.clock = Date.UTC(last.getUTCFullYear(), last.getUTCMonth(), last.getUTCDate() + 1, 9, 0);
    state.counts = { APPROVE: 0, REVIEW: 0, DECLINE: 0 };
    state.latencies = [];

    $("acct-id").textContent = p.account_id;
    $("acct-city").textContent = `${p.home_city}`;
    $("stat-spend").textContent = money(p.typical_amount);
    $("stat-history").textContent = `${p.n_transactions} txns`;
    $("stat-merchants").textContent = String(p.distinct_merchants);
    $("recent").replaceChildren(...p.recent.slice(0, 4).map((r) => h("li", {},
      h("span", { class: "when" }, shortDate(r.ts)),
      h("span", { class: "who" }, r.merchant, h("span", { class: "muted" }, ` · ${r.card_present ? r.city : "online"}`)),
      h("span", { class: "mono" }, money(r.amount)))));

    $("feed").replaceChildren();
    $("empty").hidden = false;
    $("clock").textContent = clockLabel(state.clock);
    $("f-amount").value = (Math.round(p.typical_amount * 100) / 100).toFixed(2);
    renderMerchantOptions();
    renderTally();
  } catch (err) {
    toast(`Couldn't load account: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------- boot
async function boot() {
  renderScenarios();

  $("account-select").addEventListener("change", (e) => loadAccount(e.target.value));
  $("reset").addEventListener("click", () => state.profile && loadAccount(state.profile.account_id));
  $("compose").addEventListener("submit", onCompose);
  $("f-merchant").addEventListener("change", updateHints);
  $("f-amount").addEventListener("input", () => {
    $("f-error").textContent = "";
    $("f-amount").closest(".money").classList.remove("invalid");
    updateHints();
  });
  $("f-advance").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-ms]");
    if (!b) return;
    state.advanceMs = Number(b.dataset.ms);
    $("f-advance").querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  });

  try {
    const [meta, accounts] = await Promise.all([api("/meta"), api("/accounts")]);
    state.meta = meta;
    state.accounts = accounts;
    $("model-chip").textContent = `${meta.model} · ${meta.n_features} features`;
    $("lg-review").textContent = meta.review_threshold;
    $("lg-decline").textContent = meta.decline_threshold;
    document.querySelector(".legend .scale").replaceChildren(...zoneSpans());
    $("account-select").replaceChildren(...accounts.map((a) => h("option", { value: a.account_id },
      `${a.home_city} · ${a.account_id}`)));
    // Shareable links: ?scenario=burst runs a scenario on arrival, &expand opens
    // its latest decision, &account=ACC00005 picks the account.
    const params = new URLSearchParams(location.search);
    const wanted = accounts.find((a) => a.account_id === params.get("account"));
    const start = wanted || accounts.find((a) => a.account_id === "ACC00042") || accounts[0];
    $("account-select").value = start.account_id;
    await loadAccount(start.account_id);

    const SCENARIO_KEYS = { everyday: 0, burst: 1, travel: 2, takeover: 3 };
    const key = params.get("scenario");
    if (key in SCENARIO_KEYS) {
      await runScenario(SCENARIO_KEYS[key]);
      if (params.has("expand")) document.querySelector("#feed .row .row-main")?.click();
    }
  } catch (err) {
    toast(`Couldn't reach the scoring service: ${err.message}`);
  }
}

boot();
