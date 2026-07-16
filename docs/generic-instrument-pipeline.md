# Generic instrument pipeline — design

**Goal.** One pipeline for any currency, OIS or IBOR, spread or no spread, any averaging /
compounding / IBOR future. **No index, currency, calendar, or curve-build strategy may appear in
engine code.** Those live ONLY in tests/fixtures. Engine code sees numbers: sub-period times, accrual
factors, spreads, convexity, curve indices.

This document is the single source of truth for the refactor. It is complete: implement from it
without further input.

---

## STATUS (reconciled 2026-07-16, branch `feat/generic-instrument-pipeline`, gates green: 74/74 + perf PASS)

**The pipeline is BUILT and TESTED but NOT ADOPTED.** Every section below is implemented EXCEPT the
retirement in §2 — the generic model was **appended** alongside the legacy one, never substituted.
That is exactly why "existing numbers unchanged" holds, and it is also the outstanding debt.

| § | Item | Status |
| --- | --- | --- |
| 1 | ONE rate formula for every shape | **LANDED** |
| 2 | `RateObservation`/`FloatCoupon`/`FixedCoupon` + pricing | **LANDED** |
| 2 | Backward-compat invariant (legacy reduction) | **LANDED** — asserted bit-exact (`EXPECT_EQ(d, 0.0)`) |
| 2 | **"Retire `OisSwap`/`CompoundedFuture`/`AveragedFuture`"** | **NOT DONE** — still live structs with their own pricing fns in `cashflows.hpp`; still what the reference market and all 4 benchmarks build |
| 3 | `FloatLeg`/`FixedLeg`, roles on legs | **LANDED** |
| 3 | `Instrument` + `ParRate`/`ParSpread`/`Rate` | **LANDED** — but constructed ONLY by `tests/{extract,generic_instrument}_test.cpp` |
| 3 | Documented deterministic residual order | **LANDED** — `avg_futs, comp_futs, swaps, bases, instruments`; generic block appended LAST so no existing row renumbers |
| 4 | ONE float batch (`BundleFloatBatch`) | **LANDED** — legs + comp + avg futures; both fused fast paths (`sub_is_identity`, `cpn_is_plain`) preserved |
| 4 | Analytic Jacobian chain vs AAD ~1e-9 | **LANDED** — achieved **6.4e-16** |
| 5 | Generic extractors, dispatch on coupon type | **LANDED** — legacy per-shape extractors retained alongside |
| 6 | All 10 regression items | **LANDED** |
| 7 | Non-goals | respected — `W_all`, interpolation, knot strategy, risk-ladder API untouched |
| 8 | Both gates green every commit | **HELD** |

**Remaining work (the adoption step), in order:**
1. Migrate `tests/reference_curve.hpp::build_problem` onto `extract_float_leg`/`extract_fixed_leg` +
   `Instrument`. **Assert, do not assume:** the generic OIS path uses `valueDates().front()/back()`
   where the legacy path uses `accrualStartDate()/accrualEndDate()`. They coincide on the reference
   market (both hit `fairRate` at ~6e-17) — pin it with a test before the call sites move.
2. Migrate the four benchmarks; re-run the perf gate (baselines are legacy-path numbers today).
3. Delete the legacy structs, their pricing functions, their extractors and their batches; drop the
   now-dead residual-order blocks 1–4 and the `cashflows.hpp:107–126` index-naming comments with them.

**Deviations from this document, deliberate and already merged** (see CLAUDE.md §7c for the why):
- §5's IBOR `[fixingPeriodStart, fixingPeriodEnd]` does not exist in QL 1.34 → `fixingValueDate()` /
  `fixingEndDate()` (par-coupon approximation; **not** `fixingMaturityDate()`).
- §5's averaged-OIS `tau_index` "on the index day count" → `accrualPeriod()` (the COUPON's day
  count), to match QL's pricer. Identical in the standard case.
- Gearing and partially-fixed compounded coupons fold into the EXISTING weight/`realized` fields; no
  new fields were added (`P·X − 1 ≡ P·(X−1) + (P−1)`).

---

## 1. The unifying formula

Every floating rate in scope is:

```
rate = ( Σ_k w_k · [ DF_fc(s_k) / DF_fc(e_k) − 1 ] + realized ) / τ_index
```

| Instrument | sub-periods | realized | τ_index |
| --- | --- | --- | --- |
| Compounded OIS coupon (any index) | ONE `[accStart, accEnd]` (daily compounding telescopes to the DF ratio) | 0 | index accrual over the period |
| Averaged OIS/FF coupon (any index) | one per business day | Σ past `fixing_d · accrual_d` | period year fraction |
| IBOR fixing (any tenor/ccy) | ONE `[fixingPeriodStart, fixingPeriodEnd]` | 0 | index accrual of that period |
| Already-fixed coupon | NONE (empty) | `fixing · τ_index` | index accrual |
| Compounding future | ONE | realized part | index accrual |
| Averaging future | one per business day | realized part | period year fraction |
| IBOR future | ONE | 0 | index accrual |

`w_k` defaults to 1 (standard OIS/IBOR/averaging). It exists so weighted schemes are representable
without another type. Keep the weight vector; if empty, treat as all-ones.

**There is exactly ONE rate type in the engine.** "SOFR 3M compounding future", "FF averaging
future", "Euribor future" are not engine concepts — they are just different `RateObservation` data
built by a test.

## 2. Cashflow model (`pricing/cashflows.hpp`)

```
RateObservation {
  vector<double> sub_start;    // forecast-curve times
  vector<double> sub_end;
  vector<double> weight;       // empty => all 1
  double realized  = 0.0;      // constant (zero derivative)
  double tau_index = 0.0;      // denominator
}

FloatCoupon {
  RateObservation obs;
  double pay;      // discount-curve time
  double tau_pay;  // payment accrual (its OWN day count)
  double spread;   // additive, may be 0
}

FixedCoupon { double pay; double tau; }   // rate supplied by the instrument/quote
```

Pricing (templated on Scalar, AAD-safe — seed accumulators from the first curve-dependent term,
add constants as raw `double`):

```
rate(obs, fc)        = ( Σ w_k (DF_fc(s_k)/DF_fc(e_k) − 1) + realized ) / tau_index
float_coupon_pv(c, fc, dc) = DF_dc(c.pay) · ( rate(c.obs, fc) + c.spread ) · c.tau_pay
float_leg_pv(leg, fc, dc)  = Σ float_coupon_pv
annuity(leg, dc)           = Σ DF_dc(pay) · tau
```

**Backward-compatibility invariant (MUST hold):** with ONE sub-period, `realized=0`, `spread=0`, and
`tau_pay == tau_index`, `float_coupon_pv` reduces algebraically to today's
`DF_dc(pay)·(DF_fc(accStart)/DF_fc(accEnd) − 1)`. The existing 43 tests must pass **unchanged and
bit-comparable** (to ~1e-15). If a test moves numerically, the generalization is wrong.

Retire `OisSwap`, `CompoundedFuture`, `AveragedFuture` as distinct *pricing* concepts; they become
data shapes a test builds. Keep a swap as two legs (see §3).

> **NOT DONE — see STATUS above.** These three are still live pricing structs in `cashflows.hpp`
> with their own pricing functions, and are still what `tests/reference_curve.hpp` and all four
> benchmarks build. The generic model was added alongside them, not in place of them.

## 3. Instruments, legs, roles, quotes

**Curve roles belong to LEGS, not instruments** — that is what makes tenor-basis and cross-currency
representable without new types.

```
FloatLeg { vector<FloatCoupon> coupons; int forecast; int discount; }
FixedLeg { vector<FixedCoupon> coupons; int discount; }
```

An instrument = legs + a quote transform + a market quote. Quote transforms (all RATE units,
CLAUDE.md §2):

| Quote | model value |
| --- | --- |
| `ParRate` | `float_leg_pv / annuity` |
| `ParSpread` (basis) | `(pv_bench − pv_fwd) / annuity` |
| `Rate` (future) | `rate(obs, fc) + convexity` |

`convexity` is an **input number**, never a model in engine code. Hull–White / Ho–Lee live in tests.

`residual = model_quote − market_quote`, in rate units. Residual ORDER must remain deterministic and
documented (it is depended on by the Jacobian, the W-cache batches, `market()`, and warm/streaming).

## 4. Compiled engine (`pricing/compiled_book.hpp`)

Today `BundleFloatLegs` assumes one sub-period per coupon and `BundleAvgFutures` carries the
sub-period reduction. **These become ONE primitive**: a float batch with TWO sparse reductions:

1. `R_sub` : sub-periods → per-coupon rate numerator
2. `R_cpn` : coupons → per-leg PV

```
num   = R_sub · ( w ⊙ (DF[s]/DF[e] − 1) )          // per coupon
rate  = (num + realized) / tau_index                // per coupon
pv    = R_cpn · ( DF[pay] ⊙ (rate + spread) ⊙ tau_pay )
```

A future is the same batch stopping at `rate + convexity`. **PERF RULE (measured, do not regress):**
materialize each per-coupon/per-sub-period vector into a `VectorXd` BEFORE any sparse reduction
`R * v`. Handing Eigen's sparse×dense an unevaluated gather/divide expression re-does the work per
access (~1.28× slower). See `BundleFloatLegs::pv`.

Analytic Jacobian keeps its factorization `J = −(dr/dDF · diag(DF)) · W_all`. New chain for a float
coupon (derive and verify against AAD to ~1e-9):

```
d rate / d DF[s_k] = w_k / (tau_index · DF[e_k])
d rate / d DF[e_k] = − w_k · DF[s_k] / (tau_index · DF[e_k]^2)
d pv   / d DF[pay] = (rate + spread) · tau_pay
d pv   / d (rate)  = DF[pay] · tau_pay          // chain into d rate/dDF above
```

`W_all` construction (per-curve blocks + spread-base ancestry) is UNCHANGED by this refactor.

## 5. Extractors (`ql/extract.hpp`) — generic QuantLib → plain data

The ONLY QuantLib-touching layer. It must dispatch on **QuantLib coupon type**, never on index
identity. No index names, no currency, no calendar constants in this file.

- `OvernightIndexedCoupon` → read `averagingMethod()`:
  - *Compounded* → one sub-period `[accrualStart, accrualEnd]`, `tau_index` = index accrual.
  - *Simple/averaged* → per-business-day sub-periods; past days fold into `realized` from the index
    history; `tau_index` = period year fraction on the index day count.
- `IborCoupon` → one sub-period `[fixingPeriodStart, fixingPeriodEnd]`, `tau_index` =
  index accrual of that period, `spread` = coupon spread, `tau_pay` = `accrualPeriod()`.
- `FixedRateCoupon` → `FixedCoupon{ pay, accrualPeriod() }`.
- Already-fixed coupons → empty sub-periods + `realized`.

All *times* use the CURVE day counter (`dc.yearFraction(ref, date)`); all *accruals* (`tau_pay`,
`tau_index`) come from the instrument/index's own day counter. This separation is what makes
30/360-vs-ACT/360 correct — keep it.

Futures: the caller supplies the sub-periods (or a QL schedule) and the convexity NUMBER.

## 6. What moves to tests

Everything index/market specific: SOFR / FF / EURIBOR index objects, FOMC & ECB meeting dates, the
reference market, Hull–White convexity computation, knot-placement strategy, which instruments pin
which knots. Engine code must not name them.

Required regression coverage (each vs a QuantLib oracle where one exists, else vs a known analytic
value; tolerances per `tests/tolerances.hpp`):

1. OIS compounded swap (existing reference market) — **numbers unchanged**.
2. Averaged (1M) and compounded (3M) futures — **numbers unchanged**.
3. Vanilla IBOR swap (e.g. EURIBOR 3M) vs QuantLib `VanillaSwap::fairRate`, single curve.
4. IBOR swap multi-curve (forecast ≠ discount) vs QuantLib.
5. Float coupon **with a spread** vs QuantLib.
6. **30/360 fixed vs ACT/360 float** — mixed day counts.
7. **Mixed frequency**: semi-annual basis + annual outright in ONE calibration.
8. IBOR future (single-period) — generic future path.
9. Already-fixed / partially-fixed coupon (realized).
10. Analytic Jacobian vs AAD for every new coupon shape (~1e-9).

## 7. Non-goals for this pass

Cross-currency FX-linked instruments; a new convexity model; changing the interpolation or knot
strategy; changing `W_all`; touching the risk-ladder API.

## 8. Invariants (non-negotiable)

- **Both gates green at every commit** (CLAUDE.md §3). Never commit red.
- Existing 43 tests keep passing with unchanged numbers.
- No perf regression on `curve_build`, `risk_full_jacobian`, `portfolio_analytics`,
  `warm_recalibration` (hard gate = speedup vs QuantLib).
- Hot paths stay templated on `Scalar`, header-only, no QuantLib, no virtual dispatch, no heap in
  inner loops, no hard-coded SIMD width.
- Residual order stays deterministic and documented.
