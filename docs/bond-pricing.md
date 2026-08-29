# Bonds & bond asset swaps (Stage 6)

Design for adding **bond pricing** and, next, **bond asset swaps** to the engine, starting with US
Treasuries but built so any bond type is *data*, not a new abstraction layer (CLAUDE.md §0/§1). This
document is the reference; the code is `pricing/bond.hpp`, `portfolio/bond_universe.hpp`,
`build/bond.hpp`, gated by `tests/bond_yield_test.cpp` (QL-free) and `tests/bond_oracle_test.cpp`
(QuantLib oracle).

## 1. The north star

- **A bond is DATA.** A US Treasury is not a class — it is a set of field values a *builder* fills in:
  semiannual, ACT/ACT (ICMA) accrual, unadjusted regular periods, street compounding `f = 2`. A Gilt,
  Bund, corporate, ACT/365 money-market bond or FRN is another builder filling the **same** structs. No
  bond-type branch ever enters the pricing kernel.
- **Reprice a whole universe as ONE sweep.** The point (as with the swap book) is to price price/yield
  calcs across thousands of bonds without a per-bond QuantLib pricing loop — by finding the linear/closed
  structure and vectorizing it, while staying penny-perfect to QuantLib as the reference oracle.

## 2. Two pricing modes (both native shapes)

### (A) Curve space — `Σ amount · DF(pay)`
A bond's future cashflows are fixed amounts at fixed pay dates. Its present value is
`PV_today = Σ_i amount_i · DF(pay_i)` — **linear in `DF = exp(-Wx)`**, exactly the form the W-cache was
built for. So a universe of bonds rides the *same* `CompiledCurveSet` W-cache as calibration and the swap
book: build `W` once from all bond cashflow times, then a reprice is one matvec + vectorized `exp` +
sparse per-bond reductions.

- **Model dirty price** = forward value of the cashflows to the settlement date:
  `dirty = PV_today / DF(settle)`. This is exactly QuantLib's `Bond::settlementValue() =
  NPV()/discount(settlementDate)`, so our curve dirty/clean equal a QuantLib `DiscountingBondEngine` on the
  same discount factors to `1e-10` (`BondOracle.CurveSpaceMatchesDiscountingEngine`).
- **z-spread** — the constant continuous spread `s` (curve time) added to the curve that reprices the bond
  to a target dirty price: a per-bond Newton against the DFs (`bond_z_spread` /
  `CompiledBondBook::z_spreads`). Per-bond by nature, so a B-loop, not the vectorized hot path.
- **Bucketed/key-rate bond risk** falls out of the same `Scalar = ad::Dual` dual-use the swap book has
  (`PV_today` is templated on the curve) — reusing the calibration/risk machinery unchanged. (Curve-space
  DV01/key-rate wiring beyond PV is a small follow-up; the linear form makes it analytic.)

### (B) Yield / street space — `Σ CF · (1+y/f)^{−E}`
The classic curve-free treasury calcs. The street convention discounts cashflow `i` by the chained factor
`Π_{j≤i}(1+y/f)^{−f·τ_j}` where `τ_j` is the day-count year fraction of period `j`. With the cumulative
exponent `E_i = f·Σ_{j≤i} τ_j` and `base = 1 + y/f`:

```
dirty(y) = Σ_i CF_i · base^{−E_i}
```

For a regular ACT/ACT (ISMA) treasury this is `E_i = w + i` (i = 0 at the current coupon), `w` the
fraction of the current coupon period remaining at settlement. This is **exactly** QuantLib's compounded
bond-yield discounting (its per-period chained discount factors multiply to `base^{−E_i}`), so:

- clean = dirty − accrued, accrued = `(coupon/f)·(settle−prev)/(next−prev)` (ACT/ACT ISMA);
- `bond_yield_from_clean` — Newton on `dirty(y)` (strictly decreasing & convex → quadratic convergence);
- `bond_risk` — modified duration `−(1/P)dP/dy`, Macaulay = modified·`(1+y/f)`, convexity `(1/P)d²P/dy²`,
  all closed form.

All penny-perfect vs `QuantLib::BondFunctions::{cleanPrice,dirtyPrice,yield,duration,convexity}`
(`BondOracle.YieldSpaceMatchesBondFunctions`).

## 3. The universe sweep — and what the "W-cache" is for price↔YTM

`portfolio/bond_universe.hpp`. **The dominant workflow is price↔yield-to-maturity, so the sweep is built
around it.** Two facts shape the design:

**(i) YTM has no shared curve, but fits the same `exp`-of-a-linear-form shape.** Every bond carries its
*own* yield `y_b`, so the calibration W-cache (`DF = exp(-Wx)` over ONE knot vector `x`) does not apply
directly. But rewrite `P(y) = Σ CF_i·(1+y/f)^{−E_i} = Σ CF_i·exp(−E_i·r)` with `r ≡ ln(1+y/f)`: that is
`exp(−E·r)` — the W-cache shape, where each bond is its own *one-knot flat curve in period-time*, `E_i` is
the structure-only "W" (schedule-dependent, never `y`-dependent, built once), and `r` is a per-bond free
scalar. So the cache = `(amounts, exponents, bond→cashflow map)`, and a reprice is `exp` + reduce.

**(ii) The real lever unique to YTM: the powers are geometric → a coupon POLYNOMIAL evaluated by Horner.**
For a regular bond `E_i = w + i` (`w` = fraction of the current coupon period left at settlement), so
`v^{E_i} = v^w·v^i` and

```
P(y) = v^w · Σ_i CF_i·v^i = v^w · Q(v),   v = 1/(1+y/f)
```

`Q(v)` is a polynomial in `v`. `BondUniverse` caches the coefficient matrix `A` (amounts at integer
powers), the offset `w` and `f` — structure only — and the hot loop is **Horner**: `Q`, `Q'`, `Q''`
accumulate across the `B` lanes with one FMA per cashflow column (synthetic differentiation), then a single
`pow(v,w)` per bond and the chain rule `v→y` give price, `dP/dy`, `d²P/dy²`. So a Newton iteration costs
**O(cashflows) fused-multiply-adds + O(bonds) pows**, not O(cashflows) transcendentals. `yields_from_clean`
is then a BATCHED Newton across the whole universe (every bond steps together; a converged bond's step is
~0 — no masking), replacing a per-bond `BondFunctions::yield` loop. It is exact (polynomial arithmetic
equals `Σ CF·v^E` to machine precision), so still penny-perfect vs QuantLib.

The powers are geometric only when every `E_i = w + integer` (no odd coupon breaking the grid). A *builder*
bond is always regular (`BondUniverse::is_regular()` is true); an externally-supplied irregular schedule
falls back to the general per-cashflow `exp` path — still correct, just without the FMA fast path. Padded
lanes are self-annihilating in both paths (Horner: a zero leading coefficient), so there is no scalar
remainder path (§5).

**Both directions are fast, and accrued is decoupled.** The reverse (`yield → clean/dirty price`,
`dirty_prices`/`clean_prices`) is the SAME Horner cache run value-only (no derivatives), returning a const
ref into reusable scratch — allocation-free, so re-marking a universe as yields move never touches the
allocator. **Accrued** is a pure schedule quantity (`coupon_per_period × ACT/ACT-ICMA day fraction`) with
no dependence on yield/price/curve, so it is computed ONCE per bond at build (`build::accrued_interest`,
the single definition) and cached; `BondUniverse::accrued()` returns it in O(1), and it is what converts
clean↔dirty in both directions (add it going clean→yield, subtract it going yield→clean). Because accrued
and the street offset `w` are both linear in the settlement date within a coupon period — and the Horner
coefficients don't move — accrued (and a full reprice) recomputes O(1) for a **rolled settlement** via
`accrued_interest(coupon, freq, prev, next, settle')` off the cached `BuiltBond` period bounds, with no
rebuild (a coupon-date crossing changes the cashflow set and does need a rebuild).

**`CompiledBondBook` (curve space).** The curve-discounting counterpart: registers every bond's cashflow +
settlement times on one self-discounting `CurveStructure`, builds the genuine `W` once, and reprices
PV/dirty/clean + per-bond z-spread off `DF = exp(-Wx)` — the bond analogue of `CompiledPortfolio`, used
when a bond is priced off a *discount curve* (z-spread, asset-swap, cross-bond relative value) rather than
its own yield.

Both are allocation-light on the hot path (reusable scratch), matching the engine's real-time discipline.

## 4. Penny-perfect, to a known-good reference

QuantLib is the oracle throughout the repo, so it is the reference here too. `tests/bond_oracle_test.cpp`
pins, to `tests/tolerances.hpp`:

| Ours | QuantLib |
|------|----------|
| `bond_clean_from_yield` / `bond_dirty_from_yield` / accrued | `BondFunctions::cleanPrice/dirtyPrice`, `Bond::accruedAmount` |
| `bond_yield_from_clean` | `BondFunctions::yield` |
| `bond_risk` (modified duration, convexity) | `BondFunctions::duration(Modified)/convexity` |
| curve `bond_dirty_price` / `bond_clean_price` | `DiscountingBondEngine` on the same DFs |
| `bond_z_spread` | `ZeroSpreadedTermStructure` |
| `BondUniverse` batched yield sweep | `BondFunctions::yield` bond-for-bond |

The QL-free `tests/bond_yield_test.cpp` proves the internal invariants (round-trip, duration/convexity vs
finite difference, batched == scalar, curve PV/z-spread consistency) with no oracle.

### Cross-validation beyond QuantLib

A single oracle can hide a *shared* convention assumption, so the bond math is also checked against
independent references (`tests/bond_reference_test.cpp`, QL-free; harness in `tools/bond_reference/`):

| Reference | Role | Independence |
|-----------|------|--------------|
| **Excel / OpenFormula `PRICE`/`YIELD`** (basis 1 = Act/Act) | reimplemented from its published formula — a *different algebra* than our Horner kernel — and checked to 1e-12 | high (different lineage, same convention) |
| **31 CFR Part 356 Appendix B** | the **official** US Treasury formula, reimplemented from the regulation. It is a **different convention** from ours (see below), so the test pins the exact relationship, not equality | authoritative — it *defines* the Treasury convention |
| **[Rateslib](https://rateslib.com)** `calc_mode="us_gb"` **and** `"ust_31bii"` | `us_gb` is the street convention we implement (asserted EQUAL, 1e-9); `ust_31bii` is App B (asserted equal after the exact convention factor). `tools/bond_reference/gen_golden.py` emits both into a golden CSV that `BondReference.ExternalGoldenIfPresent` pins (skips if absent). **Rateslib is source-available, not open-source — non-commercial use only without a licence** | high (independent lib) |
| **FinancePy** | alternative golden source (swap into `gen_golden.py`) | high |
| **Bloomberg YAS / Tradeweb** | market truth for a specific CUSIP — manual spot-checks into the golden CSV | gold standard |

See `tools/bond_reference/README.md` for how to generate the external golden.

### Street vs Treasury (31 CFR App B) discounting — the one convention difference

Both conventions agree on the cashflows, on accrued, and on the entire coupon polynomial
`Q(v) = Σ CF_k·v^k`. They differ ONLY in how the **fractional first period `w`** is discounted:

| convention | fractional first period | `YieldConvention` | builder | QuantLib oracle |
|---|---|---|---|---|
| UK gilt / French OAT / Chinese GB | `Q(v)·v^w` (compound) always | `stub=Compound`, `final_period_simple=false` | `fixed_rate_bond` (default) | `Compounded` |
| **US Treasury STREET**, Bund | compound, **simple once only the final coupon remains** | `stub=Compound`, `final_period_simple=true` | `us_treasury`, `us_treasury_wi` | `Compounded`, then `SimpleThenCompounded` in the final period |
| **US Treasury METHOD** (31 CFR App B, Bloomberg) | `Q(v)/(1 + w·y/f)` (simple) always | `stub=Simple` | `us_treasury_tsy`, `us_treasury_wi_tsy` | `SimpleThenCompounded` |

The regulation is uniform on this — Appendix B Section II writes *every* sub-case (regular first period,
short first, long first, and the three reopened cases) as `P[1 + (r/s)(i/2)] = …`, never as a compound
`(1+i/2)^(r/s)`. The exact identity between them is

```
dirty_AppB = dirty_street · (1 + y/f)^w / (1 + w·y/f)
```

On a 6y note at `y = 2%` with `w = 30/184` that is ~7e-6 of price (**~0.7 bp**) — well above every
tolerance in this repo, so the two are **not** interchangeable.

This was found by generating the Rateslib golden (which the earlier `gen_golden.py` produced under
`ust_31bii`) and watching it disagree; the older `BondReference.CfrAppendixBShortFirstCoupon` could not
have caught it, because it reimplemented `v^w` and compared that to our `v^w` — a tautology.

**All three modes are now implemented**, selected by `pricing::YieldConvention{freq, stub,
final_period_simple}` and chosen by the named builders above — callers should not set the fields by hand.
`simple_stub()` resolves the final-period rule (it fires only when one cashflow remains, at which point
`Q(v)` is a constant). Both discount forms stay on the **Horner fast path**: the batched sweep already
accumulates `Q, Q', Q''`, and only the closing factor differs — `v^w` versus `1/(1+w·y/f)` — so the simple
form is in fact *cheaper* (no `pow`). A universe that mixes conventions is handled by a 0/1 lane mask and
one blended sweep, with no scalar remainder path; a compound-only universe takes an early return through
arithmetic that is byte-identical to before, so the `bond_sweep` gate is unaffected.

**Selecting a convention: it is DATA.** `conventions/conventions.json` has a `bonds` section (codegen'd
into `conventions_data.hpp` as `BondConv`), so a bond type is an id rather than a C++ branch:

```
build::yield_convention("US-TREASURY-TSY")                       -> pricing::YieldConvention
build::bond_from_convention(id, value, settle, issue, mat, cpn)  -> seasoned
build::wi_bond_from_convention(id, value, dated, first_cpn, mat, cpn, settle)  -> when-issued
```

and the `bonds` run_json verb takes `convention` (default `US-TREASURY`), plus `dated` + `first_coupon`
for the when-issued path — which previously had no JSON seam at all. Only conventions supported end to end
(builder **and** QuantLib oracle) are catalogued; gilts/OATs/Bunds are deliberately absent until their
calendars and ex-dividend rules exist, rather than listed half-supported.

**Note on QuantLib as the derivative oracle here.** Under `SimpleThenCompounded`, QuantLib's
`BondFunctions::duration`/`convexity` are *not* the derivatives of its own `dirtyPrice`: `CashFlows::npv`
chains STEPWISE discount factors (simple stub, then compounding), while `modifiedDuration` branches on the
CUMULATIVE time and applies a pure-compound factor over it. Measured: QuantLib's analytic modified duration
differs from a central difference of its own price by **1.4e-4** relative under `SimpleThenCompounded`,
versus **2.1e-11** under `Compounded` (where `base^{−Σ} == Π base^{−τ}` makes the two agree). Our analytic
derivative matches that finite difference to ~1e-10. So `BondOracle.TreasuryMethodMatchesSimpleThenCompounded`
checks prices and yields against QuantLib's analytic values but duration/convexity against a finite
difference of QuantLib's **price** — still a QuantLib oracle, just not its inconsistent derivative.

## 4b. Measured, not asserted — the perf gate

Until 2026-08-29 nothing in this repo had ever *timed* the bond kernel against any external library; the
"without a per-bond QuantLib pricing loop" framing above was design intent that read like a result. It is
now a gated measurement: `bench/bond_sweep_bench.cpp`, 5,000 seasoned semiannual treasuries, built from the
**same** QuantLib with the same compiler and flags (CLAUDE.md §3 perf-gate integrity), wired into
`check_perf.py` + `baselines/baselines.json` as `bond_sweep` and `bond_book`. Fingerprint `86d5211c2c03`
(Xeon W-3223, Apple clang 21):

| metric | ours | QuantLib | speedup |
|---|---|---|---|
| `bond_sweep` — price→YTM over the universe | **3.51 ms** (`BondUniverse::yields_from_clean`) | 2,500 ms (`BondFunctions::yield` bond-for-bond) | **712×** |
| — vs the *harder* baseline (see below) | 3.67 ms | 459 ms (`BM_BondSweep_QuantLibTuned`) | **125×** |
| `bond_book` — curve-space dirty prices | **414 µs** (`CompiledBondBook::dirty_prices`) | 27.9 ms (per-bond `Bond::dirtyPrice()` off a `DiscountingBondEngine`, handle relinked so no cached NPV) | **67×** |

**The 712× is not all vectorization, and must never be quoted alone.** Measured by substituting a counting
solver into the same `CashFlows::yield<Solver>` template QuantLib's default path instantiates: **one** bond
costs **34 `npv` walks + 29 `modifiedDuration` walks = 63 leg walks**, at ~11 µs each. The cause is a
scaling bug in the library — `CashFlows::IrrFinder::derivative` returns `modifiedDuration = −P′/P`, but its
objective is `npv − P(y)`, whose derivative is `−P′(y) = P·modDur`. `BondFunctions` normalizes the leg to a
100-face basis, so `P ≈ 99` and `NewtonSafe` is handed a derivative ~99× too small: every Newton step
undershoots and the safeguarded solver mostly bisects. (Starting the solve *at* the answer saves only ~16%,
so it is the step scaling, not the bracket hunt.)

`BM_BondSweep_QuantLibTuned` therefore drives the **same** QuantLib pricing (`CashFlows::npv` /
`CashFlows::duration` over the same `Leg`) from a correctly-scaled Newton, ~4 iterations. That is the honest
kernel-vs-kernel number, **125×**, and the `bond_sweep` threshold is set at 50× — below both ratios, so the
gate does not rest on the artifact.

Also measured from the same probe, and worth knowing: a leg walk is ~11 µs for 45 cashflows (~245
ns/cashflow), of which **~215 ns is `ActualActual(ISMA)::yearFraction`** — swapping in `Actual365Fixed`
drops the full solve 783 µs → 230 µs. Recomputing the accrual structure every iteration is most of a walk;
we bake it into the exponents once at build. That is the thesis of §3, quantified.

**Gate integrity.** The fixture *aborts* rather than reporting a speedup unless the batched yields match a
converged `BondFunctions::yield` to 1e-12 (measured 1.0e-14), the tuned baseline to 1e-12 (7.8e-16), and the
curve-space dirty prices match `DiscountingBondEngine` to 1e-9 relative (5.9e-15). The correctness check
drives QuantLib at accuracy 1e-14; at its **default** 1e-10 the two differ by ~1e-10, which is QuantLib's
own stopping rule and not a disagreement. Both *timed* sides run at their library defaults, and ours is the
stricter (1e-13 on the dirty-price residual).

**No small-universe crossover.** The batched Newton cannot arrest a converged bond, so it was expected to
be relatively weaker on a small universe. The size sweep (opt-in: `SWAPS_BOND_SCALE=1
./build/bench/bond_sweep_bench --benchmark_filter=Scale`) says otherwise — ours is 680 / 600 / 736 / 738 ns
per bond at 100 / 1k / 5k / 10k, against a flat ~500 µs per bond for QuantLib.

## 5. When-issued (WI) — the subtly-different yield path

A when-issued treasury trades before issue, for settlement ON the issue (dated) date, and its FIRST coupon
period is frequently irregular. Two subtleties, both in the first period, make the yield calc differ from a
seasoned bond (`build::when_issued_bond` / `us_treasury_wi`). Its **coupon proration** follows **31 CFR
Part 356 Appendix B**; its **discounting** is the street convention, not App B's — see "Street vs Treasury
(31 CFR App B) discounting" above:

1. **Issue-date settlement.** A NEW issue settles on the dated date, so accrued is exactly **zero**. A
   **reopening** settles later within the first period and carries accrued from the *original* dated date —
   the same formula with `settle > dated`.
2. **Short first coupon.** When `first_coupon − dated` is less than a full period, the first coupon is
   **prorated** to the actual days: `coupon/f · (first_coupon − dated)/E` (`E` = the full quasi-coupon
   period `[first_coupon − period, first_coupon]`) — the Treasury "daily interest decimal".

Crucially this **stays on the fast Horner path**: the discount exponents are still `w0 + integer`
(`w0 = (first_coupon − settle)/E`), so only the *first coefficient* (`coupon·s`) and the accrued differ —
the geometric structure is intact and `BondUniverse::is_regular()` stays true. A **long** first coupon
(dated before the prior quasi-coupon date, so the first payment spans >1 quasi-period) needs the App B
quasi-period sum and is rejected for now (documented follow-up). Gated by `BondWhenIssued.*`
(`bond_yield_test.cpp`) and `BondReference.CfrAppendixBShortFirstCoupon`, which checks the App B coupon
polynomial (hence the proration) exactly AND the street↔Treasury factor above.

## 6. Bond asset swaps — DONE

A **par-par asset swap** is the engine's existing machinery with a bond leg:
- The investor pays par (1.0), receives the bond (worth its market dirty price `P`), pays the bond's fixed
  coupons, and receives a floating leg + the asset-swap spread `A`.
- `A` solves so the swap's PV offsets the bond's off-par value:
  `A · annuity_float = PV_bond,curve − P_dirty`, i.e. the ASW spread is the engine's `ParSpread` quote
  where one leg is the bond's fixed cashflows (curve-priced) and the target is the market dirty price.

This reuses `float_leg_pv` / `annuity` (`cashflows.hpp`) and the curve-space bond PV above — no new pricing
primitive. **Landed** in `d4e3184` (par-par ASW spread + a `QuantLib::AssetSwap::fairSpread` oracle,
`tests/bond_asset_swap_oracle.cpp`, 5e-5) and `88e9196` (the curve-space `asset_swap` run_json verb); the
stateless street-space `bonds` verb landed in `e4d0b6b`.

## 7. Extending to other bond types (no layer change)

Only a **builder** is added; the kernel and sweep are untouched:
- **Gilt / Bund / corporate** — different frequency, calendar, day count, ex-dividend rules → new builder
  fields feeding the same `Bond`/`YieldBond`.
- **FRN** — the floating coupon amount is *projected* off a forecast curve before it reaches the kernel;
  the curve-space `Bond` then prices exactly as a fixed-cashflow bond (the projection is a builder step),
  and its discount-margin is the same per-bond Newton as z-spread.
- **Money-market / ACT-365 bonds** — a different `YieldConvention` (compounding `f`, day count) in the
  builder; the street formula is unchanged.
