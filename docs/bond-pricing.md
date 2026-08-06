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

## 5. Next: bond asset swaps

A **par-par asset swap** is the engine's existing machinery with a bond leg:
- The investor pays par (1.0), receives the bond (worth its market dirty price `P`), pays the bond's fixed
  coupons, and receives a floating leg + the asset-swap spread `A`.
- `A` solves so the swap's PV offsets the bond's off-par value:
  `A · annuity_float = PV_bond,curve − P_dirty`, i.e. the ASW spread is the engine's `ParSpread` quote
  where one leg is the bond's fixed cashflows (curve-priced) and the target is the market dirty price.

This reuses `float_leg_pv` / `annuity` (`cashflows.hpp`) and the curve-space bond PV above — no new pricing
primitive. Planned as `build/asset_swap.hpp` (construction) + a `ParSpread`-style residual/quote so a book
of asset swaps calibrates/reprices on the same W-cache, plus a JSON/`BundleSession` verb.

## 6. Extending to other bond types (no layer change)

Only a **builder** is added; the kernel and sweep are untouched:
- **Gilt / Bund / corporate** — different frequency, calendar, day count, ex-dividend rules → new builder
  fields feeding the same `Bond`/`YieldBond`.
- **FRN** — the floating coupon amount is *projected* off a forecast curve before it reaches the kernel;
  the curve-space `Bond` then prices exactly as a fixed-cashflow bond (the projection is a builder step),
  and its discount-margin is the same per-bond Newton as z-spread.
- **Money-market / ACT-365 bonds** — a different `YieldConvention` (compounding `f`, day count) in the
  builder; the street formula is unchanged.
