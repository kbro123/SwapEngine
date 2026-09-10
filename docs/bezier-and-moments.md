# Bézier/B-spline curve type + moment integration for averaging/lookback/obs-shift — design

Two features on branch `feat/bezier-and-moment-integration`. Both stay on the `is_linear_map`
fast path (W-cache + analytic Jacobian), both are ADDITIVE (never disturb an existing number), both
gate-green at every commit. This doc is authoritative; implement from it.

---

## Part A — B-spline region policy (control-point parameterization)

A region alongside `Flat` / `Linear` / `NaturalCubic` / `Hermite` in `curve/regions.hpp`,
satisfying the same contract (`n_values`, `t_end`, `build`, `forward`, `integral`, `out`,
`is_linear_map = true`). Scalar-templated so AAD flows through.

> **Note (order-agnostic update, engine `88de1ab`):** regions are no longer front/back — BSpline can
> LEAD, follow, or sit in the middle like any scheme (ARCHITECTURE.md `curve/` row). The "back-end" /
> "C⁰ join to the front" framing below describes BSpline as a FOLLOWING region and is unchanged in that
> case; when BSpline LEADS (region 0, `Boundary::has_predecessor == false`) `P_0` is instead tied to the
> first FREE control point `x[off]` and the region starts with a flat pre-segment at that calibrated
> value — symmetric with the far-end clamp, not a phantom `f(0)=0`. The math below is otherwise as-is.

**Parameterization (decided): the free variables ARE the control points.** A cubic B-spline
`f(u) = Σ_i P_i N_{i,3}(u)` over the back-knot times, with:
- **Clamped C⁰ join to the front:** the FIRST control point is PINNED to `boundary.value`
  (`P_0 = in.value`), exactly mirroring how `Hermite`/`NaturalCubic` pin their leading value. The
  remaining `n` control points are the free vars `x[off .. off+n)`. So `n_values() == n` back knots,
  same DOF as today — a drop-in swap for `Hermite` (`flat_bspline` vs `flat_hermite`).
- **C² internally** (cubic B-spline is automatically C² across its interior knots) — smoother than
  Hermite's C¹.
- **Convex-hull property:** `f` stays within the hull of its control points, so forwards do not
  overshoot and positivity is enforceable by keeping control points ≥ 0 (a later, optional bound).
- **Linear in the control points** ⇒ `is_linear_map = true` ⇒ W-cache + analytic Jacobian unchanged.

**Basis:** Cox–de Boor for `N_{i,3}` over a CLAMPED knot vector built from the back-knot times
(triple-repeat the end knots so the spline interpolates its first/last control point). Evaluate
`forward` by de Boor; `integral(t)` in closed form via the standard B-spline antiderivative
(the integral of a degree-k B-spline is a degree-(k+1) B-spline: `∫_0^t N_{i,k} = (ξ_{i+k+1}-ξ_i)/(k+1)
· Σ_{j≥i} N_{j,k+1}(t)`), plus `boundary.integral` as the origin offset.

**Risk transform (control-point deltas → forward-at-knot deltas).** Because `f` is linear in the
control points, `forward_at_knot = B · P` for a fixed, invertible collocation matrix `B` (evaluate the
basis at the knot times). So `d(quote)/d(forward_at_knot) = d(quote)/dP · B^{-1}`. Provide `B` (and its
factorization) as a free function `bspline_collocation(knots)` so the risk ladder can be presented in
either basis. `B` depends only on the knot structure — precompute once.

**Validation (tests/bspline_test.cpp):**
- C⁰/C¹/C² continuity at interior knots (finite-difference the forward and its derivatives).
- `integral(t)` matches a fine numerical quadrature of `forward` to ~1e-12.
- Convex-hull: `min(P) ≤ f(u) ≤ max(P)` on each span.
- Recovers a known cubic exactly (a cubic is in the span).
- Wrapped in `CurveTermStructure`, QuantLib prices instruments off it — same oracle as the other
  regions (no new oracle needed).

## Part B — moment integration (region interface + rate primitives)

### B.1 Region moment accessor
Extend every `is_linear_map` region with ONE new method:
```
Scalar integral2(double a, double b) const;   // ∫_a^b f(u)^2 du  over this region's polynomial
```
`f` is a polynomial in `u` whose coefficients are linear in `x`, so `∫f²` is a quadratic form
`x^T M(a,b) x`. Closed-form for every region (Flat: `f²·(b−a)`; Linear/cubic: integrate the squared
polynomial; B-spline: Bernstein/B-spline Gram integrals — the cleanest of the lot). The curve exposes
`integral2(a,b)` by dispatching to the region(s) the window spans. `integral` (first moment) already
exists. Higher moments (`∫f³`) added only if the truncation error below demands it.

### B.2 Moment rate — the numerics
For an observation window `[a,b]` with a fixed daily fixing schedule, the EXACT arithmetic numerator is
`Σ_d (exp(δ_d) − 1)`, `δ_d = ∫_{t_d}^{t_{d+1}} f`. Moment expansion (measured errors, docs conversation):
```
num ≈ ∫_a^b f  +  ½ · (Σ_d τ_d²/(b−a)-weighted) · ∫_a^b f²  +  O(∫f³·τ²)
```
- **Leading term `∫f`** telescopes to `log(DF(a)/DF(b))` — one term, linear in x, the W-cache.
- **Correction `½·⟨τ²⟩·∫f²`** couples the curve's `integral2` with a PRECOMPUTED per-window day-count
  moment `Σ_d τ_d²` (encodes weekends/holidays from the real fixing calendar — this is what makes it
  match QuantLib's daily arithmetic, not a pure continuous approximation).
- QUADRATURE (2026-09-10, E3 register R4/G2): ∫f² and ∫f³ are integrated on KNOT-ALIGNED Gauss panels --
  the window is split at the curve's pieces (region knots, de Boor breakpoints, turn edges, the base chain)
  and each piece gets 2 panels of 4-pt (f²) / 5-pt (f³) Gauss-Legendre, exact for cubic pieces
  (`pricing::moment_gauss_nodes`, the ONE rule both the templated path and the compiled quadratic forms
  use). The earlier fixed 32×2 / 8×2 rule was not aligned: on the shipped Flat-front Fed-funds shape with
  policy steps inside a 1Y window it lost 3.3e-3 of ∫f² = 1.7e-7 of the rate (tests/moment_quadrature_test).
- Truncation & the REAL-CALENDAR floor (measured, honest): on UNIFORM daily fixings 2 moments give
  ~4e-11 on a smooth curve and ~4e-10 on a 1Y window with 25 bp steps inside it (the truncation term grows
  with the forward's variation). On a REAL calendar (weekend 3-day accruals) the moment averaging floors at
  ~3.4e-9 (= 3e-5 bp; re-measured after the quadrature fix -- the old "~5e-9" included a quadrature share)
  vs QuantLib's exact averaged future -- the residual is the f-variation x weekend-day-structure
  correlation in the 2nd-moment coefficient (exact only for constant f); higher moments do NOT remove
  it. So the moment path is a FAST APPROXIMATION (~3e-9, far below market relevance), NOT a 1e-10
  replacement. The exact sub-period path (fixing_step==0) stays available whenever 1e-10 is required.
  Gate: tests/bspline_oracle_test.cpp isolates moment-vs-exact-daily (~3.4e-9) from exact-daily-vs-
  QuantLib (~1e-16, the calendar walk is exact); the shape ladder pins moment-vs-daily on real 1Y FF
  windows at 1e-9 (measured 2.3e-10).

Terminal transforms:
- **Averaging:** `rate = num / τ_index`.
- **Compounding / lookback:** `rate = (exp(∫f_window) − 1 + lookback_correction) / τ_index`
  — integrate then EXPONENTIATE (the DF ratio); the misalignment correction is the same moment form.
- **Observation shift:** the window is shifted; still telescopes ⇒ leading term only, ONE sub-period
  with shifted `[a,b]`. No correction needed (it is exact compounding over the shifted window).

### B.3 Cashflow model (additive, opt-in)
`RateObservation` gains an optional moment description WITHOUT breaking the daily path:
```
enum class ObsMethod { SubPeriods, Moment };   // default SubPeriods == today, bit-for-bit
// Moment mode carries: window [a,b], tau_index, day-count moments (sum_tau2, ...), realized, mode
//                       (arithmetic vs compounded), and (compounded) the shift.
```
`SubPeriods` is the current exact path — untouched, so every committed number is preserved. `Moment`
is the new fast path, selected by the extractor/instrument. `rate()` / `float_coupon_pv()` branch on
`ObsMethod`; the compiled `BundleFloatBatch` gains a moment path that gathers `DF(a)`, `DF(b)` and the
curve's `integral2` block instead of per-day sub-periods.

### B.4 Analytic Jacobian
- `∂(∫f)/∂x = w` (existing W row).
- `∂(∫f²)/∂x = 2 M x` (the quadratic form's gradient) — a NEW primitive beside the linear W, cached
  per window from the curve structure. Verify vs AAD to ~1e-9 (mirror the existing compiled-vs-AAD
  tests). The compiled residual assembles both.

### B.5 Extraction (ql/extract.hpp)
- **Observation shift:** confirm QuantLib's `OvernightIndexedCoupon` with `applyObservationShift`
  returns shifted `valueDates()`; if so, obs-shift already works through the existing 1-sub-period
  path — add only a test. This is the first, cheapest deliverable.
- **Lookback / lockout:** read `lookbackDays()` / `lockoutDays()` and build a `Moment`-mode
  observation (window + day-count moments + shift), NOT day-by-day. Match QuantLib's exact coupon rate
  to the gate; if the moment truncation cannot reach `rel ≤ 1e-10`, the exact `SubPeriods` path remains
  available and the instrument chooses (speed vs bit-exactness is then a documented per-instrument
  knob, never a silent approximation).

## Sequencing (each its own gate-green commit)
1. **B-spline region** (Part A) + `bspline_test.cpp`. Self-contained; no pricing changes.
2. **Region `integral2` moment accessor** (B.1) for all regions incl. B-spline + a moment-vs-quadrature test.
3. **Observation shift** (B.5) — extractor confirmation + test. Cheapest pricing win.
4. **Averaging moment fast-path** (B.2–B.4) opt-in, oracle-checked vs QuantLib; existing averaged
   futures unchanged (still `SubPeriods`).
5. **Lookback / lockout** (B.2–B.5) — the compounded moment path + QuantLib extraction + tests.

## Invariants (non-negotiable)
- Both gates green every commit; existing 74 tests keep their exact numbers.
- No index/currency/calendar in engine code (tests only).
- Everything stays templated on `Scalar`, header-only, `is_linear_map`, no hard-coded SIMD width.
- The moment path is ADDITIVE — the `SubPeriods` exact path is never removed, so bit-exactness vs
  QuantLib is always available as the fallback/oracle.
