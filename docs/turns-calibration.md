# Turns as a calibration instrument — design handoff

> **Status:** design only, nothing implemented yet. Branch `claude/curve-turns-calibration-qectmz`.
> **Analysed against commit:** `453aeedb0abf29b6e6124f3957686804b8824b11`
> ("api: expose a frozen-Newton step tolerance on start_streaming", 2026-07-26).
> Read the **[Re-analysis first](#0-re-analysis-first-read-this-before-trusting-the-anchors)**
> section before trusting any file:line anchor below — the code may have moved since.

---

## 0. Re-analysis first (read this before trusting the anchors)

This doc was written from a reading of the engine at the commit pinned above. Line numbers
**will drift**; symbol names and *invariants* are the stable anchors. Before implementing, confirm
the design still holds against current `HEAD`:

```bash
# What changed in the files this design depends on, since the analysis commit:
git diff 453aeedb0abf29b6e6124f3957686804b8824b11..HEAD -- \
  include/swaps/pricing/compiled_book.hpp \
  include/swaps/pricing/curve_spec.hpp \
  include/swaps/pricing/compiled.hpp \
  include/swaps/pricing/cashflows.hpp \
  include/swaps/calibration/compiled_bundle.hpp \
  include/swaps/calibration/problem.hpp \
  include/swaps/calibration/regularize.hpp \
  include/swaps/curve/curve_module.hpp \
  include/swaps/curve/regions.hpp
```

Then walk the **[Invariants we relied on](#6-invariants-we-relied-on-re-verify-these)** checklist.
If an invariant no longer holds, the affected mode/section is flagged there with what to reconsider.

---

## 1. The aim

Add **turns** — localized instantaneous jumps in the overnight forward over a ~1-day window
(turn-of-year, quarter/month-ends, central-bank reserve-maintenance-period ends) — to the curve
calibration.

A turn of size `δ` over window `[a,b]` must push the surrounding **smooth** forwards *down* for a
positive `δ`: an instrument spanning the turn absorbs the extra `δ·(b−a)` of integrated forward, so
the smooth curve compensates by dropping around it. Instruments maturing before the turn are
untouched → the effect stays local. This matches the desk intuition of a turn as an add-on spike on
top of the "unmodified" curve.

**Hard requirement:** do it **without leaving the linear W-cache fast path** (CLAUDE.md §2).

## 2. Core insight (why this is cheap)

A turn is **linear in log-discount space**. The turn contribution to the log-DF integral is

```
∫₀ᵗ f(u) du  =  w(t)·x  +  Σⱼ δⱼ · overlap(t, [aⱼ, bⱼ])
overlap(t, [a,b]) = max(0, min(t,b) − a)
```

`overlap` is a **fixed, closed-form weight** depending only on the turn *times*, never on `x` — no
AAD needed to build it. Because the window is ~1 day, `overlap` is essentially a step: `0` for
cashflows before the turn, the full width `(b−a)` after. So `DF = exp(−(Wx + turn terms))` stays on
the existing `exp(−Wx)` machinery, and the analytic Jacobian, warm/streaming re-cal, and risk ladder
all generalise **for free** because linearity is preserved.

## 3. Design — two modes

### Mode 1 — fixed turns (known / quoted δ) — BUILD FIRST
`δⱼ` is a market input. Precompute a per-registered-point constant
`c = Σⱼ δⱼ · overlap(tᵢ, [aⱼ,bⱼ])` once at setup, and compute `DF = exp(−(Wx + c))`.

- **Jacobian is unchanged**: `c` is constant, so `dDF/dx = −DF·W` still holds. No edit to the
  Jacobian formula.
- **Cost on the hot path: one vector add inside the existing `exp`.** Effectively free.
- ~5-line change plus the `Turn` struct and the templated-path adapter.

### Mode 2 — calibrated turns (solved from turn-sensitive instruments) — SUPERSET
`δⱼ` is a free variable. Append it to the stacked state and append its `overlap` weights as extra
W columns:

```
x_ext = [x ; δ]      W_ext = [ W | L ]      L(i,j) = overlap(timesᵢ, [aⱼ, bⱼ])
```

Then `J = −(G·diagDF)·W_ext` gains the correct turn columns automatically from the matmul — **no new
Jacobian code, no AAD.** `StreamingCalibrator` builds `M = (JᵀJ)⁺Jᵀ` from that
`J`, so it picks up turns automatically; the risk ladder `dx/dq` reports turn sensitivities for
free. A calibrated turn column is just a knot whose basis function is a one-day flat bump.

## 4. Integration points (symbol anchors — confirm each still exists)

Line numbers are **as of the analysis commit**; the symbol is the stable anchor.

| Where | Symbol (grep this) | Change |
|---|---|---|
| `include/swaps/pricing/curve_spec.hpp` (~L18) | `struct CurveStructure` | add `std::vector<Turn> turns;` (`struct Turn { double start, end, size; bool calibrated; };`); include calibrated turns in `n_knots()` |
| `include/swaps/pricing/compiled_book.hpp` (~L83–88) | `CompiledCurveSet::df`, `df_into` | Mode 1: `exp(−(W·x + c))`. Mode 2: matmul against `W_ext` |
| `include/swaps/pricing/compiled_book.hpp` (~L97) | `CompiledCurveSet::logdf_weight` / `finalize` | fill `c` (fixed) and/or append `L` columns (calibrated). Overlap is closed-form — do **not** route it through the AAD `integral_weight_matrix` |
| `include/swaps/pricing/compiled.hpp` (~L33) | `integral_weight_matrix` | leave as-is; turn weights are computed directly, not via this AAD pass |
| `include/swaps/calibration/compiled_bundle.hpp` (~L181) | `CompiledBundleResidual::jacobian_vs` (`-((G * DF.asDiagonal()) * cs_.W())`) | **no edit** — turn columns arrive via `cs_.W()` automatically |
| new, near `curve_module.hpp` | `TurnedCurve<Curve>` adapter | analogue of the spread `SpreadHandle`: adds `Σδⱼ·overlap` to `integral(t)` and `δⱼ·1_{[a,b]}(t)` to `forward(t)`; `discount` composes as `base·exp(−Σδⱼ·overlap)`. Keeps the templated/AAD/QuantLib-oracle path in agreement with the compiled path |
| `include/swaps/calibration/problem.hpp` (~L293) | `CalibrationProblem::residuals` | wrap the built curve in `TurnedCurve` on the templated path |

Turn **dates are DATA** (engine names no calendar — CLAUDE.md §1): thread them in at setup like
meeting dates, as year fractions.

## 5. Anti-patterns (each breaks the fast path)

- **Do not** bump the forward day-by-day inside `obs_forward_sum` (`cashflows.hpp` ~L134) — it
  destroys telescoping and drops coupons off the fused standard-shape paths (`FloatCoupon::standard()`).
- **Do not** model a turn as a `Flat` region — regions must partition `[0,T]` and be non-overlapping
  (`check_region_joins`, `curve_module.hpp` ~L148); a turn *overlaps* the region it sits inside. It
  is an additive overlay, not a partition.
- **Do not** make it a Hermite back-end knot — C¹ smearing spreads the spike across neighbours and
  defeats localization. The turn must be a discontinuous flat bump = its own basis function.
- Keep turn size in **rate units** (a forward bump, `1/time`) contributing the dimensionless
  `δ·window` to log-DF (CLAUDE.md §2 "residuals in RATE units").

## 6. Invariants we relied on (re-verify these)

The design is correct **only if** these still hold at current `HEAD`. Each names what to reconsider
if it has changed.

1. **`DF = exp(−W·x)` is the single DF entry point.** `CompiledCurveSet::df`/`df_into` compute it and
   everything downstream consumes it. → If DF construction moved or gained a nonlinearity, Mode 1's
   `+c` insertion point moves with it.
2. **The bundle Jacobian is `−(G·diagDF)·W`** and reads `W` only through `cs_.W()`
   (`compiled_bundle.hpp`). → If the Jacobian stopped going through `cs_.W()`, Mode 2's "free turn
   columns" claim breaks and turn columns must be added explicitly.
3. **`integral(t) = w(t)·x` is linear for the shipped `flat_hermite` layout**, and `is_linear_map`
   gates the W-cache (`curve_module.hpp`, `regions.hpp`). → Turns rely on adding another linear term;
   still true for any linear scheme, but confirm no default flip to a value-dependent back end.
4. **A spread curve already adds a linear overlay to the base integral** (`logdf_weight` recurses on
   `base`; `SpreadHandle`). This is the *precedent* the `TurnedCurve` overlay copies. → If the spread
   overlay mechanism changed, mirror the new form in `TurnedCurve`.
5. **The regulariser is a second-difference penalty over contiguous knot blocks**
   (`regularize.hpp`, `second_difference_operator` composed via `RegularizedEngine`). → See §7.3: turn knots must be
   excluded from it.
6. **Warm/streaming build `M` from `J` generically via `residual_engine_t<Problem>`.** → Confirms
   Mode 2 turns flow into warm/streaming/risk without bespoke code.

## 7. Open questions to resolve

1. **Identifiability (Mode 2).** A turn is observable only if instruments *bracket* it (serial
   1M/3M SOFR futures spanning vs. not spanning the date, or a turn-of-year FRA/OIS). Without an
   isolating instrument the turn column is collinear with its neighbouring smooth knot → singular
   `JᵀJ`. Decide: require a spanning instrument, or lean on regularisation. If only a quoted turn
   size is available with no isolating instrument → use **Mode 1**.
2. **Turn-date sourcing.** Where do the windows come from (calendar of year/quarter/month-ends, CB
   maintenance dates)? They are DATA extracted at setup; the engine names no calendar.
3. **Regulariser interaction.** The curvature penalty (`regularize.hpp`) must **not** run *across* a
   turn knot — it would penalise the intended discontinuity. Exclude turn columns from the
   second-difference operator (adjust the `(offset, n_knots)` segment layout so a turn knot is not an
   interior knot of a penalised block).

## 8. Validation plan (both gates — CLAUDE.md §3)

- **Pricing (rel ≤ 1e-10):** compiled turned-DF vs the `TurnedCurve` templated path (~1e-15, the
  workhorse compiled-vs-templated oracle); wrap `TurnedCurve` in `CurveTermStructure` so QuantLib
  prices instruments off the turned discount factors.
- **Calibration:** assert first-order optimality `‖Jᵀr‖∞ ≈ 0` — never assert instruments reprice
  exactly (over-determined, CLAUDE.md §2).
- **Jacobian (rel ≤ 1e-6):** turn columns vs bump-and-reprice on `δ`.
- **Behavioral:** a positive turn lowers surrounding forwards vs the no-turn baseline — encodes the
  §1 requirement directly.
- **Perf:** confirm the fixed-turn hot path is neutral (one add) and the calibrated case adds only
  `T` columns.
- **Bookkeeping (same commit):** update `ARCHITECTURE.md`, `tests/ORACLE_TESTS.md` (+ the
  `swaps_oracle_tests` list — enforced by `tools/check_oracle_tests.sh`), and add turn dates to the
  conventions data.

## 9. Recommended build order

1. **Mode 1** — `Turn` struct + the `c` offset in `df`/`df_into` + `TurnedCurve` adapter + the
   oracle and behavioral tests. Small, perf-neutral, covers the common desk workflow of a quoted
   year-end turn.
2. **Mode 2** — append `δ` to the state and `overlap` columns to `W_all`, add turn-sensitive
   instruments, and resolve identifiability (§7.1) + regulariser exclusion (§7.3). Reuses the same
   `overlap` weight; needs no Jacobian/warm/streaming/risk changes.

## 10. Provenance

Design produced from a reading of, at commit `453aeed`:
`curve/curve_module.hpp`, `curve/regions.hpp`, `pricing/compiled.hpp`, `pricing/compiled_book.hpp`,
`pricing/curve_spec.hpp`, `pricing/cashflows.hpp`, `calibration/problem.hpp`,
`calibration/compiled_bundle.hpp`, `calibration/regularize.hpp`. No existing "turn" concept was found
in the tree at that commit.
