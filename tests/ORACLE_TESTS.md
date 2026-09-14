# Oracle tests and consistency tests — the two QuantLib-linked registries

PRINCIPLES.md P8: a test **earns** the word *oracle* only by comparing an **engine number to a QuantLib
number**. Everything else is labelled for what it is. Until 2026-09-08 eleven self-consistency files carried
the `@oracle-test` banner and this document described them as "vs QuantLib"; an independent review found
that only ~46 of the 135 tests in the oracle binary compared to a QuantLib number. This registry is now honest.

## Rules

1. **Never delete or weaken an oracle or consistency test to make a change pass.** Reproduce the comparison
   instead (re-derive against QuantLib and re-commit).
2. **The CMake lists ARE the registries**: `swaps_oracle_tests` and `swaps_consistency_tests` in
   `tests/CMakeLists.txt`. `tools/check_oracle_tests.sh` (run by `verify.sh` after the build) fails the gate if a
   listed file is missing, lacks its banner, (oracle only) has no QuantLib reference, or has fewer `TEST`s or
   fewer `EXPECT_/ASSERT_` lines than `tests/oracle_assertions.lock` — comment text stripped, so a
   commented-out assertion counts as gutting. With `--build` it also fails if a binary is missing or lists fewer
   tests than locked. Counts may only grow silently; a reduction is re-locked with `--update` in a commit
   that says why. `tools/selftest_guards.sh` proves the guard trips on a gutted file and on deleted tests.
3. Banners: `// @oracle-test — …` / `// @consistency-test — …` on the first line.
4. QuantLib is **required** for a gated build (root `CMakeLists.txt` fails to configure without it unless
   `-DSWAPS_ALLOW_NO_ORACLE=ON`, which marks the build non-gated).

## `swaps_oracle_tests` — engine number vs QuantLib number

| Test file | What is compared to QuantLib | Tolerance |
|---|---|---|
| `pricing_test.cpp` | OIS par, 3M and 1M futures priced by QuantLib off our curve (`YieldTermStructure` adapter); the SHIPPED averaged-observation and par-swap builders (DB → `swap_conv` → `build::par_swap`) vs a QuantLib OIS on the same conventions, 1y–30y, with negative controls | `curve_rel` 1e-10; builders measured 7.6e-17 |
| `calibration_oracle_test.cpp` | A four-curve CALIBRATION (SOFR, FF on SOFR discounting, ESTR, EURIBOR-6M on ESTR discounting): QuantLib's own instrument objects define every residual, our LM solves it, and the **knot forwards** must match the ones our own instruments produce. Both arms build OUR curve from the same `x`, so the comparison is exact and in the unit a client consumes | quotes 5e-13 (measured 1.1e-16); knot forwards 1e-4 bp (measured 9.2e-9 bp) |
| `modular_curve_test.cpp` | QuantLib prices swaps + futures off our **Hermite** `ModularCurve` through the adapter; runtime builder == compile-time layout | `curve_rel` |
| `extract_test.cpp` | Every coupon shape (compounded/averaged OIS, IBOR, futures) NPV / fair rate / forecast fixing vs QuantLib, with negative controls | `curve_rel` |
| `bond_oracle_test.cpp` | Price / yield / accrued / duration / convexity vs `BondFunctions`; curve dirty/clean vs `DiscountingBondEngine`; z-spread vs `ZeroSpreadedTermStructure`; universe sweep | 1e-10 prices; 1e-6 dur/cvx |
| `bond_asset_swap_oracle.cpp` | Par-par asset-swap spread vs `AssetSwap::fairSpread` | **5e-5 abs (≈0.5 bp — loose; E5 tightens or documents). Likely cause (hypothesis, 2026-09-13, not measured): settle == the evaluation date, so `DiscountingSwapEngine` drops the par upfront on that date (predicted gaps 3.3–3.8e-5); `asset_swap_oracle_test.cpp` settles T+1 and agrees at 1e-10** |
| `asset_swap_oracle_test.cpp` | The `asset_swap` VERB end to end (run_json -> calibrated bundle -> float leg -> spread) vs `AssetSwap(parSwap=true)` on identical DFs and QuantLib's own schedules: spread in all three price modes, annuity, curve dirty/clean, accrued, the proceeds identity (par / dirty), a leap-day maturity's float roll | 1e-10 (`tol::curve_rel`) |
| `risk_ladder_oracle_test.cpp` | The DELTA LADDER of the `portfolio_risk` and `generate_risk` verbs and the `risk` operator M = dx/dq (n_knots × n_res), on the calibration oracle's four-curve bundle and a five-trade OIS/IRS book, vs QuantLib bump-and-RECALIBRATE: QuantLib instruments define the residuals, each quote is bumped ±0.25 bp, the curve is re-solved against QuantLib (chord Newton, |r| ≤ 1e-14) and a QuantLib book is repriced (central FD). Verb NPV and knots vs QuantLib too. Negative controls: a forward-difference bump must fail (measured 423× tol); D dropped on an in-band banded row must fail (6e5×) while both banded verbs still match. At 1 bp two rows missed a 2e-5 tolerance by 1.4× and fell exactly h² at 0.25 bp: truncation, so the bump shrank | `5e-6·max(\|ref\|,1)` per unit quote: measured worst ladder 0.35× (1.7e-6 rel), operator 0.20×; NPV 5e-9; knots 1e-11 |
| `bond_rv_oracle_test.cpp` | The three RV VERBS end to end through run_json. `bond_universe`: accrued, yield↔clean, modified duration, convexity vs `BondFunctions` on QuantLib's own `FixedRateBond` (settlement rolled by QuantLib on the DB calendar, T+1 over Veterans Day; seasoned, final-period and when-issued bonds; street and Treasury method). `govvie_fit` NS/Svensson: the objectives do NOT match `FittedBondDiscountCurve` (QuantLib frees κ, 1/duration weights, Simplex), so QuantLib EVALUATES at the engine's parameters (`maxEvaluations=0`): its discount function, `BondHelper` residuals and `FittingCost`, plus first-order stationarity of QuantLib's cost with κ held. `govvie_fit` spline (no QuantLib counterpart): residuals and z-spreads vs `DiscountingBondEngine` / `ZeroSpreadedTermStructure` / `BondFunctions::zSpread` off the engine's fitted curve via the adapter. `swap_spread` HEADLINE: benchmark yield vs `BondFunctions::yield`, anchor vs `MakeOIS` maturity, the emitted swap row vs `OvernightIndexedSwap::fairRate`, composed rows quote fairRate − yield (`matched_maturity` excluded: open gap). Negative controls: settlement calendar, final-period rule, explicit settle, Treasury stub, κ mapping, weights, z = 0, spot lag | 1e-10 (`tol::curve_rel`); anchor `literal`; Simple-stub dur/cvx 1e-8/1e-5 vs FD of QuantLib price (as `bond_oracle_test.cpp`); cost bound derived from the residual tolerance; stationarity cosine 1e-6. Passed its first run at these tolerances (2026-09-14); margins not printed |
| `scenario_oracle_test.cpp` | The `scenario` and `scenario_grid` VERBS end to end on a DB-built 3-curve bundle (ESTR with a year-end turn, EURIBOR-6M on ESTR discounting, SOFR) and a book of 4 dated OIS/IRS from `swap_conv` → `build::par_swap`: every move's book `npv` / `npv_delta` and sampled DF / zero, and every grid cell's `npv` / `pnl`, vs QuantLib `OvernightIndexedSwap` / `VanillaSwap` on QuantLib schedules and indices, priced on (L1) our curve rebuilt at the verb's `base.x` + shift and (L2) `ZeroSpreadedTermStructure` over the base. A PARALLEL move on a SPREAD curve (EURIBOR-6M over ESTR) moves it once, as QuantLib reads it -- the pin that found the 2x double move fixed in b5f3a4d. Controls: SC1 override vs add rule (both verbs), sign, turn δ untouched, wrong role. Out of scope: MtM xccy / FX moves, turn calibration | measured worst: NPV 2.6e-8 on 1e8 gross, DF 3.3e-16, zero 3.9e-15; locked at NPV 1e-6, DF/zero 1e-13 |
| `portfolio_verb_oracle_test.cpp` | The `portfolio`, `portfolio_risk`, `risk` and `transform` VERBS end to end on a hard square bundle (SOFR, ESTR, EURIBOR-6M as a spread over ESTR) and a book of TYPED TRADES booked through `book.trades` (value_date, curve_roles, CSA): per-trade and book NPV and PV01 vs QuantLib `OvernightIndexedSwap` / `VanillaSwap` on QuantLib's own schedules and signs (PV01 as +1bp on every curve once, written out test-side); `curve_grad`, the risk operator and the ladder vs QuantLib's central-difference Jacobian and its inverse; `transform` vs J_src·J_this⁻¹ (reg OFF only). Controls: pay direction, discount role (two-sided), fixed day count, a SEASONED trade must be refused (this control found SW1: a stateless session forecast the realized part), par feedback, the ladder sees discounting, the transform sees the forecast role, the xccy block is zero | measured: NPV 5.5e-16/notional (book 7.0e-10 on 7e7), PV01 3.4e-10 rel, gradient 5.2e-10, M 2.1e-9, ladder 3.1e-8, transform 1.2e-8 (cond 475); locked NPV `curve_rel`·N, PV01/gradient `jacobian_rel`, M/ladder/transform 3e-6 |
| `calib_report_oracle_test.cpp` | The `calib_report` VERB end to end (run_json object seam) on the four-curve bundle of `calibration_oracle_test` OVER-DETERMINED by 8 no-knot rows + 1 exact duplicate (36 rows x 27 knots, so identifiability is not trivially 1): every quote's model / residual vs QuantLib `fairRate` / futures at the calibrated x; singular values, condition number, rank and per-quote identifiability vs Eigen BDCSVD of a QuantLib central-difference Jacobian (hard pins; bands / regulariser / FX have no QuantLib counterpart). Tolerances are DERIVED at runtime from QuantLib's numbers (a Richardson FD error bound, Weyl for sigma, a projector perturbation bound) with ceilings | measured: model 1.1e-16 (bound 5e-13); ||E||_2 4.9e-8; sigma 4.0e-9 (bound 4.9e-8); identifiability 3.3e-11 (bound 1.9e-5); kappa 565.63 = QuantLib; sigma_min 5.1e-3 |
| `pnl_reference_test.cpp` | The `pnl` VERB end to end on an ALIGNED horizon (t1 = the book's first coupon date, dt = 1.0): npv_t0 / npv_t1 / total vs QuantLib OIS NPVs off our curve; carry vs `DiscountingSwapEngine(npvDate = settlement = t1)`; roll vs the curve re-anchored at t1; dq vs QuantLib fair rates; the market ladder vs J⁻ᵀg from QuantLib bumps; the residual. Plus a T5 flat-curve closed form (roll ≡ 0, carry = financing). A coupon straddling t1 has NO oracle (the engine's documented approximation; QuantLib would use realized fixings) | NPVs 1e-4 on 1e6 (`curve_rel`·N); dq 1e-12; ladder 1e-5 rel with κ·FD asserted < ¼ (measured κ 190.5 × FD 1.3e-10 = 2.5e-8); flat 1e-6. First run passed; QuantLib components npv_t0 -14034.7, carry 1638.22, roll 680.96, market -219.33, residual 1074.15; per-component margins not printed |
| `bundle_test.cpp` | `MultiCurveBasisMatchesQuantLib` (1 of 11 tests is vs QuantLib; the other 10 are self-consistency and move to `swaps_consistency_tests` in E5) | `curve_rel` |
| `bspline_oracle_test.cpp` | B-spline curve DFs/forwards vs QuantLib term structure; moment average vs exact daily sum | 1e-10; 1e-8 |
| `monotone_cubic_oracle_test.cpp` | Hyman monotone cubic vs `MonotonicCubicNaturalSpline` (forward + primitive) | 1e-11 |
| `tension_oracle_test.cpp` | Tension-spline curve prices OIS consistently with QuantLib, 4 tensions | `curve_rel` |
| `turns_oracle_test.cpp` | Turned curve prices OIS identically in QuantLib; turn δ shifts QuantLib DFs by exp(−δ·overlap) | `curve_rel`; 1e-11 |
| `rfr_coupon_oracle_test.cpp` | RFR compounded/averaged, lookback/lockout/shift coupons vs QuantLib 1.35 | 1e-12 |
| `ois_weekend_test.cpp` | OIS weekend/holiday accrual vs QuantLib and a hand walk | 1e-12 |
| `multicurrency_test.cpp` | 5 of 41 tests vs QuantLib (par rates, payment delay, FF OIS + basis, FF averaging future, instruments); the rest are self-consistency (E5 splits the file; two `SUCCEED()`-only diagnostics to be removed) | `curve_rel`; **two tests at 1e-3 (10 bp) labelled "sub-bp" — E5** |
| `conventions_test.cpp` | Conventions DB strings vs QuantLib's index conventions; one bond price vs QuantLib | 1e-9 |
| `scheme_value_oracle_test.cpp` | Linear / NaturalCubic / Hermite interpolant VALUE and PRIMITIVE vs `LinearInterpolation` / `CubicNaturalSpline` / `CubicInterpolation(Parabolic)` (E5.3, 2026-09-10) | `literal` 1e-12 (measured 7e-18) |
| `sabr_ql_oracle_test.cpp` | General-β Black SABR vol `sabr_black_vol` vs `sabrVolatility`, β∈{0,¼,½,¾,1} × 4 expiries × ±180 bp (E5.3, 2026-09-10; the NORMAL expansion has no QuantLib counterpart — open gap) | 1e-13 (measured 2.5e-16) |
| `convexity_test.cpp` | Test-side Hull-White futures convexity vs QuantLib (validates the reference builder, not engine code) | 1e-15 |
| `vol_bachelier_ql_oracle.cpp` | Bachelier price/greeks/implied vol vs QuantLib across a grid | 1e-12; **1e-4 gamma/IV** |
| `calendar_ql_oracle_test.cpp` | EVERY DB calendar vs QuantLib's calendar of the same market, day by day 2024-01-01..coverage year (rule-based markets through 2035; tabulated ones to where QuantLib's tabulation ends); documented exceptions only (IDR May-1); calendars with no oracle must be documented (ARS, RUB) | exact (0 mismatching days) |
| `ql_test_isolation.cpp` | (listener fixture: clears QuantLib's global `IndexManager` before each test; not a test) | — |

## `swaps_consistency_tests` — QuantLib builds the market; the engine is compared to ITSELF or to hand formulas

These are valuable (cross-path parity caught 13 of 17 kernel mutations in the 2026-09-08 audit) and protected by
the same lock. They are **not** oracles: the "market" is `residuals(x_true)` from the engine's own model at a
formula-generated `x_true` (`reference_curve.hpp`, `reference_bundle.hpp`, `reference_multicurrency.hpp`), and
"recovers x_true" proves solver convergence on a self-generated problem, never agreement with QuantLib.

| Test file | What is actually compared | Class |
|---|---|---|
| `calibration_test.cpp` | LM recovers the self-generated market; rms / stationarity thresholds | T2 calibration |
| `aad_test.cpp` | AAD Jacobian vs the engine's own central finite differences (`jacobian_rel` 1e-6); `sv_min > 0` | T3 parity |
| `risk_test.cpp` | Analytic IFT delta ladder vs the engine's own bump-and-recalibrate (1e-4 rel) | T3 parity |
| `portfolio_test.cpp` | Compiled batched book vs the scalar templated kernel (1e-10) | T3 parity |
| `spread_test.cpp` | Fixed-base spread decomposition vs `spread_reference.hpp` (test-only re-implementation) + own FD | T2/T3 |
| `streaming_test.cpp` | Frozen-Newton streaming round-trips to the tick (1e-8); prefetch assertion gated by `SWAPS_TIMING_ASSERTS` | T3 parity |
| `compiled_residual_test.cpp` | W-cache residual/Jacobian vs templated/AAD (1e-12 / 1e-9) | T3 parity |
| `generic_cashflow_test.cpp` | Generic `FloatCoupon` identities vs hand formulas (1e-15); legacy shapes reprice exactly | T3/T5 |
| `migration_guard_test.cpp` | Legacy shapes re-expressed as generic instruments reprice bit-for-bit; QuantLib coupon date facts | T3 parity |
| `tension_regularizer_oracle_test.cpp` | Regulariser drops the Jacobian condition number; first-order optimal on the self-consistent EUR trio (**no QuantLib in the file**; `raw_err < 1e-2` = 100 bp — E5 tightens) | T2 calibration |

## Reference builders

`reference_curve.hpp`, `reference_bundle.hpp`, `reference_multicurrency.hpp`, `conventions_ql.hpp`,
`spread_reference.hpp` build the QuantLib-consistent schedules/markets the tests use; they are part of the
gate surface and covered by `tools/check_oracle_tests.sh` indirectly (deleting one breaks the build).

## Known honesty items carried into E5 (test rewrite)

- Split `bundle_test.cpp` and `multicurrency_test.cpp` so each file is one class.
- Value pins for **Flat / Linear / NaturalCubic / Hermite** through the adapter (Hermite now has one via
  `modular_curve_test.cpp`; a Bessel-tangent swap passed all 33 non-oracle tests in the audit).
- Compiled `FxForward` / `XccyMtmBasis` rows vs QuantLib; SABR β∈(0,1) vs `SabrSmileSection`; ZCIS / CDS vs
  QuantLib engines; CME conversion-factor table rows.
- Loose tolerances: CMS 15 %, asset-swap 5e-5 abs, multicurrency 1e-3 "sub-bp", tension-regularizer 1e-2.

## Coverage BY LAYER — what a QuantLib number actually reaches (2026-09-10)

The tables above answer *"what does each oracle FILE compare?"*. This answers the dual question, the one the
averaged-weight bug turned on: **for a given piece of shipped code, is there any oracle that could have caught
a wrong number in it?** That bug lived in `build/observations.hpp`; every oracle that might have caught it
built its observations **test-side**, so no oracle's include closure entered the shipped builder, and eleven
green oracles said nothing about it. A file-level guard cannot see that; `tools/oracle_coverage.py` can.

Run `python3 tools/oracle_coverage.py` for the live table. `--check` (a `verify.sh` gate) fails when a header
**loses** its oracle reach — coverage usually erodes as an unnoticed side effect of a refactor, never as a
decision. Reachability is a **floor, not a certificate**: an oracle can include a header and still compare only
numbers that never touch it. A header with no oracle in its closure provably has no QuantLib check; one with an
oracle merely might.

| Layer | Reached | The gap, and whether it matters |
|---|---|---|
| `ad`, `ql` | 1/1, 2/2 | complete |
| `pricing` | 6/9 | `bond_future.hpp`, `ndf.hpp`, `fixings.hpp` — NDF numbers and the past/future fixing split have no QuantLib comparison. Bond-future conversion factors have NO QuantLib oracle either, but ARE pinned to CME's published factors (T1: `tests/delivery_basket_test.cpp`, end to end `tests/bond_future_reference_test.cpp`, 2026-09-13) |
| `build` | 10/13 | `credit_instruments.hpp`, `inflation_instruments.hpp`, `swap_spread.hpp`. (`instruments.hpp` + `conventions.hpp` + `ref_data.hpp` were the S1 gap below and are now oracled by `Pricing.ParSwapFromTheShippedBuilderMatchesQuantLib`.) |
| `calibration` | 13/21 | `risk.hpp` (the IFT delta ladder clients hedge on) and `consistent_risk.hpp` are ORACLED since 2026-09-14 by `risk_ladder_oracle_test.cpp` (QuantLib bump-and-recalibrate); still no oracle: `bond_fit`, `pnl_explain`, `structure_fingerprint`, `diagnostics`, credit/inflation problems |
| `curve` | 2/5 | `hazard.hpp`, `inflation.hpp`, `parametric.hpp`: **credit and inflation curves have no oracle at all**, though QuantLib ships engines for both |
| `vol` | 3/8 | oracled: Bachelier, general-β SABR vol, `normal`. Not: `swaption.hpp`, `fx_black.hpp`, `fx_vol_surface.hpp`, `sabr_calibration.hpp`, `vega_ladder.hpp` |
| `portfolio` | 1/4 | book aggregation is cross-path parity only (compiled vs templated) — a shared error cancels |
| `api` | 3/21 | `bond.hpp`, `codec.hpp`, `bundle_api.hpp` reached since 2026-09-13 (`asset_swap_oracle_test.cpp` drives the verb); the rest see below — measured by verb, not by include |
| `market`, `trade`, `csa`, `xva`, `derive` | 0 | mostly containers and role plumbing (`quote.hpp`, `currency.hpp`, `trade.hpp`, `csa.hpp`) where there is no independent number to compare; `derive/asset_swap.hpp` and `xva/exposure.hpp` DO produce numbers and do not have one |

**Calibration, not just pricing.** Every oracle above except `calibration_oracle_test.cpp` compares a QUOTE at a
FIXED curve. That checks the pricing kernel; it does not check what a client actually consumes — the curve the
calibration lands on — and an error in the instrument ASSEMBLY arrives there magnified by the inverse Jacobian.
`calibration_oracle_test.cpp` closes that: QuantLib's instruments define the residuals, our LM solves them, and
the answer is compared in knot forwards. It found **E3** (`ASSUMPTIONS.md`) on its first run — an averaged
window opening on a non-business day dropped that accrual, pricing the August 2026 FF contract at 29/31 of the
correct rate. Its scope is what QuantLib can express: a soft BAND, a TURN JUMP as a free variable, a BUTTERFLY
as one combination residual and an MtM XCCY basis have no QuantLib counterpart and stay on cross-path parity.

**The two S1 gaps.**

1. **`build/instruments.hpp` — the shipped instrument builders (`par_swap`, `basis_swap`, `xccy_mtm_basis`,
   `zero_coupon_swap`, `float_leg_from`) are reached by nine tests and by no oracle.** This is precisely the
   class of the item-17/E2 bug: the builder assembles conventions, schedules and observations into the rows
   the calibration consumes, and it was wrong by 365/360 for a year. Every oracle prices instruments the
   *fixture* built. The fix is one test that builds a swap **through the shipped builder from the conventions
   DB** and prices it against QuantLib's `VanillaSwap`/`OvernightIndexedSwap` — item 2 of this programme did
   exactly that for one observation shape (`pricing_test.cpp`, averaged future).
   **CLOSED 2026-09-10** by `Pricing.ParSwapFromTheShippedBuilderMatchesQuantLib`: the engine is given a
   currency, an index and a maturity, and everything else — spot lag, payment lag, roll convention, both
   frequencies, both day counts, the compounding mode — comes from the DB row through `swap_conv` and
   `build::par_swap`; QuantLib's OIS is built to those same conventions (not `MakeOIS` defaults) at 1y–30y.
   Agreement is 7.6e-17. Its negative controls also pin *why* E2 hid: on a **compounded** leg the float day
   count cancels **exactly** (`tau_pay / tau_index` = 1 when the accrual and observation windows coincide),
   so getting it wrong is invisible there — it was only ever visible on an averaged leg. The fixed day count
   and the fixed frequency do move the rate, and the test demands that they do. Still open in this layer:
   `swap_spread.hpp`, `credit_instruments.hpp`, `inflation_instruments.hpp`.
2. **No oracle names any of the 28 `run_json` verbs.** Every verb is covered by shape and smoke tests only, so
   the request → kernel assembly — role wiring, conventions lookup, unit and sign conventions on the way out —
   is unchecked against an independent number, even where the kernel below it is well oracled. (`portfolio` and
   `vol_cube` are not named by any test at all.)

Neither gap means the arithmetic is unchecked; both mean the **assembly** around it is, which is where the last
S0 lived.
