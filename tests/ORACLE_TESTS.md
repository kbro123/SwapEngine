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
| `pricing_test.cpp` | OIS par, 3M and 1M futures priced by QuantLib off our curve (`YieldTermStructure` adapter) | `curve_rel` 1e-10 |
| `modular_curve_test.cpp` | QuantLib prices swaps + futures off our **Hermite** `ModularCurve` through the adapter; runtime builder == compile-time layout | `curve_rel` |
| `extract_test.cpp` | Every coupon shape (compounded/averaged OIS, IBOR, futures) NPV / fair rate / forecast fixing vs QuantLib, with negative controls | `curve_rel` |
| `bond_oracle_test.cpp` | Price / yield / accrued / duration / convexity vs `BondFunctions`; curve dirty/clean vs `DiscountingBondEngine`; z-spread vs `ZeroSpreadedTermStructure`; universe sweep | 1e-10 prices; 1e-6 dur/cvx |
| `bond_asset_swap_oracle.cpp` | Par-par asset-swap spread vs `AssetSwap::fairSpread` | **5e-5 abs (≈0.5 bp — loose; E5 tightens or documents)** |
| `bundle_test.cpp` | `MultiCurveBasisMatchesQuantLib` (1 of 11 tests is vs QuantLib; the other 10 are self-consistency and move to `swaps_consistency_tests` in E5) | `curve_rel` |
| `bspline_oracle_test.cpp` | B-spline curve DFs/forwards vs QuantLib term structure; moment average vs exact daily sum | 1e-10; 1e-8 |
| `monotone_cubic_oracle_test.cpp` | Hyman monotone cubic vs `MonotonicCubicNaturalSpline` (forward + primitive) | 1e-11 |
| `tension_oracle_test.cpp` | Tension-spline curve prices OIS consistently with QuantLib, 4 tensions | `curve_rel` |
| `turns_oracle_test.cpp` | Turned curve prices OIS identically in QuantLib; turn δ shifts QuantLib DFs by exp(−δ·overlap) | `curve_rel`; 1e-11 |
| `rfr_coupon_oracle_test.cpp` | RFR compounded/averaged, lookback/lockout/shift coupons vs QuantLib 1.35 | 1e-12 |
| `ois_weekend_test.cpp` | OIS weekend/holiday accrual vs QuantLib and a hand walk | 1e-12 |
| `multicurrency_test.cpp` | 5 of 41 tests vs QuantLib (par rates, payment delay, FF OIS + basis, FF averaging future, instruments); the rest are self-consistency (E5 splits the file; two `SUCCEED()`-only diagnostics to be removed) | `curve_rel`; **two tests at 1e-3 (10 bp) labelled "sub-bp" — E5** |
| `conventions_test.cpp` | Conventions DB strings vs QuantLib's index conventions; one bond price vs QuantLib | 1e-9 |
| `convexity_test.cpp` | Test-side Hull-White futures convexity vs QuantLib (validates the reference builder, not engine code) | 1e-15 |
| `vol_bachelier_ql_oracle.cpp` | Bachelier price/greeks/implied vol vs QuantLib across a grid | 1e-12; **1e-4 gamma/IV** |
| `vol_cms_replication_oracle.cpp` | Hagan G-function vs `GFunctionStandard` (1e-10); replicated CMS convexity vs `NumericHaganPricer` | **15 % relative on the adjustment — near-vacuous; E4.5 fixes the pole and tightens to 1e-3** |
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
| `warm_test.cpp` | Warm cached-Jacobian re-solve vs the engine's own full LM (1e-7) | T3 parity |
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
