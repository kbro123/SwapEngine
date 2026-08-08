# Oracle tests — DO NOT DELETE without reproducing

The engine's correctness rests on one invariant: **every priced cashflow must match QuantLib, cashflow
for cashflow.** The tests that pin that invariant against QuantLib are the *oracle tests*. They are the
most valuable tests in the repo and the easiest to lose in a refactor (they look "redundant" precisely
because the engine reimplements what QuantLib already does — that duplication *is the point*).

## Rules

1. **Never delete or weaken an oracle test to make a refactor pass.** If a change breaks one, the change
   is wrong until proven otherwise, or the oracle comparison must be **reproduced** (re-derived against
   QuantLib and re-committed) — not removed.
2. Every oracle test carries the banner:
   ```
   // @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
   // without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
   ```
3. Oracle tests link QuantLib and live in the **`swaps_oracle_tests`** binary (`tests/CMakeLists.txt`),
   kept separate so the engine's own gate (`swaps_tests`) never links QuantLib.
4. `tools/check_oracle_tests.sh` (run by `verify.sh`) enforces all of the above: it fails the gate if a
   file listed in `swaps_oracle_tests` is missing, lacks the `@oracle-test` banner, or no longer
   references QuantLib. **The CMake `swaps_oracle_tests` list is the registry** — add/remove there and the
   guard follows.

## What each oracle test pins

| Test | Pins against QuantLib |
|------|------------------------|
| `pricing_test.cpp` | Core discounting / coupon PV off our curve as a `YieldTermStructure`. |
| `bond_oracle_test.cpp` | Bond price/yield/accrued/duration/convexity vs `BondFunctions`; curve dirty/clean vs `DiscountingBondEngine`; z-spread vs `ZeroSpreadedTermStructure`; batched universe yield sweep vs `BondFunctions::yield`. |
| `generic_cashflow_test.cpp` | The generic FloatCoupon/FixedCoupon model reprices legacy shapes exactly. |
| `convexity_test.cpp` | Futures convexity adjustment. |
| `extract_test.cpp` | QuantLib → plain-data schedule/coupon extraction round-trips. |
| `calibration_test.cpp` | Calibrated single curve reprices the QuantLib market. |
| `aad_test.cpp` | AAD Jacobian vs QuantLib-priced residuals. |
| `risk_test.cpp` | Analytic risk ladder vs bump-and-reprice on QuantLib. |
| `portfolio_test.cpp` | Portfolio valuation vs QuantLib. |
| `spread_test.cpp` | Fixed-base spread decomposition + calibration (uses `tests/spread_reference.hpp`). |
| `warm_test.cpp` | Warm re-calibration converges to the QuantLib-consistent curve. |
| `streaming_test.cpp` | Frozen-Newton streaming stays on the exact curve. |
| `bundle_test.cpp` | Multi-curve bundle (joint + staged) reprices QuantLib. |
| `compiled_residual_test.cpp` | Compiled W-cache residual == templated/AAD residual. |
| `bspline_oracle_test.cpp` | B-spline curve vs QuantLib term structure. |
| `monotone_cubic_oracle_test.cpp` | Hyman monotone cubic vs QuantLib `MonotonicCubicNaturalSpline`. |
| `rfr_coupon_oracle_test.cpp` | RFR (compounded/averaged, lookback/lockout) coupons vs QuantLib. |
| `ois_weekend_test.cpp` | OIS weekend/holiday accrual vs QuantLib. |
| `migration_guard_test.cpp` | Legacy shapes re-expressed as generic instruments reprice bit-for-bit. |
| `multicurrency_test.cpp` | Cross-currency (FX forward, MtM basis) vs QuantLib. |
| `conventions_test.cpp` | Market-conventions DB (`conventions.json`) vs QuantLib's own index conventions. |
| `tension_regularizer_oracle_test.cpp` | Tension-energy regularizer drops the Jacobian condition number + stays first-order optimal on the QuantLib-built ill-conditioned EUR trio. |
| `tension_oracle_test.cpp` | Tension-spline curve vs QuantLib term structure. |
| `turns_oracle_test.cpp` | Turned curve (turn overlay) prices OIS identically to QuantLib; the δ shifts QuantLib DFs by exp(−δ·overlap). |
| `vol_bachelier_ql_oracle.cpp` | Bachelier (normal) option kernel — price/greeks/implied-vol equal QuantLib across a strike/vol/expiry grid. |
| `vol_cms_replication_oracle.cpp` | Hagan static-replication CMS — the standard-model G matches QuantLib `GFunctionStandard` exactly; the replicated convexity tracks `NumericHaganPricer`. |
| `bond_oracle_test.cpp` | Bond pricing (street + curve) — clean/dirty/YTM/duration/convexity/z-spread + universe sweep vs QuantLib `BondFunctions`/`DiscountingBondEngine`/`ZeroSpreadedTermStructure`. |

Reference builders (`reference_curve.hpp`, `reference_bundle.hpp`, `reference_multicurrency.hpp`) build
the QuantLib-consistent markets these tests calibrate to; they are part of the oracle surface.
