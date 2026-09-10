# ASSUMPTIONS.md — SwapEngine (every approximation, threshold, premise and silent default, with its evidence)

Living register, first cut 2026-09-10 from the E4.G fresh-eyes audit (`../_review-2026-09/review-2026-09-09/
E4G-assumptions-register.md`, probes `G-*.cpp`). `PRINCIPLES.md` is the law; this file is the inventory of
where the engine deliberately approximates, routes on a threshold, or relies on a premise -- and what pins each
one. **Rule (P8/P11): an assumption that is not in this file does not exist; an entry with no pinning test is a
bug in waiting.** Update the entry in the same commit as the code.

Classes: **A** numerical approximation that changes a value · **B** routing threshold · **C** mathematical
premise stated as fact · **D** silent fallback / default · **E** numeric hygiene.
Status: **PINNED** (test named, assertion read) · **OPEN** (needs work; the verdict says what) · **RETIRED**
(the premise was false and has been removed) · **MOOT**.

## Tolerances shared by the tests (`tests/tolerances.hpp`)

`curve_rel = 1e-10` (curve values vs an oracle) · `jacobian_rel = 1e-6` · `analytics_rel = 1e-10` ·
`kRankThreshold = 1e-10` (`calibration/residual_engine.hpp`: THE relative rank threshold on singular values;
every consumer -- LM completion, the streamer's operator, the prefetch worker, WarmCalibrator, generate_risk's
null completion -- uses it since 2026-09-10).

## Retired premises (kept so nobody re-introduces them)

| # | where | the premise | why it was false | retired |
|---|---|---|---|---|
| R1 | `calibration/regularize.hpp` | the tension-energy operator's cubic fit is exact on UNIFORM B-spline breakpoints | de Boor averaging moved the breakpoints (25 % energy error, probe G-2) | 3b8571f: regions expose `pieces()` + analytic `forward_d1/d2`; the operator integrates the true pieces |
| R2 | `calibration/regularize.hpp` | every scheme's piece is a cubic, so per-piece 3-pt Gauss is exact | `Scheme::Tension` pieces are sinh/cosh (18–100 % error, growing with σ) | 3b8571f: composite ⌈σh⌉×6-pt on Tension pieces; oracle test 1e-12 on every scheme |
| R3 | `portfolio/compiled_multi.hpp`, `cashflows.hpp`, `bundle_api.cpp`, `compiled_bundle.hpp`, `problem.hpp` | "the MtM leg is not a single exp(−Wx)" ⇒ every xccy position rides AAD | the leg IS a product of registered DFs (`BundleFloatBatch::add_mtm`, 2026-09-09) | 8569197: Kind::Xccy compiles as two rows; the five texts rewritten |
| R4 | `pricing/cashflows.hpp`, `compiled_book.hpp` | a fixed 32×2 / 8×2 Gauss rule integrates f², f³ "to ~1e-15" | not knot-aligned: 3.3e-3 of ∫f² (1.7e-7 of the rate) on the Flat-front Fed-funds shape | 8eeb9f0: knot-aligned 4/5-pt Gauss per piece in both kernels (`moment_gauss_nodes`) |
| S1 | `calibration/structure_fingerprint.hpp` | an edge-only hash is collision-safe for structural edits | interior sub-period / weight edits collided (probe G-4) | MOOT since e09f472: no hash gates anything; `structure_equal` compares full vectors |
| S2 | `cashflows.hpp` `xccy_mtm_leg_pv`, `compiled_book.hpp` | the observation window is the accrual period and a reset is always a curve-implied forward | a seasoned coupon's window shrinks on resolution; a past reset was discounted at DF ≡ 1 | 18f91c9: `FloatCoupon::accrual_*`, `reset_fx`; past reset without it REFUSED; seasoned MtM → AAD block |
| S3 | `calibration/background_jacobian.hpp`, `streaming.hpp` | the prefetch worker's M-only hand-over is exact for any stream | wrong for banded / regularised / over-determined streams; un-thresholded QR (|M| 6e14) | 12c4cb8: COD at `kRankThreshold`; prefetch armed only for square, unbanded, unregularised |
| S4 | `api/generate_risk.cpp` | a knot is "unseen" when its normal-equation eigenvalue < 1e-9·max | that is σ < 3e-5·σmax, five orders looser than `kRankThreshold` | 12c4cb8: JacobiSVD rank at `kRankThreshold` |
| S5 | `calibration/warm.hpp` | the warm problem is full column rank | no protection on a rank-deficient bundle | 12c4cb8: thresholded COD |
| C1 | `calibration/streaming.hpp` | a failed tick leaves a usable anchor | the anchor sat at the divergent iterate; later ticks reported converged on a 1289 bp stale curve | ba6708e: `StreamStatus`, anchor restored to the last committed (x, q) |
| C4/G1 | `calibration/regularize.hpp` | (see R1/R2) | | |
| E1/E2/B12 | curve factory, `validate_problem` | callers never pass an empty curve, a knot at t ≤ 0, an empty leg | they did (segfault, silent W corruption, NaN) | 6e7ad6d: refused with a message |

## Approximations that change a value (class A)

| # | where | what is approximated | magnitude (measured) | pinned by | status |
|---|---|---|---|---|---|
| K1 | moment path (`docs/bezier-and-moments.md` Part B, `cashflows.hpp` `obs_numerator`, `build/observations.hpp` `moment_observation`) | daily arithmetic average ≈ ∫f + ½⟨τ²⟩∫f² + ⅙⟨τ³⟩∫f³ | uniform days: 4e-11 smooth, 4e-10 on a 1Y window with 25 bp steps inside; real calendar floor 3.4e-9 (re-measured 2026-09-10); 30/360 refused (constant-ratio guard 1e-10) | `bspline_oracle_test::MomentAveragingMatchesQuantLibRealCalendar` 1e-8; `hotpath_shapes_test::MomentPathAgrees…` 1e-9 (measured 2.3e-10); `moment_quadrature_test` (kernels vs an independent rule 1e-12, compiled == templated 1e-14) | PINNED |
| T1 | `calibration/lm.hpp` rank-deficient completion | anchor rows `w = 1e-8·max pivot` bias identified directions by (w/σ)² | claimed ppb; never measured | `tension_regularizer_oracle_test::FirstOrderOptimalAndDampsTheNullWander` (1e-2 only) | OPEN: pin ≤ 1e-8 rel vs an unanchored LM on the full-rank sub-problem |
| T2 | `calibration/gamma.hpp` `market_gamma_gn` | omits the calibration-curvature term Σ dNPV/dx·d²x/dq² | bounded 1e-3·scale on one fixture (diagonal only) | `gamma_test::MarketGammaGaussNewtonTransportIsConsistentAndSymmetric` | OPEN: test the omitted term on an off-par book, state the bound |
| T3 | `calibration/pnl_explain.hpp` roll leg | shifted times floored at 0; an elapsed coupon is not re-fixed | residual < 1 % of total (never isolated) | `pnl_explain_test::PureMarketMoveIsAllMarket` | OPEN: isolate the roll error vs a re-fixed book (comment corrected 2026-09-10: a negative time gives DF ≡ 1, not DF > 1) |
| T5 | `pricing/bond.hpp`, `portfolio/bond_universe.hpp` yield Newton | unsafeguarded Newton from y0 = 5 % | fine near par; can overshoot on distressed prices / negative yields | `bond_oracle_test` (near-par sweeps) | OPEN: distressed and negative-yield cases + a bracket |
| T6 | `vol/fx_vol_surface.hpp` | Acklam Φ⁻¹, |err| < 1.15e-9 | harmless for strike placement | no delta→strike→delta round trip | OPEN: round trip under all four `DeltaConv` |
| T7 | `build/credit_instruments.hpp` CDS protection leg | default paid at period end, `prot_steps` per period | O(h) bias never stated; ISDA accrual-on-default omitted (+1.5 bp at 300 bp, +29 bp at 1220 bp, probe F-7b) | `credit_test::FlatHazardRepricesParSpreadExactly` 5e-4 abs (1.7 % rel) | OPEN (E4.E F5): AoD term, ≥16 steps, QuantLib `MidPointCdsEngine` oracle |
| T8 | `vol/sabr.hpp` | ζ/x̂(ζ) → 1 inside |ζ| < 1e-7 | ≤ 5e-8 rel vol discontinuity at the guard edge | `vol_sabr_test::ContinuousThroughAtm` 1e-6 | OPEN: evaluate the series inside the guard or pin at 1e-8 |
| S6 | `vol/cms_replication.hpp` | replication strip below 0.2·F dropped | 0.73 % of the adjustment (probe F-2) | `vol_cms_replication_oracle` at 15 % (20× looser than the truncation) | OPEN: state the 0.7 % in the header; pole-avoiding QuantLib configuration as the oracle |
| D5 | `cashflows.hpp` xccy | MtM basis = constant-notional basis under deterministic curves | the FX-vol convexity term ~ σ_FX·σ_r·ρ·T² (tenths of a bp at 10y) is never shown | — | OPEN: document the order of magnitude |
| D8 | `xva/exposure.hpp` vs `api/var.cpp` | PFE = order statistic of max(V,0) at ⌊q(n−1)⌋; VaR = type-7 interpolated quantile | two quantile definitions | `xva_exposure_test`, `var_test` 1e-9 | OPEN: document or unify |
| D12 | `vol/cms.hpp` | linear TSR "reduces to replication at first order" | no measured difference | closed-form tests only | OPEN: measure vs `cms_replicated_forward` |
| K21 | `curve/inflation.hpp` seasonality | 12 monthly log factors, linear within a month, anchored to the base reference month (`phase`, 2026-09-10) | whole-year maturities cancel for any value date | `inflation_test::SeasonalityShiftsMonthlyNotAnnual`, `SeasonalityPhaseAnchorsToTheReferenceMonth`, `ApiPeriphery.InflationSeasonalIsAnchored…` | PINNED (daily index interpolation between prints still OPEN, E4.8) |
| K26 | `pricing/bond.hpp` stub conventions | App-B vs street: ~7e-6 of price on a 6y note | stated | `bond_oracle_test::TreasuryMethodMatchesSimpleThenCompounded` | PINNED |
| K34 | `api/exposure.cpp` | Gaussian parallel-shock HW1F proxy, book AGED per node (2026-09-10) | labelled illustrative | `exposure_netting_test::TheBookIsAgedToEveryNode` | PINNED |

## Routing thresholds and constants (class B)

| # | where | rule | pinned by | status |
|---|---|---|---|---|
| K2 | `ad/dual.hpp`, `aad_block.hpp` | pooled dual when touched width ≤ `kPooledMaxW = 48`, else heap; bit-identical | `aad_block_pooled_test` (== 0.0) | PINNED |
| K3 | `calibration/streaming.hpp` Options | `step_tol 1e-9`, `max_frozen 8` (hard cap; the ADAPTIVE STALL refreshes when the contraction-predicted remaining steps, with a factor-2 margin, cost more than a refresh whose break-even is measured at construction), `max_refresh 6`, `max_steps 64`, `refresh_drift 1e-3` (measured 0.4 bp at 30 bp), `prefetch_drift 5e-4`; `predict_convergence` (stop when the contraction bounds the next step below step_tol/20); `rescale_update` (a band re-scale is a rank-one Sherman–Morrison update of (JᵀJ+RᵀR)⁺, falling back to a factorisation when 1+βs < 1e-3); a non-converged tick is never committed | `streaming_test`, `streaming_contract_test` (predictive vs fully iterated 1e-10), `StreamingBand::RescaledOperatorEqualsARefactorisation` (3e-14) | PINNED |
| D6 | `calibration/streaming.hpp` active set (2026-09-10) | pins are stiff rows `kPinWeight = 1e3` (gap ≈ λ/w² ~ 1e-11); `kink_tol = 1e-3·band`; multiplier hysteresis 1e-3; one release per row per tick; `max_rescales = 8 + 4·n_bands` (exhaustion = RescaleCap, a failed tick); divergence bound 1e3× the first residual or |x| > 10 | `streaming_band_test::SubBpMovesAtSmallDecay…`, `AQuoteOscillatingAroundItsEdge…` (first-order optimality in 240 directions) | PINNED |
| K5′ | `problem.hpp` Huber band | r² is convex iff the MARKET lies inside its band; under streaming the bands are absolute and the market can walk out, giving two basins (3 bp / decay 0.5: streamer at a verified kink optimum, cold LM 4 % lower elsewhere) | `streaming_band_test::ColdLeastSquaresHasOneMinimum` (inside-band only) | OPEN decision (TASKS-ENGINE E4.B K5′): bands relative to the live target, or reject/clamp a market outside its band |
| K4 | `calibration/warm.hpp` | `step_tol 1e-8`, `max_frozen 3`, `max_refresh 3` | `warm_test` | PINNED (deletion of WarmCalibrator is E6.1) |
| K14 | `compiled_book.hpp`, `hybrid_residual.hpp` | compounded (product) observations refused on the W-cache, routed to AAD; seasoned MtM coupons likewise (2026-09-10) | `compiled_mtm_test::RoutingIsExplicit`, `SeasonedCouponUsesItsFixedReset…` | PINNED |
| K15 | `compiled_bundle.hpp` | FX forward / ZeroCoupon INSIDE a Portfolio refused on the compiled path | `zero_coupon_swap_test::PortfolioOfZeroCouponRatesIsRefusedOnTheCompiledPath` | PINNED |
| D7 | `hybrid_residual.hpp` `curves_are_noncacheable`, `bundle_api.cpp has_nonlinear_` | `Scheme::MonotoneCubic` enumerated by name in two places | — | OPEN: replace both with the curve's own `is_linear_map()` (P4) |
| K18 | `regularize.hpp` | eigen-rank `1e-12·max` in the operator | `tension_regularizer_oracle_test` | PINNED |
| K22 | `vol/sabr_calibration.hpp` | LM clamps α ≥ 1e-10, |ρ| ≤ 0.9999, ν ≥ 1e-8, β ∈ [1e-6, 1]; λ ∈ [1e-12, 1e12]; butterfly grid 200, tol 1e-8 | `vol_sabr_calibration_test` | PINNED |
| K28 | `portfolio/bond_universe.hpp` | Horner path iff exponents integer-spaced to 1e-9 | `bond_oracle_test::UniverseSweepMatchesBondFunctions` | PINNED |
| K31 | `calibration/lm.hpp` | LM `xtol/ftol 1e-14`, `maxfev 4000`; non-finite seed or seed residual THROWS (2026-09-10) | `calibration_test`, `calibration_status_test` | PINNED |
| K33 | `api/compile.cpp`, `market/quote.hpp` | band default `decay = 1`; the target need not be the band mid | `market_quote_test`, `api_test` | PINNED |
| D4 (compile) | `api/compile.cpp` `compile_reg_spec` (2026-09-10) | smoothing table: tension light 0.02 / strong 0.2, second_difference 0.5 / 5.0; under-determined or banded floors "off" to light | `ApiPeriphery.CapiSessionAndCompileRewriteHonourTheSpecsSmoothing` | PINNED |

## Mathematical premises (class C)

| # | where | premise | pinned by | status |
|---|---|---|---|---|
| K6 | `calibration/risk.hpp`, `bundle_api.cpp` | risk operator by the IFT with D = diag(−∂r/∂q): decay for bands, 1/(q·T) for FX | `risk_test::AnalyticLadderMatchesBumpAndRecalibrate` 1e-4 | PINNED |
| K7 | `curve/regions.hpp` MonotoneCubic | transcribed from QuantLib's `MonotonicCubicNaturalSpline` | `monotone_cubic_oracle_test` 1e-11 | PINNED |
| K8 | `curve/regions.hpp` Tension | σ → 0 recovers the natural cubic; linear in the knot forwards | `tension_test` 1e-12 / 1e-13 | PINNED |
| K9 | `curve/regions.hpp` BSpline | convex hull; per-breakpoint 2-pt Gauss exact for the cubic's integral | `bspline_test` 1e-12 / 1e-13 | PINNED |
| K10 | `curve/regions.hpp`, `curve_module.hpp` | a leading region flat-extrapolates its first free knot | `region_combinatorial_test` 1e-10 | PINNED |
| K11 | `cashflows.hpp`, `compiled_book.hpp` | k-form reduction bit-exact; `cpn_is_plain` / `sub_is_identity` fast paths | `generic_cashflow_test` 1e-15, `compiled_generic_test` | PINNED |
| K12 | `cashflows.hpp`, `build/observations.hpp` | observation SHIFT telescopes; lookback/lockout do not (⇒ compounded) | `rfr_coupon_oracle_test` 1e-12 vs QuantLib | PINNED |
| K13 | `build/observations.hpp` | partial-fix compounded prefix as `weight = 1 + rτ_past`, `realized = f − 1` (exact identity) | `fixings_test`, `ois_weekend_test` | PINNED |
| K16 | `calibration/aad_block.hpp` | DF-cache query times are x-independent; affine map only on linear ancestry; cursor replay with search + exact fallback | `aad_block_pooled_test`, `hotpath_shapes_test` | PINNED |
| K17 | `calibration/regularize.hpp` | a Flat piece's tension energy is exactly 0 (constant-baseline subtraction) | `tension_regularizer_test::FlatStepDiscontinuitiesHaveZeroTensionEnergy` | PINNED |
| K19 | `pricing/fixings.hpp` | a fixing ON the evaluation date counts as past iff present | `fixings_test`, `api_fixings_test` | PINNED |
| K20 | `calibration/credit_problem.hpp` | piecewise-flat hazard (Hermite measured failing: h(10y) = −20 %) | `credit_test`, `curve_property_test` | PINNED |
| K25 | `vol/vol_cube_interp.hpp` | linear in total variance; flat outside the nodes | `vol_cube_interp_test` 1e-14 | PINNED |
| K27 | `pricing/bond_future.hpp` | CME 6 % conversion factor closed form; rounding owned by the caller | `bond_future_test` 1e-10, Hull 5e-5 | PINNED |
| D13 | `calibration/lm.hpp` | the anchored LM lands null components exactly at the seed (up to (w/σ)², see T1) | `tension_regularizer_oracle_test` | OPEN with T1 |
| D15 | `residual_engine.hpp` `kRankThreshold` | 1e-10 separates stiffness (~1e-3..1e-5) from numerical null (~1e-11) | `calib_report_test`, `rank_safety_test` | OPEN: record which cases were measured |
| D16 | `vol/vega_ladder.hpp`, `sabr_calibration.hpp` | {α, β} near-collinear at the money (a bare 4-way fit is ill-posed) | — | fine as stated |
| D11 | `curve/curve_module.hpp` `integral` | `integral(t ≤ 0) = 0` ⇒ DF ≡ 1 for any non-positive time; nothing may rely on a negative-time DF (a past MtM reset carries `reset_fx`, the P&L roll floors at 0) | documented at the site 2026-09-10 | PINNED by S2's tests |

## Silent fallbacks and defaults (class D)

| # | where | default | status |
|---|---|---|---|
| R5 | `build/instruments.hpp` `float_leg_from` | an EMPTY index id is assumed to be an overnight (compounded) index | OPEN (P2: a missing lookup throws) — reachable from a booked trade's JSON |
| R6 | `vol/vega_ladder.hpp` | SABR parameter sensitivities by central finite difference (h = 1e-6) although `sabr_vol_gradient` is analytic; β now carried (9998e5c) but the FD path remains | OPEN (P4: one definition) |
| D9 | `api/exposure.cpp` | `sigma 0.008`, `kappa 0.08`, `pfe_q 0.975`, `reg.lambda 0.02` | labelled illustrative; λ unlabelled | OPEN: label λ |
| D10 | `api/bundle_api.cpp` `flat_x0` | mean outright quote clamped to [1e-3, 0.20], 0.02 fallback | OPEN: state in `docs/api-surface.md` |
| T4 | `pricing/compiled.hpp`, `jacobian.hpp`, `aad_block.hpp` | an empty-gradient integral zeroes its W row "defensively" | OPEN: assert in the W-builders (only t ≤ 0 is legitimately constant) |
| K30 | `calibration/bundle_stage.hpp` | staged `stationarity = −1.0` ("not computed") | reported honestly | PINNED |
| E4.A.3 | `api/bundle_api.hpp` fixings | a fixing / evaluation-date change RECOMPILES the engine, streamer and book twin (correct; not a tick) | `api_state_test::FixingsAfterStartStreaming…` | PINNED (making a fixing a scalar row update is E4.A.3) |

## Numeric hygiene (class E)

| # | where | claim | status |
|---|---|---|---|
| D1 | `curve/regions.hpp` tension helpers | series for |x| < 0.5: `coshm2` to x¹⁴ (3e-16), `sinhm1`/`xcoshm` stop at x¹¹ (9.3e-13 / 5.5e-12 at the branch; structure entries jump 2e-10 across σh = 0.5, probe G-1) | OPEN: two more terms each |
| D2 | `curve/regions.hpp` BSpline | "interior breakpoints are UNIFORM" was stale (de Boor averaging) | comment corrected 2026-09-10 |
| D3 | `curve/regions.hpp` `BSpline::integral2` | exact per-segment f² — no production caller (the moment path uses `moment_gauss_nodes`) | dead by design; test-only |
| T9 | `compiled_book.hpp`, `compiled_multi.hpp` | a·(1/b) vs a/b: ≤ ~1.5 ulp (two roundings), gates 1e-12 / 1e-13 / 1e-10 | fine; the "1 ulp" wording is the only inaccuracy |
| K23 | `vol/bachelier.hpp`, `fx_black.hpp`, `cap_stripping.hpp` | safeguarded Newton–bisection, vega > 1e-16, bracket doubling ×64 | PINNED (`vol_bachelier_ql_oracle`, `fx_black_oracle_test`, `vol_cap_stripping_test`) |
| K24 | `vol/fx_vol_surface.hpp` | 35-sd bracket cap; Fritsch–Carlson s > 9 limiter | PINNED (`fx_black_oracle_test`) |
| K29 | `api/var.cpp` | type-7 quantile; ES over m = round((1−q)N) | PINNED (`var_test` 1e-9) |
| K32 | `curve/regions.hpp` tension x > 20 asymptotics | scaled exponentials | `tension_test::LargeSigmaApproachesLinear` (5e-3, qualitative) |

## Open decisions for the owner (from the audit; unchanged)

1. Tension/B-spline regions under the tension-energy regulariser are now exact (R1/R2) — no refusal needed.
2. K5′: bands relative to the live target vs reject/clamp a market outside its band (TASKS-ENGINE E4.B).
3. `WarmCalibrator`: shipped path or benchmark twin? Now rank-safe either way; deletion is E6.1.
4. Seasoned MtM xccy trades: priced today (S2 done at the coupon level); the trade builder for a seasoned
   xccy (fixed FX from the fixing table) is not written yet.
5. CMS replication: is 15 % vs QuantLib the accepted state, or should a pole-avoiding configuration let the
   gate drop to 1e-3 (S6)?
