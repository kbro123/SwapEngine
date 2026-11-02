# Object-model simplification review — 2026-11-01

Scope: every header under `include/swaps/` (110 files, 24.7k lines) and `api/`, read in full by five
layer reviewers after checkpoint `466140b` (the G4 gate + P15). Every consumer count below is a grep over
`include/ api/ tests/ bench/ tools/` (distinct files), and the web repo's own code (`binding/ server/
swaps/ excel/ web/`, excluding the `engine/` submodule) was grepped for the session members proposed for
removal. Nothing has been changed. This is the ledger for deciding what to do.

Headline: roughly 2,500–3,000 of the 24.7k shipped lines are either production-dead (no path from any verb
or session method reaches them) or a second spelling of something that already exists once. Three real
defects surfaced on the way (section 4). The one large design call is whether to retire the
piecewise-linear W tier by giving MonotoneCubic a closed-form tangent (section 3).

## 1. Production-dead surface — delete (low risk, ~1,400 lines)

Ordered by size. "Dead" means zero readers in `include/` + `api/` + the web repo's own code; the only
consumers are tests written for the type itself.

| # | What | Where | Lines | Evidence | Notes / risk |
|---|------|-------|------:|----------|--------------|
| 1.1 | The `market/` layer beyond `{as_of, quotes}`: `market::Scenario` (a twin of `derive/scenario.hpp`), `Market::clone/curve_modules/curve_forwards/CurveEntry`, `FxMatrix`/`FxRate`, `FixingSeries`, `market::VolCell/VolSurface` + the named vol store, `CurrencyInfo`/`currency_registry`, `Quote::to_target` (a second decay default, 0.1 vs the shipped 1.0), `source()/timestamp()` | `market/*.hpp` | ~500 + ~450 test | Production `Market` construction is two sites (`derive/asset_swap.hpp:337`, `derive/bond_rv.hpp:130`): `as_of` + `add_quote`. `api/scenario.cpp` includes only `derive/scenario.hpp`. The shipped FX model is `derive/fx_move.hpp`. Web repo: `market::Scenario` appears only in two Python docstrings. | `CSA::discount_index_id()` becomes one `require_currency` line. ARCHITECTURE.md's claim that `market::Scenario` reaches production through the scenario verb is false and must be fixed either way. Option B: retire `Market` entirely and pass `(today, clean)` to `benchmark_yield`/`load_universe`/`swap_spread`. |
| 1.2 | `BackgroundJacobian` prefetch worker + `Options::prefetch/prefetch_drift`, `StreamStage::{PrefetchTake,TrackRefresh}`, `NormalOp::adopt_M`, `StreamTick::prefetched`, `have_J_/need_full_` and their guards | `calibration/background_jacobian.hpp`, `streaming.hpp:207,271,527,546,869,897,974` | ~190 | The arming condition is `n_res == n_knots`; every session bundle carries smoothing rows, so no production bundle is square (K48). `hotpath_census.lock` already records `TrackRefresh unreachable`. Set only in 4 tests. | Census lock loses two stages and the `prefetch_ois` scenario; `tools/check_hotpath_census.py` CALLS list. Hot loop loses branches only. |
| 1.3 | The staged solve: `BundleBlockProblem`, `calibrate_staged`, `bundle_adjacency`, `bundle_dependency_order`, `solve_bundle_block`, `scatter_block`, `staged_result` | `calibration/bundle_stage.hpp` | ~190 | api = 0; 3 tests, 2 informational benches (no gated metric). CLAUDE.md §7b already records it ties the joint solve at production size. | Same treatment as E6.1 (move to `tests/` or delete). Header is in `oracle_coverage.lock`; the tier diagram names it. |
| 1.4 | The old resolution-sensitive `structure_fingerprint` + `hash_*` and `BundleSession::fingerprint_`/`structure_fingerprint()` | `calibration/structure_fingerprint.hpp:22-247`, `api/bundle_api.cpp:110` | ~120 | Getter has zero callers in api/tests/bench/tools and zero in the web repo. The live identity is `structural_stamp`. Three hand-kept field walkers today (hash, stamp, equal). | Two tests retarget to `structural_stamp_test`. Second step: one field visitor that both the stamp and `structure_equal` fold over (~80 more lines). |
| 1.5 | The fixings observer design: `Observer/Observable`, `FixingSchedule`, `resolve()`, `BoundObservation`, `FixingTable::notify`; `resolve_into` round-trips through `FixingSchedule` | `pricing/fixings.hpp:51-75,127-235` | ~110 | 0 consumers in include/api; the session calls `resolve_into` directly and re-resolves itself. | Make the in-place `resolve_into(RateObservation&, ctx)` the one definition. 3 test files. |
| 1.6 | ORE-style typed conventions: `DayCount`, `OisConvention`, `IborSwapConvention`, `BasisConvention`, `XccyConvention`, `Convention` accessors, `Index::*_convention`, `float_freq`, the three `(const Convention&)` overloads of `par_swap/basis_swap/fixed_coupons` | `build/ref_data.hpp:41-146`, `build/instruments.hpp:389-407` | ~110 | The one live idiom is `Index(id).par_convention().resolve()` ≡ `swap_conv(Index(id).currency(), id)` (7 sites). | Keep `Index` (`is_overnight`, `fixing_lag`, `currency`, `par_product`); add `Index::swap_conv()`. |
| 1.7 | `SWAPS_REDUCE_LAYOUT == 0`: `pv_reference`, the layout branches, `R_sub`/`R_cpn`/`BundleFixedLegs::R` (assembled by `setFromTriplets` on every engine build, never read on layout 1), `rmul_`, `any_general_leg_` | `pricing/compiled_book.hpp:876,912,940-1062,1179,1485,1543,1558` | ~100 | No build sets the macro. `R_sub` has one test reader. | The gated `*_engine_build` metrics can only improve. |
| 1.8 | Invoice-spread path: `ctd_forward_yield`, `derive_invoice_asset_swap`; `SwapSpreadType::Invoice/ParAssetSwap` are refused | `derive/asset_swap.hpp:142-197` | ~56 | 0 consumers repo-wide. | Check the `SwapSpreadType` string map in `api/codec.cpp` before removing enumerators. |
| 1.9 | Legacy bond spellings `us_treasury`, `us_treasury_tsy`, `us_treasury_wi`, `us_treasury_wi_tsy` (inline `freq=2` + stub flags: the literal §0 forbids) | `build/bond.hpp:161-278` | ~45 | `_wi_tsy` 0 consumers; the rest 1–2 tests + `bond_sweep_bench`. Canonical path is `bond_from_convention` / `build_bond(BondId)` / `bond_from_terms`. | 4 call sites. |
| 1.10 | Dead fields and members: `Instrument::pv_currency` (hashed, encoded, never read), `Trade::fixed_rate_schedule/principal_exchanges` + `Settlement`, `act_act_icma`, `NdfSide/ndf_sign`, `ModularCurve::zero/n_regions`, `Tension::sigma()`, `pwl_spacing → node_spacing` chain, `CompiledBundleResidual::set_quotes/market_of/curve_set`, `AadBlock::set_quotes`, `StreamingCalibrator::rescale_count/anchor_market`, `StreamTick::refreshed` (≡ `refreshes > 0`), `risk.hpp` 2-arg `residual_market_scale`, `BundleProblem::CurveSpec` alias, the `cal::OutrightHandle/TurnedCurve/BundleCurveSet/CurveSpec` re-exports (0 spellings), `BookRows::n_positions` (≡ `n_rows`) | various | ~120 | Each verified 0 consumers (or 1 in-file). | `pv_currency` is a JSON key: decide accept-and-ignore vs reject; bump `kStructuralStampVersion`. |
| 1.11 | `BundleSession` members with no reader anywhere: `fixings()`, `has_bound_portfolio()` (1 test), `rescale_count`, plus 14 orphan "Stateless X verb" comment blocks at `bundle_api.cpp:682-709` | `api/bundle_api.hpp` | ~30 | Web grep: `structure_fingerprint`, `quote_diagnostics_json`, `has_bound_portfolio` have no web reader. `evaluation_date()` and the `has_fx/has_modular/has_nonlinear/stream_*/last_*_us` getters ARE read by the web's binding and `server/stream.py`: keep. | `needs_recalibrate` is always `false` since 2026-09-12 and the web's `server/stream.py:201` still reads it into `self.modular`: dead on both sides, but dropping it is a binding change. |

## 2. One concept, two spellings — merge (medium, ~1,000 lines)

| # | What | Where | Lines | The merge | Risk |
|---|------|-------|------:|-----------|------|
| 2.1 | **`CalibrationProblem` is a second problem type no verb reaches.** `CompiledResidual` (54 lines of forwarding), `single_curve_bundle`, `has_turn_jump/has_state` SFINAE, `residual_engine<CalibrationProblem>`, the `streaming.hpp:143` default arg, the four single-curve `risk.hpp` overloads; the layout `flat_hermite(meeting, back)` is written twice. CLAUDE.md names a `to_bundle` that does not exist. | `problem.hpp:823-863`, `compiled_residual.hpp`, `compiled_bundle.hpp:54-65`, `risk.hpp:30-78` | ~220 | `tests/reference_curve.hpp build_problem/build_square_problem` return a `BundleProblem` (apply `single_curve_bundle` inside the fixture); delete the rest. 29 consumer files, nearly all through the fixture; 8 name `CompiledResidual`. | `compiled_residual.hpp` is in `oracle_coverage.lock`; `compiled_residual_test`/`streaming_test` are locked consistency tests (retarget, keep counts); `sofr_23k_*` / `risk_full_jacobian` gated benches time this path (the hybrid's `whole_` delegate is byte-identical, re-measure). |
| 2.2 | **The swaption schedule walk is written four times** (`swaption_json`, `price_vol_cube`, `VolSurface` ctor, `vega.cpp`), with three structs for one cell (`SwaptionSchedule`, `VolSurface::Cell`, `CellSchedule`); `require_index_arg`/`currency_for_index` duplicated verbatim; `price_vol_cube` re-implements `api::VolSurface` with three ad-hoc session caches (`vol_sched_cache_`, `vol_fa_cache_`, `vol_fa_x_`); the `swaption` verb is a strict subset of `vol_cube`. | `api/options.cpp:45-148,236-413`, `api/vega.cpp:41-80`, `bundle_api.hpp:139,531-533,557` | ~300 | One `build::swaption_schedule(vd, conv, expiry, tenor)`; the session holds a map of `VolSurface` keyed on the spec and `price_vol_cube` = lookup-or-build + `reprice`; `swaption` decodes to one cell and re-shapes. Move the vol block out of `bundle_api.hpp` into the existing `api/options.hpp`. | Gated `sofr_swaption_cube17_*` (VolSurface's reprice is the faster path; re-measure). `VolCube.VolSurfaceMatchesPriceVolCube` becomes tautological. Verb-density counts only go down. |
| 2.3 | **`regions.hpp`: three byte-identical per-segment cubics** (NaturalCubic/Hermite/MonotoneCubic `forward/forward_d1/forward_d2/pieces/integral/out/seg`, 33 lines ×3), tangent→coefficients twice, the lead/follow node preamble five times, the Thomas solve three times, `smin/smax` twice; `TensionHolder` duplicates `RegionHolder` for one ctor argument. | `regions.hpp:127-141,199-294,318-408,444-562,1546-1587`; `curve_module.hpp:216-249,341-346` | ~190 | A `PiecewiseCubic<S>` member (nodes, coefficients, `Is_`, eval, integral, `seg`) with `set_from_tangents` / `set_from_moments`; a free `place_nodes`; one `thomas()`; policies take `(const CurveModule&)` so one `RegionHolder` serves all eight. | Bit-identity: preserve expression order (oracle pins at 1e-11, `region_combinatorial_test` 8+64+512, `kernel_pins_test`); keep Hermite's sized-once scratch. Not hot-path. |
| 2.4 | **The band is held three ways**: `Instrument::{band_lower,band_upper,band_decay}` + `optional<PenaltyMap>` + derived `penalty()`; `RowEngine::set_quote(row, market, lower, upper, decay)`; `CompiledBundleResidual::band_lo_/up_/dc_` AND `RowMeta{penalty_map, penalty}`; `AadBlock::set_quote` copies the four scalars. The streamer keeps `quote_lo_/quote_up_/band_eligible_` (a per-kind `!= FxForward` branch) beside `BandRow`, and re-runs the K5' target check per tick that `BundleSession::stream_update` already ran. | `problem.hpp:318,378,513-524`, `row_engine.hpp:23`, `compiled_bundle.hpp:232-237,918-923`, `aad_block.hpp:109-119`, `streaming.hpp:313-320,361-365,414-416` | ~60 now | Cheap step: delete the `band_residual/_d/band_slope` shims (tests only), drop `RowMeta::penalty`, carry `lower/upper` on `BandRow` and derive eligibility from the map, validate over `bands_`. See defect 4.2 for the AAD side. | The full collapse (Instrument = `{market, PenaltyMap}`) touches the JSON keys, `CalibrationTarget`, 33 tests and the per-tick unchanged-band early-out — a later step with the web migration. |
| 2.5 | **`RowEngine` carries two spellings of every evaluation**: stored-market `residuals_into/jacobian_into` are the live-market `_vs` forms at `q = row.market`; `CompiledRows::jacobian_into` allocates `Jc` per call. **`AadBlock` re-derives the closure the hybrid just computed** and copies every `CurveStructure` per block (7 blocks on G4), instruments deep-copied twice. | `hybrid_residual.hpp:144,356,391-411`, `aad_block.hpp:148,159-164` | ~60 | Hybrid keeps the market vector it already routes; two virtuals and four impls go. Pass the closure in; hold curves by pointer; `vector<Instrument>` instead of `BundleProblem sub_`. | Bit-identical by construction; pin LM cold parity. Init-only (the `g4_*_engine_build` metrics can only improve). |
| 2.6 | **Curve-tier predicates re-derived in four places**: `curve_linear_horizon`, `curves_piecewise_linear`, `curve_tier_trackable`, `curves_are_noncacheable` (`curve_handle.hpp`), `enable_pwl` (an inline copy, says so), `value_dependent`, `tangent_curve` (`compiled_book.hpp`), `compiled.hpp:43-55`, `aad_block.hpp:369` (builds a whole Dual curve to ask `is_linear_map()`), `api/bundle_api.cpp:124`. | as listed | ~40 | `CurveStructure::linear() / has_tangent() / piecewise_linear() / linear_horizon()` per the reflect-your-own-properties rule. | Setup-only. |
| 2.7 | **API codec duplicates**: five copies of the tolerant JSON readers (`json_util.hpp`, `codec.cpp`, `compile.cpp:38-78`, `credit/inflation.cpp arrf`, `fx_option/ndf.cpp has_num`); two string→`Scheme` maps (see defect 4.1); three copies of the bond-terms decoder (`Deliverable`, `AssetSwapBond` restate `BondTerms`); six `*_json` session members that exist only for the binding, one of which re-serialises a parsed object (`price_vol_cube_json`, the documented 1-ULP hazard); 24 one-line `X_json(string)` verb wrappers. | `api/*.cpp`, `codec.hpp`, `bundle_api.hpp:261-361`, `build/bond_future.hpp:49-58`, `build/par_asset_swap.hpp:78-85` | ~150 | Fold readers into `json_util.hpp`; export the codec's scheme pair; `{…, BondTerms terms}` members with one terms decoder; descriptor `body:` forms replace the `_json` members (generated binding regenerates in the web repo). | No wire change. Web repo's generated `binding/swapengine_methods.gen.inc` names the six `_json` members. |
| 2.8 | **`regularize.hpp` keeps three generations of smoothing vocabulary** (`region_knot_lambdas(c, default_lambda)` always called with 0.0 and documented against K47; `Smoothing{Off,Light,Strong,On=Light}` with Light == Strong == 1.0; `smoothing_preset` with no include/api consumer). **`RegularizedEngine`'s one shipped consumer is the LM seed anchor** (`lm.hpp:299-306`), a solver device decided post-solve. | `regularize.hpp:46,159`, `residual_engine.hpp:76-103`, `lm.hpp:198-200,299-306` | ~70 | A per-knot `bool` and `{Off, On}` with the word table in the compile verb; fold the seed anchor into `lm_normal_equations` (`w²I` on `A`, `w²(x−x0)` on `g`, the augmented norm on `fnorm`) and move `RegularizedEngine` to `tests/` as the parity fixture it is. | Rank-deficient cold path (18 tests read `rank_deficiency`); iteration counts must match. |
| 2.9 | **The `exposure` verb carries behaviour** (lock `compute=17 branches=21`): re-decodes typed netting-set trades by hand, duplicating `book_from_json`'s trade path, and builds the OU state grid inline. | `api/exposure.cpp:128-163` | ~150 moved | `derive::exposure(ExposureRequest)` with netting sets decoded through the codec (P14). | `exposure_netting_test`; bench not gated. |
| 2.10 | **File-level splits** (0 net lines, readability): `diagnostics.hpp` hosts the P15 check, the `CalibrationSession` concept and `seed_or_flat` (forcing `consistent_risk`/`pnl_explain` to include the LM for a concept); `streaming.hpp` carries `BundleSession`'s quote hand-off (`validate_targets/commit_targets/requote_bands`) and a 330-line band walk that is a `BandWalk` member in waiting; `bundle_api.hpp` carries session state + the vol object model + result PODs. | | 0 | `calibration/session_concept.hpp`; the hand-off beside the session; the vol block into `api/options.hpp`. | `tools/check_hotpath_census.py` reads ONE file for `[callsites]` — a streaming split must teach it two. |

## 3. The design call: retire the piecewise-linear tier (~450–500 lines, measure first)

`PwlTracker` is built only for a value-dependent curve with no tangent (`compiled_book.hpp:345`); by the
inclusion lists that is MonotoneCubic alone. The machinery that exists for that one scheme: nine
`RegionIface` virtuals and their `RegionHolder` forwarders, ten `ModularCurve::pwl_*` forwarders (one
consumer each), MonotoneCubic's pattern encoder/decoder (~110 lines), `integral_weight_matrix_at`,
`PwlCurve/pwl_prepare/pwl_set/pwl_rank_update/PwlTracker/pwl_init/pwl_sync/formula_row` (~250 lines), the
three-callback `sync()` in `compiled_bundle.hpp`, and `curves_piecewise_linear` (already 0 production
consumers). Within a Hyman cell MonotoneCubic's W is `L + B·Φ_P·G`, every factor structure-only given the
recorded pattern, and the region already owns `Φ_P` (`tangent_formula`) and the linear prefilter, so
`MonotoneCubic::integral_tangent / out_value_tangent` is implementable in closed form from the last build,
the same shape as MonotoneConvex's. Then `TangentCurve` is the one value-dependent tier and
`curve_tier_trackable` becomes "every region has a tangent".

Why this is a measurement, not a refactor: the tracker re-takes W only on a cell change by a rank-k
update; the tangent tier re-takes every row at a Jacobian request when x moved and pays an exact
integral per state for the correction. MonotoneConvex measured at the linear desk's Jacobian cost, so
this is probably inside the 1.25× band, but the gated `mixed_scheme` / `desk_mixed` rungs and the
per-fingerprint soak pins decide it, and the degenerate-seed hazard the tracker documents
(`compiled_book.hpp:206-209`) must be re-pinned: the tangent at a kink is the branch the value took.
If not done: at least delete the dead `pwl_spacing` chain and drop the "EXPERIMENT" label from code that
shipped on 2026-09-22 (12+ sites).

## 4. Defects found on the way (fix regardless)

1. **The compile verb's scheme map lacks `MonotoneConvex`.** `api/compile.cpp:98-119` `scheme_from_policy` /
   `policy_name` vs the exhaustive `codec.cpp:122-147`: a composer spec cannot request MonotoneConvex while a
   bundle document can, and `policy_name` silently returns `"Hermite"` for it. Export the codec pair; delete
   the compiler's. `compile_parity_test` then needs `server/compile.py` to know the scheme too.
2. **A general penalty map on an AAD-routed row cannot be re-quoted.** `AadBlock::set_quote` copies the four
   band scalars only (`aad_block.hpp:109-119`); with a `penalty_map` installed the compiled engine stores but
   does not read the band numbers (`compiled_bundle.hpp:232-237`). Either refuse `penalty_map` on a
   `noncacheable()` row in `validate_instrument` or add `RowEngine::set_penalty`.
3. **P15 is test-only.** `front_anchors` has 0 consumers in `include/` + `api/`; a client spec with an
   unanchored spread curve still compiles into the regime the rule exists to prevent. Promote it into
   `validate_problem` / a `compile` warning.
4. **Stale map (P13).** ARCHITECTURE.md says `market::Scenario` reaches production through the scenario verb
   (it does not), describes `bind_portfolio` caching the deleted `CompiledMultiCurveBook` (also
   `api/scenario_grid.hpp:43`), lists `CurveTermStructure` under `curve/` (it lives only in `ql/`, which still
   calls itself a "CORE ENGINE COMPONENT"), calls scenario/pnl/vega/exposure "hand-dispatched" (they are rows
   of the generated `STATELESS_VERBS` table), and says trade objects are engine-only (they ship through
   `book_from_json`). `include/swaps/api/scenario.hpp:3,34` and `tests/scenario_verb_test.cpp:2` repeat the
   Scenario claim. `bundle_api.hpp:267` documents a `reg` parameter `jacobian()` no longer takes.
   `compiled_bundle.hpp:79` says `cs_.init(p.curves)` makes no copy; `CompiledCurveSet::init` does
   `specs_ = specs`. `curve_handle.hpp:20` places `Turn` in the wrong header. CLAUDE.md names `to_bundle`.
5. **Web-side dead reads.** `server/stream.py:201` reads `needs_recalibrate` (always false since
   2026-09-12) into `self.modular`. `CompileResult.{smoothness, reg_op, tension_sigma, has_bands}` are retired
   semantics still read by `server/compile.py`, `excel/dna/.../Engine.cs` and `web/engine.js` — a web-side
   migration before the engine drops them.

## 5. Checked and deliberately left alone

- `CurveHandle` family vs `CompiledCurveSet` (two access models by design, parity-tested); `OutrightHandle`
  into `SpreadHandle(base=nullptr)` adds a per-call branch and the empty-gradient hazard.
- `Bond` vs `YieldBond` (different data); `Dual/DualPooled/DualDir` (one scalar, three storages, a measured
  allocation cliff); W held three ways in the curve set (measured hot-path design); `LinearTangentTwin` vs
  `integral_weight_matrix` (unifying moves every pinned W for no capability).
- `MultiCurveBook` + `Position` (the book document, the one-shot kernel and the parity oracle) and
  `CompiledPortfolio` (the only `npv_grid` state-grid reprice, used by `exposure`); the Trade → Position →
  Npv-row chain is one conversion per hop, not a duplicate.
- The risk-side `RegSpec` (K47 names it the separate damping; live in the JSON contract); `calibrate(use_aad =
  false)` as §3(b)'s independent optimizer; the reference-behaviour toggles with one parity test each;
  `NormalOp`'s single-consumer methods (P4 as intended); `CalibrationResult`.
- `cal::CurveHandle` / `cal::build_bundle_curves` re-exports (230 spellings; a sed with no model gain);
  `WeightedInstrument`; the `BundleCurveSpec` alias (22 tests + 4 api); `CreditProblem/InflationProblem/
  CalibrationProblem` as one template (the duck-typed interface already is the abstraction).
- The `build/` date/calendar/schedule/observation single-consumer helpers (the designed, oracle-pinned
  decomposition); `par_asset_swap.hpp` vs `swap_spread.hpp` (different quantities, both have verbs); every
  `derive/` family (reached by a verb, ADD rule and FX resolver defined once); the Registry's
  optional-lookup + `require_*` pair (the P2 mechanism).
- `api_surface.py`: half its descriptor feeds a generator that lives in the web repo.

## 6. Suggested order

1. Section 4 defects (small, each its own commit).
2. Section 1 deletions in size order, each with its ARCHITECTURE.md row and lock edits in the same commit;
   the perf gate after each (engine-build metrics should only fall).
3. Section 2 merges: 2.1 (the problem type), 2.3 (the cubics), 2.2 (the vol surface), then the rest.
4. Section 3 as its own measured experiment on a worktree (the A/B recipe in memory), gated by the mixed
   rungs and the soak pins before it lands.

## 7. Outcome (2026-11-02)

Executed in the suggested order, each step on both gates (`tools/verify.sh`: correctness, the guards, and the
perf gate against the committed baselines), one commit per step unless noted:

| Step | Commit | Result |
|------|--------|--------|
| 4.1–4.4 defects + stale map | `3544aa6` | done; the P15 warning builder lives in `calibration::front_anchor_warnings` (the verb-density guard refused the loop in the verb) |
| 1.1 `market/` (+1.8 invoice path) | `3637f62` | deleted: 1,554 lines; `CalibrationTarget` / `band_target` in `problem.hpp`, the derive bridge takes `(today, clean, spread)`, `CSA` reads the DB |
| 1.2 prefetch, 1.3 staged solve, 1.4 old hash | `7f65d38` | deleted / `tests/bundle_stage.hpp` / deleted; three locks re-banked with reasons |
| 1.5, 1.6, 1.9, 1.10, 1.11 | `5656304` | deleted; kept on purpose: `pv_currency` (a JSON key the stamp hashes), `trade::Settlement` (decoded), `needs_recalibrate` (the web reads it) |
| 1.7 reduce layout 0 | `bc59211` | deleted; `g4_hermite_engine_build` 0.78x |
| 2.1 `CalibrationProblem` | `d31bd65` | `tests/single_curve.hpp` (namespace `swaps::testing`) with its `CompiledResidual` delegate and risk overloads; 37 consumers re-pointed |
| 2.3 regions cubics | `4f214ff` | `CubicSegments<S>`, one Thomas, one holder; bit-identical (every oracle and pin green) |
| 2.2 vol surface | `a4c34a8` | `build::swaption_schedule`, the vol model in `api/options.hpp`, the cube as a session-held `VolSurface` per spec shape, the `swaption` verb as cells of it; `sofr_swaption_cube17_sabr_reprice_warm` 10.6 → 5.4 µs |
| 2.6, 2.5 (half), 2.4 (cheap), 2.7 (readers) | `08076f8` | done; `g4_hermite_engine_build` 0.75x (the AAD block no longer builds a Dual curve to ask its tier) |
| 2.9 exposure | `963b6de` | `derive::exposure`, netting sets through the codec's one typed-trade decoder; verb compute 17 → 0, branches 21 → 7 |
| 2.5 second half (AAD block closure / curve copies), 2.7 (`_json` members, bond-terms decoders), 2.8, 2.10 | — | not done: init-only or binding-adjacent (the six `_json` members are named by the web's generated binding; `smoothing_preset` / `RegularizedEngine` have test consumers), and the file splits need `tools/check_hotpath_census.py` taught a second file |

**Section 3, measured and NOT taken.** MonotoneCubic was given its closed-form `integral_tangent` /
`out_value_tangent` (from the recorded Hyman pattern: node tangents as fixed rows over the secants and the raw
spline tangents, the tridiagonal re-solved per slot) and `scheme_has_tangent`; it matched AAD to 3e-16..9e-16 on
every layout (leading, following Flat / Hermite, before MonotoneConvex). Routed to the tangent tier the gated
rungs read: `shape_mixed_scheme_stream_tick` 15.6 µs vs 5.9 baseline (2.64x, above the 7.2 µs target),
`shape_mixed_scheme_refresh_25bp` 34.8 vs 12.9 (2.70x), `shape_desk_mixed_stream_tick` 40.5 vs 23.4 (1.73x),
`shape_desk_mixed_requote_tick` 1.70x, with 14 allocations per tick where the pins hold 0 (the per-state curve
rebuild and exact integral). The reason is structural, not an implementation detail: a MonotoneCubic W is
CONSTANT within a Hyman cell, so the tracker's one GEMV + pattern compare per state and rank-k update on a cell
change is the right amount of work, while the tangent tier re-takes W per state because for MonotoneConvex it
must. The experiment was reverted; the tracker stays as the piecewise-linear tier's implementation, its
"EXPERIMENT" labels dropped (it shipped on 2026-09-22) and its dead `pwl_spacing` chain deleted (1.10).
