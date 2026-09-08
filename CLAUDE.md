# CLAUDE.md — SwapsEngine

Guidance for Claude Code when working in this repository. Read this first, every session.

> **[`ARCHITECTURE.md`](ARCHITECTURE.md) is the code graph — read it to orient, and KEEP IT CURRENT.**
> Any change to the object model **updates `ARCHITECTURE.md` in the same commit**: adding, removing, or
> renaming a type; moving a file between layers; changing a layer dependency; splitting or merging a
> responsibility. The layer DAG, the per-layer type table, and the calibration-tier diagram must always
> match the tree. A stale map misleads the next session worse than no map — treat "update the graph" as
> part of "done", exactly like updating a test. Likewise keep the two oracle registries honest:
> `tests/ORACLE_TESTS.md` and the `swaps_oracle_tests` list are enforced by `tools/check_oracle_tests.sh`.

## 0. Design principles

> **The law is [`PRINCIPLES.md`](PRINCIPLES.md) (P0–P13, ratified 2026-09-08).** The bullets below are the
> original statement kept for history; where they disagree with `PRINCIPLES.md`, `PRINCIPLES.md` wins.

- **Generic building blocks, not special cases.** There is ONE curve type — `ModularCurve` = an ordered
  list of interpolation REGIONS. The schemes {Flat, Linear, NaturalCubic, Hermite, MonotoneCubic, BSpline,
  Tension} compose in ANY order and ANY position — there is **no front/back concept**. Region 0 is LEADING
  and flat-extrapolates its first knot; following regions C0-join (`Boundary::has_predecessor`). A "curve
  flavour" is DATA (a module list), not a subclass. Same spirit everywhere: prefer a generic, composable
  abstraction over a per-case branch.
- **Specifics as data, not hardcoded.** Market conventions (day counts, calendars, frequencies, spot/pay/
  fixing lags, index & product definitions) live in the conventions DB (`conventions/conventions.json`,
  codegen'd to `include/swaps/conventions_data.hpp`) and FLOW through the code — nothing inlined. A new
  product/index/instrument is a DATA entry, not a code branch. The C++ reference builders pull from the same
  JSON; zero index/currency/calendar identifiers appear in engine code (§7c). See
  `docs/generic-instrument-pipeline.md` — the canonical statement of the generic-instrument model.
- **Templated & parsimonious.** Scalar-templated throughout (`double` for pricing, `AutoDiffScalar` for
  AAD) so ONE implementation yields value AND analytic Jacobian. Minimal, composable abstractions; no
  duplication.
- **Performance is a first-class constraint, protected by gates.** The hot path — the W-cache
  (`DF = exp(-Wx)`, `W` structure-only, built once), frozen-Newton streaming, analytic Jacobian — stays
  LEAN: nothing allocated or branched per tick. Region schemes are LINEAR MAPS so the W-cache / analytic-
  Jacobian fast path is preserved; a value-dependent scheme (`MonotoneCubic`) is `is_linear_map=false` and
  routes to the AAD tier EXPLICITLY. Any generic abstraction must not bloat the differentiated hot loop.
  `tools/verify.sh` + per-fingerprint baselines enforce the perf gates (§3).

## 1. What this project is

SwapsEngine is a high-performance, QuantLib-free curve-calibration and streaming-analytics engine:
globally-calibrated, AAD-differentiated, vectorized, with its own data-driven calendars/schedules/
conventions (`build/`, `conventions/conventions.json`).

**QuantLib is NOT a dependency of the shipped engine** (`api/`, the C ABI and the web binding link only
Eigen + Boost.JSON). QuantLib is the correctness **oracle** (tests) and an informational speed
**reference** (bench, non-gating — PRINCIPLES.md P8/P9), compiled with the engine's exact toolchain (§3
perf-gate integrity). The historical statement that QuantLib "IS a linked dependency" (phases 0–3, when
`ql/extract.hpp` fed the kernels) is superseded; `ql/` is test-only.

### The division of labour (the core architectural rule)
- **Reused from QuantLib (setup, done once, not differentiable):** calendars, day counts, schedules,
  instrument/cashflow definitions. Extract dates and accrual factors from QuantLib objects here.

> **Conventions live in the DB, not in literals.** `conventions/conventions.json` is the single source
> of truth for per-product market conventions (day counts, frequencies, calendars, spot/payment/fixing
> lags), cross-checked against market sources and desk-confirmed. The C++ reference builders **pull** from
> it via the codegen'd `include/swaps/conventions_data.hpp` (regenerate with
> `python3 tools/gen_conventions_hpp.py` after any JSON edit — `verify.sh` fails if stale) mapped to
> QuantLib objects through `tests/conventions_ql.hpp`; the Python web layer pulls from the same JSON via
> `server/conventions_db.py`. New instruments add a product entry to the JSON **first**; don't hardcode
> conventions inline. `tests/conventions_test.cpp` asserts the DB agrees with QuantLib's own index
> conventions so the two can't silently drift.
- **Ours (hot path, templated on `Scalar`, differentiable, vectorized):** the curve interpolation,
  the discount/forward math, the residual vector, and the batched portfolio kernels. QuantLib is not
  templated, so AAD cannot flow through `OvernightIndexedSwap::NPV()`; the differentiable kernel must
  be our own code consuming QuantLib-extracted schedules.

### North-star capabilities
1. **Our calibrated curve is a `QuantLib::YieldTermStructure`.** `ModularCurve<Scalar>` built from
   `flat_hermite(meeting, back)` is the templated math core; the generic `CurveTermStructure<Curve>`
   wrapper exposes it to QuantLib pricing engines for validation and reuse.
2. **Global curve calibration.** All knot forwards solved **jointly** with Levenberg–Marquardt over
   residuals built from QuantLib rate helpers — not QuantLib's sequential 1-D bootstrapping.
3. **Analytic Jacobian via AAD.** The LM Jacobian `J[i][k] = d residual_i / d knot_k` is computed by
   **forward-mode "vector-dual" automatic differentiation** (Eigen `AutoDiffScalar`), in one
   differentiated evaluation — not by bump-and-reprice.
4. **Analytic bucketed risk.** Via the implicit-function theorem, `dx/dquote = -(dr/dx)^{-1} (dr/dquote)`,
   reusing the calibration Jacobian, so delta ladders are analytic (no bumping).
5. **Vectorized portfolio analytics.** Cashflow structure is extracted from QuantLib swaps once, then a
   portfolio of P swaps is priced as batched Eigen matrix–vector algebra (par rate, NPV, PV01/DV01,
   bucketed delta, convexity) — **no per-swap QuantLib pricing loop**.

## 2. Curve model (the math the code must implement)

- Free variables `x = (f_1 … f_M)` are **forward rates at knot points**. Discounting is
  `DF(t) = exp(-∫_0^t f(u) du)`.
- **Spread curves:** if a curve is defined as a spread to a base curve, the free variables are
  **forward spreads** `s_k` and `forward(t) = forward_base(t) + spread(t)`; the base curve is held
  fixed (jointly-calibrated base is a later extension).
### Knot dates
`front knots = CB meeting dates`;
`back knots = { 3M-futures END dates falling after the last meeting } ∪ { par-swap maturities }`.

- **Interpolation is two-region:**
  - **Front end (up to the last CB meeting date):** instantaneous forward is **piecewise-flat
    between meeting dates**. Forwards jump only at meetings. **No calibration instrument matures
    on a meeting date** — the front-end knots are meeting dates, full stop.
  - **Back end (beyond the last meeting date):** **smooth forwards** with **C¹ and C² continuity**.
    The back-end knots are the **3M-futures end dates** (the strip carries the curve to ~3y) and
    then the **par-swap maturities** (4y onward).
  - **At the join** (last meeting date): enforce **level continuity** of the forward into the spline;
    do **not** impose C¹/C² across the join (the front end is intentionally discontinuous).

### Calibration instruments, and why the solve is over-determined
- **1M SOFR futures** (arithmetic average) cover the first year, then **3M SOFR futures**
  (compounded, IMM) take over — the two strips are **sequential, not overlapping** (the first 3M
  contract starts on/after the last 1M end date). The 3M strip runs past ~3y.
- **Par swaps from 4y** define the long back-end knots.
- The strips and swaps span multiple knots, so front and back are coupled: one **global** solve.
  Reference market: 6 front + 17 back = 23 knots vs 29 instruments (12×1M + 8×3M + 9 swaps).
  The current-month 1M contract straddles the evaluation date, so its elapsed SOFR fixings are
  seeded when generating golden data.
- Futures convexity is **Hull–White** (`½σ² …` with mean reversion `a`), not Ho–Lee — the a→0
  limit `½σ²t₁t₂` under-damps once the strip passes ~2y. Transcribed from
  `HullWhite::convexityBias` and checked against it in `tests/convexity_test.cpp`.

**Therefore the system is OVER-DETERMINED.** `min ‖r(x)‖²` has a **non-zero residual at the
optimum**. This is a modelling choice, not a defect.

> **Testing rule: never assert that instruments reprice exactly.** Residuals are non-zero by
> construction. A test asserting `residual ≈ 0` is wrong and would only pass on a rigged
> (square) market. Assert **first-order optimality** (`‖Jᵀr‖∞ ≈ 0`) instead.

> **Residuals must all be in RATE units.** Futures quote as a *price* (`100·(1 − R)`), swaps as a
> *rate*. Mixing them in `r(x)` silently weights each futures contract **100×** a swap, because a
> 1bp rate error is `1e-4` in rate but `1e-2` in price. Convert futures to rate space
> (`R = (100 − price)/100`) before forming the residual, or apply explicit weights. Observed
> directly: at the reference forwards the futures residuals were ~2e-1 (price) while the swap
> residuals were ~1e-3 (rate) — the same order of error in rate terms.

> **QuantLib's `GlobalBootstrap` cannot be the calibration oracle here.** It ends with
> `QL_REQUIRE(finalTargetError <= accuracy)` where `finalTargetError` is the *RMS residual*; on an
> over-determined fit that throws. Loosening `accuracy` also loosens LM convergence, because
> `optEps = accuracy`. See the oracle strategy in §3.
- A cubic spline's coefficients are a **linear** map of the knot forwards, so the whole interpolator
  stays cleanly differentiable — preserve that property; do not introduce non-differentiable kinks in
  the back end.

### Interpolation is a multi-region policy, and there is exactly ONE curve type
The curve is `ModularCurve<Scalar>` (`curve/curve_module.hpp`), built by the single factory
`make_modular_curve<S>(modules)` from a runtime list of `CurveModule{knots, scheme}` — each region a
policy from `curve/regions.hpp`, stitched left-to-right by a `Boundary` handoff. **Regions are
ORDER-AGNOSTIC** (§0): any scheme leads, follows, or sits in the middle. Region 0 is LEADING
(`Boundary::has_predecessor == false`) and flat-extrapolates its first free knot (`forward(t<t1)=v1`,
symmetric with the far-end flat extrapolation) rather than pinning `f(0)=0`; every following region
C⁰-joins its predecessor (byte-identical to the old back-region path). `region_combinatorial_test.cpp`
proves it: 7 singles + 49 ordered pairs + 343 triples off one `ALL_SCHEMES[]` list, C0 at every join.
The shipped SOFR curve is just ONE such list (a flat region then a smooth one); the front/back split
below is a property of THAT layout and the knot strategy, not of the curve engine.

**A curve "flavour" is DATA, not a type.** The shipped curve is the module list `flat_hermite(meeting,
back)` (flat meeting-date front, **local C¹ Hermite** back) — used everywhere: calibration, pricing,
risk, streaming, the bundle. `flat_bspline` and `flat_monotone` are the same shape with a different back
scheme; a user-composed region list from the web composer is just another list. So there is one set of
region math, one linearity check, one W-cache and one risk path — no per-flavour curve class.
(The compile-time `MultiRegionCurve<Scalar, Regions...>` was retired: it duplicated this logic and
measured *slower* — `risk_full_jacobian` 809µs vs 644µs — because AAD gradient allocation dominates the
type erasure entirely. `TwoRegionForwardCurve` was retired earlier.) Region ctors reject duplicate/
unsorted knots and `make_modular_curve` rejects overlapping regions (a zero-length segment is a 0/0 →
silent NaN; caught at construction).
- **`BSpline` (clamped cubic, CONTROL-POINT) is an alternative back end** (`flat_bspline`; branch
  `feat/bezier-and-moment-integration`). Free vars are B-spline
  control points (P₀ pinned to the front boundary for a C⁰ join), giving **C²** and the **convex-hull**
  property (forwards can't overshoot; positivity enforceable) that Hermite's C¹ does not. de Boor eval,
  2-pt-Gauss `integral` (exact for the cubic). Fully validated: QuantLib OIS oracle 6.9e-17, calibration
  fit (identifiable), W-cache reprice 1.1e-16, and a `bspline_collocation` risk transform (control-point ↔
  forward-at-knot deltas, invertible). Control points don't lie on the curve, so risk is reported in the
  forward basis via that transform. `integral_weight_matrix(flat_bspline(...), times)` puts it on the fast path.
- **Rule: the interpolation must be a LINEAR MAP of the knot values to keep the microsecond path.**
  Only linear schemes (`Flat`, `Linear`, `NaturalCubic`, `Hermite`, `BSpline`, `Tension`) preserve
  `integral(t)=w(t)·x`, hence the `W`-cache, analytic Jacobian and warm update. `is_linear_map` (AND over
  regions) gates that tier.
- **Value-dependent schemes drop to the AAD tier — and `MonotoneCubic` is the first one BUILT and
  gate-verified** (`flat_monotone`). It is a C² natural
  cubic whose node tangents pass through **Hyman's monotonicity filter**, transcribed to match QuantLib's
  `MonotonicCubicNaturalSpline` **bit-for-bit** (`tests/monotone_cubic_oracle_test.cpp`: `forward` and
  `integral` vs QuantLib to ~1e-11 on monotone AND filter-firing data). Because the filter clamps with
  data-dependent `min/max/sign` branches, the coefficients are NOT a linear map of the knot values →
  `is_linear_map = false`, so `W` is not constant and the W-cache does not apply. Calibration still works
  through the **AAD engine** (the filter branches are piecewise-differentiable, so `AutoDiffScalar` carries
  a valid one-sided gradient); `tests/monotone_cubic_test.cpp` calibrates it and reprices to <1e-9.
- **THE GUARD (detect non-linear → route to AAD, never the W-cache):** `is_linear_map` is now enforced,
  not just documented. (a) Compile-time: `integral_weight_matrix` (`pricing/compiled.hpp`) `static_assert`s
  its curve `is_linear_map`, so a value-dependent scheme can NEVER silently reach the W-cache — it is a
  compile error. (b) Type-routing: `residual_engine_t<Problem>` maps only `CalibrationProblem`/`BundleProblem`
  (both hard-wired linear) to the compiled engine; every other problem (incl. a non-linear-curve problem)
  falls to `AadResidualEngine` by default — locked by a `static_assert` in `monotone_cubic_test.cpp`.
  (c) Runtime: `ModularCurve::is_linear_map()` returns false when it contains the `Scheme::MonotoneCubic`
  region. Any FUTURE non-linear scheme (monotone-convex / Hyman-on-Hermite) inherits this guard for free.
- Locality is a real design axis even among linear schemes: natural cubic is C² but *global* (dense
  `W`, non-local deltas); a local C¹ `Hermite`/`B-spline` back end gives local deltas **without**
  leaving the fast path. Prefer it when hedging stability matters.

### The curve is a LINEAR MAP in log-discount space (the key to caching & analytic risk)
`integral(t) = ∫_0^t f(u)du = w(t)·x` is **linear** in the knot forwards `x`, with weights `w(t)`
that depend only on the knot **times** (the structure), never their values:
- **Front (flat):** `w` entries are the overlaps of `[0,t]` with each meeting segment.
- **Back (spline):** the natural-spline coefficients are a fixed linear map of the knot values (the
  tridiagonal solve is `M = (matrix in knot times)·ys`), so their integral is linear in the values.

Hence `DF(t) = exp(-w(t)·x)`: **the only nonlinearity in the engine is the `exp` and the way DFs
combine into prices.** Exploit this — it is the whole optimization thesis:
- **Cache per instrument; reprice cheap.** For a fixed curve structure, stack each cashflow's weight
  row into a matrix `W` (cashflows × knots), built **once**. Repricing as the calibrated curve moves
  (a real-time engine) is `L = Wx; DF = exp(-L)` then vectorized per-instrument combinations — no
  spline re-solve, no schedule regeneration, no per-coupon loop.
- **Swap math directly on curve parameters.** `∂DF/∂x = -DF·w(t)`, so PV and bucketed sensitivities
  to the knot forwards are direct; chain through the calibration Jacobian (IFT) for quote deltas.
- **The calibration Jacobian is ANALYTIC (no AAD in the hot path).** `CompiledResidual::jacobian`
  computes `J = ∂r/∂x = -(∂r/∂DF · diag(DF))·W`: the per-instrument `∂r/∂DF` (analytic, sparse)
  scattered, then one `W` matmul. Matches AAD to 1e-15, ~15× faster (25 µs vs 373 µs) — this is what
  makes the streaming Jacobian *refresh* cheap. AAD has exactly TWO remaining roles (PRINCIPLES.md P3):
  producing `W` once (`integral_weight_matrix`, generic for any linear region policy), and the pooled
  `AadBlock` tier for the rows the W-cache cannot express (value-dependent schemes, MtM funding legs,
  FX-in-portfolio) — `HybridBundleResidual` composes the two and the tier of every row is fingerprinted.
- **Extending/re-wrapping QuantLib is allowed where it unlocks this.** QuantLib instruments recompute
  per-coupon on every pricing call; our wrapper computes `W` once (reusing QuantLib only to build the
  schedule) and reprices by matrix algebra. Reimplement/extend the hot parts; reuse the rest.

## 3. THE TWO GATES (non-negotiable)

Every checkpoint must pass **both** gates. `./tools/verify.sh` runs them and prints a pass/fail table.

### Correctness gate (GoogleTest, `tests/`)

Because the calibration is over-determined (§2), correctness splits into two layers with very
different tolerances. **Do not conflate them.**

**(a) Pricing / interpolation — deterministic, no optimizer involved. `rel <= 1e-10`.**
Given a *fixed, hand-specified* set of knot forwards, our `DF(t)`, `forward(t)`, futures implied
rates and par swap rates must match QuantLib. Two oracles:
  - *Front-only config* → QuantLib `InterpolatedForwardCurve<BackwardFlat>` on the meeting dates.
  - *Composite curve* → wrap our engine in a `YieldTermStructure` adapter overriding
    `discountImpl(Time)`, and let QuantLib's own pricing engines price the instruments off our
    discount factors. Any disagreement is then *our pricing bug*, not an interpolation mismatch.
    This is the workhorse oracle: it needs no QuantLib equivalent of our multi-region interpolator,
    and it is now the ONLY interpolation oracle (the natural-cubic golden CSVs were retired).

**(b) Calibration — solver-dependent. Do NOT demand 1e-10.**
  - Assert **first-order optimality**: `‖Jᵀr‖∞` below a stationarity tolerance.
  - Assert the achieved `‖r‖²` is no worse than a committed golden objective value.
  - Optionally cross-check the minimizer against an independent optimizer (Eigen LM + numerical
    Jacobian) — *not* against QuantLib's `GlobalBootstrap`, which throws on over-determined fits.

**(c) AAD Jacobian** matches bump-and-reprice within `rel <= 1e-6` (bump noise dominates).
**(d) Batched portfolio analytics** match per-swap QuantLib pricing within `rel <= 1e-10`.

- **Rule: never regress correctness. A failing correctness test blocks the checkpoint. No exceptions.**

### Performance gate (Google Benchmark, `bench/`) — policy per PRINCIPLES.md P9
- **Gated against ourselves and against absolute targets, not against QuantLib.** `tools/check_perf.py`
  FAILS a metric if (a) `ours_ns > 1.25 × committed baseline` for this fingerprint (`baselines/baselines.json`,
  min-of-N repetitions, load-checked), or (b) `ours_ns > target` in `baselines/targets.json` (desk-scale
  absolute targets; they only ratchet DOWN). There is no advisory band: exceeding either is a failure.
- QuantLib (and, nightly, QuantLib+XAD / rateslib) numbers are an **informational reference table** —
  printed, published with losses as well as wins, never gating. (History: 2026-07-10 → 2026-09-08 the gate
  hard-failed only on a QuantLib-speedup floor set 7–25× below measured and a 2× gross self-slowdown; the
  self-regression band was advisory. Superseded.)
- **Rule: every commit that touches a hot path includes or refreshes a native C++ benchmark. Perf tests
  are never routed through JSON or a web layer. If the gate fails, the change is not an improvement.**

**Curve-build is benchmarked against `GlobalBootstrap`, not `IterativeBootstrap`** — this is the
honest same-algorithm-class comparison, and it is a settled decision, do not re-litigate it:
- `GlobalBootstrap` is QuantLib's OWN global Levenberg–Marquardt. On the same square problem (same
  knots, same instruments) it uses a numerical Jacobian + per-coupon object pricing; we use an AAD
  Jacobian + templated kernel. Measured (fingerprint `52e94be82bc4`): **~6.8× faster** (2.35 ms vs
  16.1 ms), even though our spline back end does *more* interpolation work than QuantLib's flat
  forwards — so the win is purely the AAD + templated pricing.
- `IterativeBootstrap` (sequential 1-D) is a **different, cheaper algorithm** and is actually *faster*
  on the simple square case (~0.6 ms). It cannot do the over-determined global fit or supply analytic
  risk, so it is not our target. Say so openly; never quote a curve-build win over it.
- The **largest** win is the risk ladder: our AAD + implicit-function-theorem gives the full bucketed
  delta from one calibration; QuantLib bumps-and-reprices (one full re-bootstrap per quote). That is
  the `risk_full_jacobian` gate (≥20×). See `bench/curve_build_bench.cpp`.

### Numerical tolerances (single source of truth: `tests/tolerances.hpp`)
- Discount factors / par rates vs QuantLib: `rel <= 1e-10`.
- AAD Jacobian vs bump-and-reprice: `rel <= 1e-6` (bump noise dominates; tighten if we refine the bump).
- Update tolerances only with a documented reason in the commit message.

### Perf-gate integrity (how we keep the benchmark honest)
- **QuantLib must be compiled from source with the SAME compiler and the SAME optimization flags
  as our engine.** The flags have ONE source of truth — `cmake/DetectISA.cmake` (today on this machine:
  `-O3 -DNDEBUG -fno-math-errno -march=x86-64-v3`, AVX-512 deliberately off) — probed by
  `tools/archprobe` for `bootstrap_deps.sh` and `fingerprint.sh`; the QuantLib toolchain is recorded in
  `third_party/quantlib/install/TOOLCHAIN.json` and is PART of the baseline fingerprint. Never benchmark
  against a generically-compiled QuantLib package/bottle.
- Baselines in `baselines/baselines.json` are **keyed by a machine+toolchain fingerprint**
  (`./tools/fingerprint.sh` → CPU, ISA, compiler, arch flag). The perf gate **must refuse to compare
  measurements across different fingerprint keys** and instead demand a re-baseline. A Kaby Lake
  AVX2 number and an M-series NEON or AVX-512 Xeon number are not comparable, and silently
  comparing them would manufacture a fake speedup. Changing compiler *or* CPU changes the key.
- `thresholds` in that file are policy (how much we must beat QuantLib by) and are
  machine-independent; only the measured timings are per-fingerprint.
- Benchmarks run on a quiesced machine; report medians, and prefer `benchmark::DoNotOptimize` /
  `ClobberMemory` to stop the optimizer eliding the work under test.
- The gate is `tools/check_perf.py` (run by `verify.sh`): it runs the benchmarks, refuses to compare
  across fingerprints, and enforces the thresholds. It pairs `BM_*_QuantLib` with `BM_*_Ours` per
  metric — keep that naming when adding benchmarks.

## 4. Build environment (this machine) & commands

### The dev machine (current) & what's a fact vs a design constant
> Development moved to a **2019 Mac Pro** (details below); the earlier machine was a 2017 MacBook Pro
> (Kaby Lake, AVX2-only, Ventura). The infra (ISA autodetect + per-fingerprint baselines) already handles
> the switch — the notes below describe *this* box, and none of the ISA/core numbers are design constants.

- **Mac Pro 2019 (`MacPro7,1`), Intel Xeon W-3223 Cascade Lake, 8 physical cores / 16 threads.**
- **SIMD *on this host*: AVX-512 (→ 8 doubles/register), auto-enabled.** This is a *fact about this
  machine*, **not a design constant.** The ISA is detected at configure time by `cmake/DetectISA.cmake`
  (AVX-512 / AVX2 / AVX / NEON / SSE2) — it sets `SWAPS_ARCH_FLAGS` (`-march=native`), `SWAPS_ISA_NAME`,
  and `SWAPS_EIGEN_DEFINES` (`EIGEN_ENABLE_AVX512` here); `SWAPS_ENABLE_AVX512` is ON when detected. See
  the no-hard-coded-width rule in §5. The project builds optimally on any host.
- **macOS 26.5 (Darwin 25.6.0)** — upgraded from 14.7.3 Sonoma; see the toolchain note below.
- **Perf baselines are per machine-fingerprint** (`baselines/baselines.json`, keyed by cpu+isa+compiler):
  this box is now fingerprint **`86d5211c2c03`** (Xeon W-3223 / AVX-512 / **Apple clang 21**), baselined
  2026-08-29. The OS/CLT upgrade changed the compiler string, hence the key: the previous entry for this
  same machine, **`a8c9a844826e`** (clang 16, baselined 2026-07-18), and the old MacBook Pro
  (`52e94be82bc4`, AVX2 / clang 14) are retained but never compared against. `check_perf.py` refuses to
  compare across fingerprints, so a machine or toolchain switch can't manufacture a fake speedup — it is
  why the clang-21 capture had to redo every metric. (`risk_full_jacobian` reads 183x there vs 34.9x under
  `a8c9a844826e` because of the R5/R11 pooled-AAD commits landed in between, NOT the toolchain.)
- **Cap build parallelism at the physical-core count** (8 here). `bootstrap_deps.sh` / `verify.sh` cap via
  `SWAPS_BUILD_JOBS`. History (old 4-core/16 GB box): 10 concurrent clang processes exhausted memory and
  tripped an **APFS kernel panic** (`OSMetaClassBase::_RESERVEDOSMetaClassBase6`, panicking task `clang`)
  plus clang segfaults that looked like ICEs but weren't. Less acute on the Mac Pro, but keep the cap.
- **Never run benchmarks while anything else is compiling.** Perf numbers taken under load are garbage.
- **Do not use Homebrew for project dependencies** (Tier-3 on this OS → source builds, drags in
  `go`/`rust`/`llvm`). Vendor deps into `third_party/` instead.
- Toolchain: **Apple clang 21 via Command Line Tools 26.x** (`xcode-select -p` →
  `/Library/Developer/CommandLineTools`; no full Xcode installed). C++20 (`-std=c++20`) requires clang 15+.
  **libc++ lives in the SDK, not the toolchain** — a missing
  `/Library/Developer/CommandLineTools/usr/include/c++` directory is NORMAL, and plain `-std=c++20` builds
  resolve libc++ from the active SDK (`xcrun --show-sdk-path`) with no extra flags.
  ⚠️ **After an OS upgrade, DELETE AND RECONFIGURE `build/`.** CMake caches the absolute `-isysroot` path;
  the upgrade removed `MacOSX15.2.sdk` and every compile failed `'stdexcept' file not found` with a
  `no such sysroot directory` warning. `rm -rf build && cmake -S . -B build …` fixes it. The vendored
  QuantLib/gtest/benchmark `.a`s built under clang 16 link fine against clang 21 (libc++ ABI is stable) —
  no `bootstrap_deps.sh` re-run was needed. The old **macOS 14.5-SDK `CPLUS_INCLUDE_PATH` workaround for
  QuantLib's `std::format` ADL collision is GONE and no longer needed**: that SDK no longer exists on this
  box, and QuantLib 1.35 + clang 21 build clean without it.
  ⚠️ **Historical gotcha (fixed 2026-07-26):** a corrupted/partial CLT install had left an *empty-but-
  present* `usr/include/c++/v1` stub dir (just a `__cxx_version` file) that **shadowed** the SDK's copy, so
  every compile failed `'cstddef' file not found` and clang never fell back. **Fix that worked:** a clean
  reinstall — `sudo softwareupdate --install "Command Line Tools for Xcode-16.2"` — removed the stub. If
  that symptom ever returns without reinstalling, the crutch is
  `-nostdinc++ -isystem "$(xcrun --show-sdk-path)/usr/include/c++/v1"` on `CMAKE_CXX_FLAGS` (SDK copy is
  complete; codegen identical, perf gate unaffected).

### Dependencies (all vendored under `third_party/`, gitignored)
Eigen (header-only), GoogleTest, Google Benchmark, Boost headers, and QuantLib
(built from source with our flags — see the perf-gate integrity rule in §3).

```bash
# One-time: fetch + build vendored deps (resumable; the network here is unreliable)
./tools/bootstrap_deps.sh

# Configure + build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Correctness gate
ctest --test-dir build --output-on-failure

# Performance gate
cmake --build build --target bench && ./tools/verify.sh --bench-only

# Both gates
./tools/verify.sh
```

- Optimized builds only for benchmarking: `-DCMAKE_BUILD_TYPE=Release` (`-O3 -march=native`).
- Any network fetch must be **resumable and retried** (`curl -C - --retry`); connections drop here.

## 5. Coding standards

- **C++20. The shipped engine does NOT link QuantLib** (tests and reference benches do). Calendars/
  schedules/instruments come from `build/` + the conventions DB. The *hot, differentiable kernels*
  (curve, pricing, residuals, batched analytics) are **header-only and templated on the scalar type** (`double` for pricing, `AutoDiffScalar<…>` for AAD) — never
  hard-code `double` there. Extract dates/accruals from QuantLib **once** at setup; keep QuantLib
  objects and virtual dispatch **out of the hot loop** (that per-coupon loop is exactly the baseline
  we are beating).
- **Never hard-code a SIMD width.** No literal `4`, no `_mm256_*` intrinsics in engine code.
  Use `swaps::simd::packet_size<Scalar>` and `swaps::simd::padded_count<Scalar>(n)` from
  `include/swaps/simd.hpp`. The same source must compile optimally to **2 lanes (SSE2/NEON),
  4 (AVX/AVX2), or 8 (AVX-512)** with no edit. `simd.hpp` `static_assert`s that CMake's detected
  width agrees with Eigen's `packet_traits<double>::size`, so a misconfigured build fails loudly
  rather than silently running at the wrong width.
- Pad the swap dimension of portfolio matrices to `padded_count<Scalar>(P)` so batched loops need
  no scalar remainder path.
- Hot paths: **no heap allocation in inner loops** (the scope of this claim is exactly what the T4
  invariant tests prove — `tests/alloc_free_test.cpp` today covers `residuals()` only), no `virtual`
  dispatch, no UB. Prefer Eigen fixed/
  dynamic matrices with contiguous storage. Keep data layout SoA-friendly for vectorization.
- Vectorize with Eigen expressions; **avoid per-swap `for` loops** in analytics — that is the point.
- Determinism: results must be reproducible run-to-run (mind FP contraction/`-ffast-math`; do not
  enable `-ffast-math` without re-baselining correctness).
- Public headers in `include/swaps/…`; keep interfaces small and free-function where possible.

## 6. Git workflow, checkpoints & web backup

- Work on `main`. Commit at every meaningful checkpoint with a clear message.
- **Commit message convention:** `type(scope): summary`, where type ∈
  `feat|perf|fix|test|bench|refactor|build|docs|chore`. For `perf` commits, include the measured
  before/after speedup in the body.
- Commit trailer states the **actual** gate result, e.g. `Verified: correctness=PASS perf=SKIP`.
  `checkpoint.sh` derives it from the real `verify.sh` output — never hand-write a trailer
  claiming a gate passed when it skipped or was not run.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Major checkpoints** get an annotated tag `checkpoint-NN` and are **pushed to the web backup**
  (GitHub private repo `origin`) **only after both gates pass**. Use `./tools/checkpoint.sh "message"`.
- **Do not push to the web backup automatically or on every commit.** Pushing publishes outside the
  machine — only at major checkpoints, gates green, and surface what is being pushed.

## 7. Directory layout

```
cmake/DetectISA.cmake        automatic AVX-512/AVX2/NEON/SSE2 detection -> packet width
cmake/simd_config.hpp.in     template for the generated swaps/simd_config.hpp
include/swaps/simd.hpp       packet_size<T>, padded_count<T>() — the ONLY source of vector width
include/swaps/curve/         regions.hpp (Flat/Linear/NaturalCubic/Hermite/MonotoneCubic/BSpline region
                             math; ctors reject duplicate/unsorted knots), curve_module.hpp (THE curve:
                             ModularCurve + make_modular_curve from CurveModule{knots,scheme}, plus the
                             named layouts flat_hermite -- the SHIPPED one -- flat_bspline, flat_monotone),
                             ql_term_structure.hpp (generic CurveTermStructure<Curve>)
include/swaps/calibration/   problem.hpp (CalibrationProblem + the GENERIC Instrument/FloatLeg/FixedLeg/
                             QuoteKind model -- see §7c), lm.hpp, risk.hpp, warm.hpp (cached-Jacobian +
                             linear update), streaming.hpp (exact frozen-Newton live feed),
                             residual_engine.hpp (per-problem residual/Jacobian engine trait driving
                             warm/streaming),
                             compiled_residual.hpp (single-curve = 1-curve delegate to compiled_bundle.hpp),
                             compiled_bundle.hpp (CompiledBundleResidual: multi-curve W-cache residual),
                             bundle_problem.hpp + bundle_stage.hpp (Stage 3 multi-curve bundle)
include/swaps/pricing/       templated, QuantLib-free pricing kernel (cashflows.hpp: the GENERIC
                             RateObservation/FloatCoupon/FixedCoupon model (§7c) -- the index-flavoured
                             legacy structs (OisSwap/CompoundedFuture/AveragedFuture) were retired; every
                             shape is now that one generic model with different DATA);
                             compiled.hpp (integral_weight_matrix W primitive) + compiled_book.hpp
                             (CompiledCurveSet + role-aware batches incl. BundleFloatBatch, the ONE
                             float primitive: the multi-curve W-cache engine);
                             curve_spec.hpp (CurveStructure: the per-curve topology shared with the
                             calibration layer -- BundleCurveSpec is an alias of it)
include/swaps/ql/            extract.hpp: QuantLib -> plain-data extractors (the one QL-touching layer);
                             generic (dispatch on COUPON TYPE) + the legacy per-shape extractors
include/swaps/ad/            AAD scalar typedefs / dual helpers
include/swaps/portfolio/     portfolio NPV kernel (portfolio.hpp) + vectorized reprice (compiled.hpp)
src/                         non-header impl / example drivers
tests/                       GoogleTest correctness gate (QuantLib YieldTermStructure oracle -- no golden CSVs)
bench/                       Google Benchmark performance gate
baselines/baselines.json     committed baseline timings (ours vs QuantLib)
tools/                       bootstrap_deps.sh, verify.sh, checkpoint.sh, stream_sim, etc.
third_party/                 Eigen, GoogleTest, Google Benchmark, Boost headers, QuantLib
                             (gitignored; fetched + built locally by tools/bootstrap_deps.sh)
```

## 7a. Stage 2 — warm re-calibration to microseconds

Goal: re-calibrate after a *small* market perturbation from a solved curve in microseconds. Three
measured levers (the order matters — measure before optimising, CLAUDE.md discipline):

- **Cached-Jacobian Newton (`swaps/calibration/warm.hpp`, `WarmCalibrator`).** A warm re-cal doesn't
  need a full LM: cache `J0` (and its factorization) at the base solution and take frozen-Jacobian
  Gauss-Newton steps. **Automatic envelope detection** — if frozen steps stop converging, `J0` is
  stale (perturbation outside the reuse envelope) and it auto-refreshes. Full LM 2.8 ms → **48 µs**
  exact (matches full LM to ~1e-13).
- **First-order live-tick update = ONE matvec.** At the base `r(x0) = -dq` exactly, so the first
  Newton step is `x = x0 + M·dq` with `M = (JᵀJ)⁻¹Jᵀ` precomputed — which is the *same* operator as
  the analytic risk ladder `dx/dq`. **168 ns**, error `O(|dq|²)` (~1e-6 at 1bp). This is the
  microsecond path for real-time ticks; `recalibrate()` (48 µs, exact) is the fallback for big moves.
- **Vectorized compiled residual (`compiled_residual.hpp` on `pricing/compiled.hpp`).** The
  calibration instruments are just a portfolio valued off `DF = exp(-Wx)`; `CompiledResidual` shares
  that engine. **Honest result: only ~1.3× vs the scalar kernel** — our scalar residual was already
  fast (no virtual dispatch, telescoped, L1-resident), so the `W`-cache that gave 126× vs QuantLib's
  slow per-swap loop gives little here. It's the DRY unification, not the speed lever.

> **Measured, not assumed:** the warm-recal working set is ~9 KB (L1-resident), so cache-placement /
> memory-hierarchy tuning buys ~0. The wins are algorithmic (cached `J0`, the one-matvec update),
> not memory management.

### The accuracy-first streaming path (`swaps/calibration/streaming.hpp`, `StreamingCalibrator`)
For a **live pricer to sharp clients**, a stale price is a free option — so the default streaming mode
is `exact`: **iterate the cached `M = J⁻¹` as a frozen-Newton *preconditioner* to convergence every
tick** (`x ← x − M·(model_rates(x) − q)` until `‖dx‖∞ < 1e-9`), not a single linear extrapolation.
- For a **square, full-rank** problem the fixed point is `r = 0` **for any invertible `M`**, so every tick
  lands on the EXACT solution (round-trip `‖model_rates(x) − q‖∞ ≈ 1e-12`, machine-exact) regardless of
  `M`'s accuracy; `M` only sets the convergence *rate* `ρ = ‖I − M·J(x)‖` (measured: ~0.04 at 10 bp of
  drift → ~3 steps/tick). For an **over-determined / banded / regularised** problem the fixed point is
  `J_refᵀ r = 0`, which equals the least-squares condition `J(x)ᵀ r = 0` only while `J(x) == J_ref`: the
  answer is exact to second order in the drift (≤0.015 bp at 30 bp for plain par rows), and the streamer
  keeps it that way with (a) a drift-triggered refresh (`Options::refresh_drift`, 10 bp; square problems
  are exempt) and (b) band-edge tracking — a banded row's slope is exactly `decay` inside / `1` outside
  (the Huber band, `problem.hpp band_residual`), so a crossing re-scales that frozen row and re-factorises
  `M` (tens of µs, `StreamTick::rescales`) instead of leaving `J` a 10x moving target. A tick that hits its
  refresh cap reports `converged = false` and is NOT committed. Pinned in `tests/streaming_band_test.cpp`.
- **The Jacobian is recomputed only on genuine staleness** — when frozen-Newton needs more than
  `max_frozen` steps — NOT on a drift envelope. That staleness envelope is ~30 bp of curve move (a
  level move; `tools/jacobian_staleness.cpp` measures `ρ` vs move size) vs the linear path's 0.35 bp,
  so ~85× fewer recomputes AND exact. The recompute reuses the cached `W` (analytic `J`, no AAD).
- **The legacy `linear` single-step path (`x = x_anchor + M·dq`, O(dq²) error, re-anchor at 0.35 bp)
  is a *smooth-market artifact*.** Under a realistic **~0.15 bp/tick** feed it re-anchors ~20% of
  ticks (drift crosses 0.35 bp every few ticks); the exact path rides one Jacobian all day. The
  "168 ns / 98% fast-path" numbers above assume an unrealistically smooth crawl — keep them for the
  *WarmCalibrator small-perturbation* use, not the live tick feed. Gate: `tests/streaming_test.cpp`
  (round-trip + exactness every tick). Demo: `tools/stream_sim.cpp` (trending day, both modes).

### The async pricer/calibrator split — the "pricing branch" (`calibration/live_curve.hpp`)
A live desk runs the CALIBRATOR on one thread and PRICES on others; the pricers must never block the
calibrator and never read a half-updated curve. `LiveCurveFeed` is the lock-free hand-off: ONE writer
(the `StreamingCalibrator` thread) `publish(x)`es the newest knot vector each tick; any number of
readers (pricer threads) take a consistent `snapshot(out)` LOCK-FREE and price off it.
- **Mechanism = atomic pointer swap over a preallocated buffer ring + a per-slot seqlock.** `publish`
  fills the NEXT ring slot (round-robin, so a slot a reader just grabbed is not overwritten for ring−1
  more publishes), brackets the payload write with an odd/even generation counter, then **release-stores**
  the slot index into `published_`. `snapshot` **acquire-loads** `published_`, copies that slot, then
  re-checks the slot generation AND that `published_` still points there — retrying only if a publish
  lapped it mid-copy (rare: a publish is per-tick, a copy is ~µs). No locks, **no allocation after the
  ctor** (the writer never blocks; the reader is bounded-retry).
- **Guarantees:** release/acquire on `version_` gives a reader a happens-before view of the whole payload,
  so it always prices off SOME wholly-published `x`, never a blend of two (§5 determinism across threads).
- Gates (`tests/live_curve_test.cpp`, `tests/streaming_test.cpp` `AsyncPricer…`): 1 writer + 4 readers
  over 300k publishes → **zero torn reads, zero version regressions**; and a real
  `StreamingCalibrator` thread + a `CompiledResidual` pricer thread (its OWN scratch — the compiled
  engine's per-instance memo buffers are NOT shared) → every snapshot equals EXACTLY the curve published
  at that version. NB each pricer needs its own `CompiledResidual` (mutable DF/scratch memo); the
  `LiveCurveFeed` is the only shared state and it is all atomics.
- **Coherent data-parallel reprice (`portfolio/parallel.hpp`, `ParallelPortfolio`).** The OTHER pricing
  mode: split ONE large book across N threads that all price the SAME pinned curve, for a consistent
  point-in-time cut (vs the feed's latest-wins, per-reader independence). It partitions the book into N
  `CompiledPortfolio` slices (each its OWN scratch), and a `reprice(x)` fans them across threads — every
  slice reads the same `x`, writes a DISJOINT output segment. Because a position's NPV depends only on `x`
  and its own cashflows (never on the rest of the book), the sliced result is **bit-identical to a
  single-thread full-book reprice** (`tests/parallel_portfolio_test.cpp`, `== 0.0`), so the cut is
  deterministic AND coherent. The two compose: `feed.snapshot(x)` ONCE (pin a version) → `book.reprice(x)`
  fans that one curve across workers. Measured: **~4× wall-clock** on a 20k-swap book, 8 slices
  (`bench/portfolio_parallel_bench.cpp`, 2.25ms→0.57ms).
- **Persistent thread pool (`parallel/thread_pool.hpp`, `ThreadPool`).** `std::async` spawns a thread per
  task; a pool created ONCE and reused amortizes that. `parallel_for(count, fn)` fans fn(i) across workers
  and blocks. Both `calibrate_staged_parallel(..., pool)` and `ParallelPortfolio(..., pool)` take an
  optional `ThreadPool*` (nullptr → the `std::async` fallback). Determinism is untouched (the pool changes
  WHEN a task runs, never WHAT — gated: pool path == serial to 0.0). **Measured, and honest about where it
  matters:** the portfolio reprice — a FREQUENT fan-out over FAST slices — goes **3.5×→4.3×** (spawn cost
  per reprice removed; main-thread CPU 167µs→25µs). The bundle COLD staged solve is a **wash** (24.5ms
  async ≈ 25.1ms pool): each block solve is ~ms-heavy, so a one-time thread spawn is negligible. So the
  pool is for the repeated real-time fan-outs (per-tick reprice), not one-shot heavy calibration.
- **DONE — speculative background Jacobian (`calibration/background_jacobian.hpp`, tail-latency win).**
  The streaming refresh (recompute J + factorize M) was SYNCHRONOUS: the tick that hits the staleness
  envelope paid the whole cost. Measured (`bench/jacobian_cost_bench.cpp`): a fast tick is **8 µs**, a
  refresh tick is **47 µs** (single-curve analytic), **~530 µs** (single-curve AAD / non-linear curve),
  or **20 ms** (8-curve bundle AAD) — a live bundle pricer FROZE for milliseconds on a refresh tick.
  Now `StreamingCalibrator` (exact mode, `Options::prefetch`) runs a dedicated `BackgroundJacobian<Problem>`
  worker: as drift passes `prefetch_drift` it `request`s an M at the current x; the worker computes J+M off
  the critical path (its OWN engine — compiled engines have per-instance scratch, not shareable); when a
  refresh fires the tick `try_take`s the ready M and swaps it in (~µs) instead of computing inline.
  **Correctness is FREE (square problems):** frozen-Newton's fixed point is `r=0` for ANY invertible M, so a slightly-stale
  background M is exact — it only sets the convergence rate; a too-stale one merely triggers another
  refresh (bounded by `max_refresh`). No accuracy is traded. Gates: `tests/streaming_test.cpp`
  `PrefetchIsExactMatchesSyncAndFires` (round-trip 2e-12, matches the sync path to 2e-11, 263/384 refreshes
  served off-thread on a stress feed); `tests/bundle_test.cpp` `StreamingPrefetchHidesTheRefreshSpike` on
  the realistic 4-curve bundle at a live cadence → **2/2 refreshes prefetch-served, worst tick 949 µs → 110 µs**
  (~8.6×; an 8-curve/AAD case hides a bigger ms spike). It is a TAIL-LATENCY win (p100), NOT throughput —
  refreshes are rare and the compute lands on a spare core; value scales with refresh cost (skip it for the
  cheap 47 µs analytic single-curve; it is transformative for the 0.5–20 ms AAD/bundle refreshes). Hit rate
  scales with feed cadence vs refresh cost: ~100% on a realistic desk cadence, lower on an aggressive feed
  that outruns the worker (which then just falls back to the inline compute — never wrong, only slower).

## 7b. Stage 3 — the curve bundle (N curves calibrated together)

`BundleProblem` (`calibration/bundle_problem.hpp`) calibrates **N curves simultaneously** over one
stacked `x = [x_0; x_1; …]`. Instruments reference curves BY ROLE — a `forecast`, a `benchmark`
(for basis quotes) and a `discount` curve — so a discount curve ≠ its forecast curve. The multi-curve
pricing kernel (`pricing/cashflows.hpp`) is a faithful extension of the single-curve one: the OIS
schedule already separates `float_acc_start/end` (forecast) from `float_pay` (discount), so
`ois_par_rate(sched, fc, dc)` and `basis_par_spread(sched, fwd, bench, disc)` just thread two curves.
`fc == dc` reduces to the single-curve form. This is exactly QuantLib's multi-curve setup and is
validated against it to ~1e-16 (`tests/bundle_test.cpp`, a realistic SOFR + FF + PRIME + PRIME2 chain,
all SOFR-discounted, 73 knots / 85 instruments; joint & staged recover to ~1e-13).

- **It reuses everything.** `BundleProblem` exposes the SAME `residuals<Scalar>(x)` / `n_knots` /
  `n_residuals` (+ `market()`) interface, so `calibrate`, `aad_jacobian`, the risk ladder, AND the
  warm/streaming re-calibrators drive it UNCHANGED. The block-structured Jacobian falls straight out of
  AAD over the stacked residual — a curve-0-only instrument has an identically-zero derivative w.r.t.
  curve-1's knots.
- **One compiled `W`-cache engine for the whole bundle (`pricing/compiled_book.hpp`).** Bundle
  log-discounts stay LINEAR in the stacked `x` (a spread curve's integral just adds to its base's), so
  concatenating every curve's DF into one global vector gives `DF_all = exp(-W_all x)` with a
  block-structured `W_all` (each row = a curve's own knot weights + its spread-base ancestry). The
  single-curve gather/reduce and analytic Jacobian then generalize UNCHANGED — the multi-curve-ness lives
  entirely in the global indices and `W_all`'s blocks. `CompiledCurveSet` + role-aware
  `BundleFloat/Fixed/Comp/Avg` batches are the shared primitives; `calibration/compiled_bundle.hpp`
  `CompiledBundleResidual` is the residual view (`J = -(dr/dDF diag(DF)) W_all`, analytic). It matches
  the templated residual to ~5e-15 and AAD to ~2e-15 (outright) / ~3e-16 (spread).
- **This is the ONE `CompiledBook`.** The single-curve `CompiledResidual` is now a thin delegate to a
  1-curve `CompiledBundleResidual`, and `portfolio/compiled.hpp` `CompiledPortfolio` reprices NPVs off the
  SAME `CompiledCurveSet` + `BundleFloat/Fixed` primitives — one compiled kernel for calibration AND
  analytics, not three. Perf-neutral (measured): `portfolio_analytics` and `warm_recalibration` gates
  unchanged. NB the batches materialize the per-coupon vector BEFORE the sparse reduction `R * v` — handing
  Eigen's sparse×dense an unevaluated gather+divide expression re-does that work per access (~1.28× slower;
  see `BundleFloatLegs::pv`).
- **Warm & streaming re-cal are problem-generic (`calibration/residual_engine.hpp`).** `WarmCalibrator`
  and `StreamingCalibrator` are templated on the problem via `residual_engine_t<Problem>`:
  `CalibrationProblem` → `CompiledResidual`, `BundleProblem` → `CompiledBundleResidual`, anything else →
  the generic AAD engine. Same frozen-Jacobian control flow, now on the analytic fast path for the bundle.
  Measured on the 4-curve bundle at a ~1bp tick (`bench/bundle_build_bench.cpp`, `tests/bundle_test.cpp`):
  warm exact re-cal **~272 µs** (~71× vs a 19 ms cold LM re-solve; matches an independent cold solve to
  ~1.5e-11), one-matvec linear update **~390 ns**, streaming exact path round-trips every tick to ~3e-12.
  The bundle's frozen envelope is *tighter* than the single curve's (the coupled multi-curve residual is
  more nonlinear), so a 1bp move triggers one Jacobian refresh — but the refresh is now ANALYTIC (no AAD
  sweep), which is what took warm re-cal from ~7× (AAD engine) to ~71×.
- **Staged solve (`bundle_stage.hpp`).** Decompose the dependency graph (Tarjan SCC), solve each SCC
  in dependency order: a **singleton** SCC → a LOCAL LM over just that curve's block with earlier
  curves FROZEN as constants; a **cycle** → a joint LM over just its members. Frozen curves must carry
  a block-SIZED zero gradient (`xb[0]*0.0`), not an empty one, or Eigen `AutoDiffScalar` adds
  mismatched-size derivative vectors → NaN. Honest result: staging beats the joint solve on small
  bundles but ~ties it at production size (curve-rebuild-per-eval dominates); the real win is that both
  are **~20× faster than QuantLib IterativeBootstrap, ~64× vs GlobalBootstrap** (`bench/bundle_build_bench.cpp`).
- **DONE — branch detection + parallel SCC solve** (`calibrate_staged_parallel`, `bundle_stage.hpp`).
  The staged solver condenses the dependency graph into SCCs in dependency order; `bundle_waves` now
  groups those SCCs into topological WAVES (an SCC whose dependencies are all solved joins the current
  wave), and each wave's mutually-independent SCC blocks are solved CONCURRENTLY (`std::async`
  thread-per-SCC off a frozen snapshot). It is **DETERMINISTIC and BIT-IDENTICAL to the serial
  `calibrate_staged`** — same-wave SCCs never reference each other's curves, so thread order cannot change
  any block's inputs and blocks write disjoint x-segments (no shared FP reduction, §5 holds). On a
  triangular chain every wave is one SCC (== serial); on a **star/forest** (K curves each spread straight
  off the base) wave 1 is K concurrent solves. Measured on the **canonical realistic star**
  (`tests/reference_bundle.hpp`: full-structure SOFR base + K FF-style basis curves off it, each 6 meetings
  + 12×1M futures + basis swaps): **~3.8× wall-clock** on a K=8 star (`bench/bundle_parallel_bench.cpp`,
  84ms→22ms, quiesced), bounded by Amdahl (the serial base-curve wave + thread-spawn) — gated by
  `tests/bundle_test.cpp` `BundleParallel.*` (parallel == serial to 0.0, recovers x_true, on BOTH a
  synthetic and the realistic star). Threads over independent SCCs, NOT processes (a solve is 1–22 ms, far
  below IPC overhead). (Across-problem parallelism — a scenario grid / multi-currency book — remains
  trivially available given the allocation-light engine; needs no new code.) **`reference_bundle.hpp`
  (`build_realistic_bundle`, chain OR star) is the canonical realistic multi-curve model for benchmarks.**
- **Outright vs spread is in the curve DEFINITION, not the solver.** `CurveSpec` carries `base`:
  `base < 0` = OUTRIGHT (free vars are its own forwards); `base >= 0` = SPREAD, i.e. the curve IS
  `curves[base] + spread` and its free vars are the forward SPREADs. `build_bundle_curves<Scalar>`
  wraps each as a type-erased `CurveHandle<Scalar>` (`OutrightHandle` / `SpreadHandle`, the latter
  `forward = base + spread`, `DF = base_DF·exp(-∫spread)`), so the pricing kernel and every residual
  are oblivious to the parameterization — the engine does the right thing off the spec alone. The
  virtual dispatch lands ONLY in the calibration residual; the microsecond streaming path stays on `W`.
  If the `base` is another FREE curve, base and spread calibrate JOINTLY (AAD couples the blocks through
  the base discount); if it is a curve you never pin, it is effectively a fixed base — that choice IS
  "jointly-calibrated vs fixed spread base". A spread curve adds a `base` dependency edge, so staging
  builds the base first (singleton SCCs) or forces a cycle into one joint block automatically. Gate:
  `tests/bundle_test.cpp` `BundleSpread.*` (joint & staged recover base+spread to ~1.5e-15; the handle
  math is exact). Requirement: `base < c` (bases defined before the spreads that reference them).
- **TEST/BENCH CONVENTION (desk-realistic): only the base curve is OUTRIGHT; every OTHER curve is a
  SPREAD over the one below it.** In `tests/bundle_test.cpp` `BundleRealistic` and `bench/bundle_*_bench.cpp`
  SOFR is outright and FF = SOFR+spread, PRIME = FF+spread, PRIME2 = PRIME+spread (a spread chain) —
  this is how a trading desk quotes basis curves, and it exercises the spread-calibration path in the
  realistic bundle, not just the hand-built `BundleSpread` machinery tests. `x_true`/`x0` for a spread
  block are SPREAD levels (~bp), not forward levels. The QuantLib oracle wraps the engine's spread-aware
  `CurveHandle` (`CurveTermStructure<CurveHandle<double>>`) so QuantLib prices the ACTUAL forward curve
  (base+spread), not the raw spread. New realistic multi-curve tests should follow this convention.

## 7c. Stage 4 — the generic instrument pipeline (design: `docs/generic-instrument-pipeline.md`)

**Status: the generic pipeline is BUILT, GATE-VERIFIED, and ADOPTED. The legacy shapes are DELETED.**
Every call site — the reference market (`tests/reference_curve.hpp` `build_problem`/`build_square_problem`),
ALL benchmarks (`baselines/baselines.json`), the `Portfolio`, the single-curve `CalibrationProblem` and
the multi-curve `BundleProblem` — constructs generic `Instrument`s only. `OisSwap`/`CompoundedFuture`/
`AveragedFuture`, their pricing functions (`ois_par_rate`/`compounded_future_rate`/`averaged_future_rate`/
`basis_par_spread`/`ois_swap_npv`), their extractors (`extract_ois_swap`/`extract_compounded_future`/
`extract_averaged_future`) and the compiled `BundleFloatBatch`/`BundleFixedLegs` legacy adapters are GONE.
There is ONE cashflow model (`RateObservation`/`FloatCoupon`/`FixedCoupon`) and ONE residual/W-cache path.

> The byte-safety of the swap migration rests on a DATE fact pinned by `tests/migration_guard_test.cpp`:
> on the reference market every overnight coupon has `valueDates().front()/back() ==
> accrualStartDate()/accrualEndDate()` (no realized prefix), which is why the generic swap reprices the
> old `OisSwap` form bit-for-bit. Futures map exactly (3M = one telescoped sub-period; 1M = per-business-day
> observation via `rb::avg_future_obs`). The index/market knowledge (the SOFR calendar walk) lives in the
> test builder, never in engine code — the engine names no index (CLAUDE.md §1).

### What the generic model IS (all of this is real and tested)
- **ONE rate formula** (`pricing/cashflows.hpp`):
  `rate = ( Σ_k w_k·[DF_fc(s_k)/DF_fc(e_k) − 1] + realized ) / tau_index`. `RateObservation` +
  `FloatCoupon{obs, pay, tau_pay, spread}` + `FixedCoupon{pay, tau}`. Compounded OIS, averaged
  OIS/FF, IBOR fixings, already-/partially-fixed coupons and all three future flavours are this ONE
  type with different DATA. `w` empty ⇒ all-ones. It reduces to the legacy OIS form exactly at
  `tau_pay == tau_index`, `spread = 0`, one sub-period, `realized = 0`.
- **Roles live on LEGS, not instruments** (`calibration/problem.hpp`): `FloatLeg{coupons, forecast,
  discount}` / `FixedLeg{coupons, discount}`. That is what makes tenor-basis representable without a
  new type. `Instrument` = legs + `QuoteKind{ParRate, ParSpread, Rate}` + market quote;
  `instrument_model_quote<Scalar>(ins, C)` / `instrument_residual<Scalar>(ins, C)` are role-accessor-
  templated, so ONE residual definition serves `CalibrationProblem`, `BundleProblem` and
  `BundleBlockProblem`.
- **Residual order (THE CONTRACT — the Jacobian rows, W-cache batches, `market()`, the risk ladder
  and warm/streaming all depend on it):**
  `1. avg_futs | 2. comp_futs | 3. swaps | 4. bases | 5. instruments (insertion order)`.
  The generic block is LAST precisely so no existing row is renumbered.
- **The generic instruments ride the analytic W-cache** (`compiled_bundle.hpp`), not a slow path:
  `ParRate` and `ParSpread` share ONE pair of float batches (ParRate's subtracted leg is empty ⇒ `pv`
  is exactly 0.0); `q_rows_`/`r_rows_` map batch position → residual row, so a mixed quote-kind list
  keeps insertion order without grouping by kind. Compiled vs templated kernel: **3.6e-17**; analytic
  block Jacobian vs AAD: **6.4e-16** (design bar: 1e-9).
- **`BundleFloatBatch` is the ONE float primitive** — legs, compounded futures and averaged futures
  are all it. **PERF (measured, do not regress):** it has two fused fast paths detected at
  `finalize()` — `sub_is_identity` (`R_sub == I`) and `cpn_is_plain` (`konst == 0 && k == 1`).
  Without `cpn_is_plain` the portfolio book costs **1.28×** (192 µs vs 150 µs). Any new coupon shape
  must keep the standard shape fused. Always materialize a `VectorXd` BEFORE a sparse reduction `R*v`.
- **Extractors dispatch on QuantLib COUPON TYPE, never index identity** (`ql/extract.hpp`).
  `extract_float_coupon` is the ONE type switch (`OvernightIndexedCoupon` / `IborCoupon`).
  Times use the CURVE day counter; accruals (`tau_pay`, `tau_index`) use the instrument/index's own —
  that separation is what makes 30/360-fixed vs ACT/360-float correct. Keep it.

### Non-obvious facts worth not rediscovering
- **Gearing folds into the weights**, no new field: `g·(Σ w_k(…) + realized)/τ ≡ (Σ (g·w_k)(…) +
  g·realized)/τ`. At `g == 1` the weight vector is left EMPTY so the standard shape stays on the
  fused fast paths.
- **A partially-fixed compounded overnight coupon folds in too.** QuantLib's fixed part is
  *multiplicative* (`P·X − 1`), which looks incompatible with the additive `realized` — but
  `P·X − 1 ≡ P·(X−1) + (P−1)`, i.e. `weight = P, realized = P − 1`. With no past fixings `P == 1.0`
  identically ⇒ weight empty, `realized == 0` ⇒ the legacy shape bit for bit.
- **IBOR sub-period is `fixingValueDate()`/`fixingEndDate()`** — NOT `fixingMaturityDate()`, and the
  design's `fixingPeriodStart/End` names do not exist in QL 1.34. Under QL's default *par-coupon
  approximation* the estimation period ends at the next fixing, not at index maturity. These are
  computed by the coupon's `IborCouponPricer`, so **the coupon must have a pricer** (`IborLeg`/
  `VanillaSwap` set one). An IBOR *future* is different: its dates come from the INDEX
  (`valueDate`/`maturityDate`), because it settles on the actual fixing — par-coupon does not apply.
- **`RateAveraging::Simple` + `telescopicValueDates = true` selects QL's Takada log-approximation**, a
  different model that will NOT agree. Build averaged coupons non-telescopic.
- Averaged-OIS `tau_index` uses `accrualPeriod()` (the COUPON's day count) to match QL's pricer, not
  the index day count the design names; identical whenever the coupon uses the index's day count.
- In-arrears IBOR is rejected with `QL_REQUIRE` (its timing adjustment is a model ⇒ tests).
- `Rate` is spelled `rate + (convexity − market)`, NOT `(rate + convexity) − market` — algebraically
  identical, but the latter is not BIT-identical and would drift every existing futures residual by
  an ulp. `tests/generic_instrument_test.cpp` asserts `EXPECT_EQ(d, 0.0)` on this.

### Index-specific knowledge in `include/` — audited, and the honest residue
Zero index/currency/calendar identifiers appear in engine **code**. All 14 name-hits are comments,
and most are ANTI-leak documentation (`problem.hpp:29`, `compiled_book.hpp:126`, `cashflows.hpp:168`,
`extract.hpp:107` exist to say "'SOFR 3M future' is not an engine concept, it is DATA") — do not
delete those, they enforce the rule. Two honest residues:
- `cashflows.hpp:107–126` section headers DO assert index identity — but only on the **legacy**
  structs the design retires. They go when those go.
- **`meeting_times` / `back_times` is FOMC vocabulary threaded through nearly every header**
  (`problem.hpp`, `bundle_problem.hpp`, `compiled_bundle.hpp`, `risk.hpp`, `curve/`, `portfolio/`).
  It is *vocabulary*, not strategy — the dates are DATA from `tests/reference_market.hpp` and the
  engine only takes knot times — so it does not breach the rule, but the name is a CB artifact.
  Renaming to `front`/`back` touches ~every header for cosmetic gain; deliberately not done.

## 8. Phased roadmap (update the checkbox as phases land)

- [x] **Phase 0** — Toolchain, repo, CLAUDE.md, CMake skeleton, QuantLib baseline builds.
      *(Apple clang 14.0.3 / C++20; vendored cmake+ninja+Eigen+Boost+GTest+Benchmark;
      QuantLib 1.34 built static with `-O3 -march=native`; ISA auto-detect → AVX2+FMA,
      4 doubles/reg; correctness gate green. Perf checker still a stub → Phase 6.)*
- [x] **Phase 1** — Correctness harness + golden reference from QuantLib.
      *(Reference market: 6 FOMC front knots + 17 back knots (8 from 3M-futures end dates + 9
      swap maturities) vs 29 instruments (12×1M then 8×3M sequential futures + 9 swaps 4y–30y),
      over-determined.
      The two-region forward curve (now `ModularCurve` via `flat_hermite`, see §2) validated against
      QuantLib's BackwardFlat and natural-cubic interpolators to ~1e-16, with a 1bp negative control.
      Hull–White futures convexity matches `HullWhite::convexityBias` exactly. `ql_term_structure.hpp`
      exposes our curve to QuantLib as a
      `YieldTermStructure`, so QuantLib prices instruments off our discount factors.)*
- [x] **Phase 2** — Core engine: templated pricing kernel + residual vector (RATE units) + global LM.
      *(Kernel prices OIS swaps, 1M averaged & 3M compounded SOFR futures from QuantLib-extracted
      schedules, matching QuantLib to ~1e-16 incl. the current-month contract's realized fixings.
      Global LM (numerical Jacobian) recovers a generating curve to 3e-14 and reaches
      ‖Jᵀr‖∞≈1e-8 on the over-determined market. `include/swaps/{pricing,ql,calibration}/`.)*
- [x] **Phase 3** — AAD Jacobian (forward-mode vector-dual); verify vs bump; verify speed.
      *(`swaps/ad/dual.hpp` seeds M knot forwards as `AutoDiffScalar<VectorXd>`; one differentiated
      pass of the residual code yields the full M-wide Jacobian. Matches bump-and-reprice to <1e-6 on
      significant entries; AAD-driven LM reaches ‖Jᵀr‖∞≈8e-15. Cross-checked against the linear-map
      weights w(t) (exact to 1e-15). Speed (dev box, not quiesced): AAD Jacobian 2.7× vs bump; full
      calibration 4.8× vs numerical (AAD LM 4 iters vs 16). Kernel accumulators are AAD-safe
      (seeded from the first curve-dependent term; raw-double constants). Jacobian is moderately
      ill-conditioned (cond≈636): one knot direction is weakly identified — a smoothness/Tikhonov
      regulariser is the eventual fix. AAD's VectorXd allocation is the next speed target.)*
- [x] **Phase 4** — Spread curves (forward-spread interpolation to a base curve).
      *(`forward = base + spread`, `DF = base_DF·exp(-∫spread)`, spread interpolated with the shipped
      flat_hermite layout so it stays linear/AAD-differentiable; base fixed. This FIXED-base spread path
      is TEST-ONLY — production spreads calibrate JOINTLY via the bundle (SpreadHandle) — so `SpreadCurve`
      and `SpreadCalibrationProblem` live in `tests/spread_reference.hpp` (namespace `swaps::testing`),
      not in the shipped headers. They duck-type the calibration interface, so the templated
      LM/AAD/`calibrate`/`aad_jacobian` code drives them unchanged. Decomposition exact to 1e-16; recovers
      a known spread to 1.5e-13; spread AAD Jacobian matches bump to 1e-7 — see tests/spread_test.cpp.)*
- [x] **Phase 5** — Vectorized portfolio analytics + analytic bucketed delta.
      *Analytic bucketed delta (`swaps/calibration/risk.hpp`): AAD `d(NPV)/dx` + IFT
      `dx/dq = (JᵀJ)⁻¹Jᵀ` → full ladder from one calibration. Matches bump-and-recalibrate to ~2e-8;
      **39× vs QuantLib bump-and-reprice**.*
      *Vectorized book reprice (`swaps/portfolio/compiled.hpp`): `W` (cashflow-time × knot integral
      weights) built once via one AAD pass; reprice = `DF = exp(-Wx)` + gathered elementwise coupon
      math + sparse per-swap reductions, no scalar loop. Matches the scalar kernel to ~1e-15;
      **~126× vs QuantLib's per-swap `NPV()` loop** on a 1000-swap book (0.19 ms vs 23.9 ms,
      fingerprint `52e94be82bc4`). All three perf baselines now populated.*
- [x] **Phase 6** — Perf-gate hardening. `tools/check_perf.py` runs the benchmarks, computes the
      machine+toolchain fingerprint, REFUSES to compare across fingerprints, and HARD-gates on
      `min_speedup_vs_quantlib` (load-robust). `max_self_regression` is advisory (absolute ns is
      load-sensitive): exceeding it only warns; only a gross >2× regression hard-fails. `verify.sh`
      gates on it — both gates green report `perf=PASS`. Re-baseline with
      `SWAPS_CAPTURE_UTC=$(date -u +%FT%TZ) ./tools/check_perf.py --build build --baselines
      baselines/baselines.json --update`. *(Nice-to-haves: quiesced re-capture for authoritative
      absolute ns; further SIMD/layout tuning.)*
- [x] **Stage 2** — Warm/streaming re-calibration (see §7a): cached-Jacobian Newton (48 µs exact) and
      the accuracy-first exact frozen-Newton streaming path (exact every tick, staleness-gated recompute).
- [x] **Stage 3** — Curve bundle (see §7b): N curves calibrated simultaneously over a stacked `x`,
      multi-curve pricing kernel (forecast ≠ discount) + basis chains, joint and staged (SCC-decomposed)
      solves. Realistic SOFR+FF+PRIME+PRIME2 chain validated vs QuantLib to ~1e-16; ~20× vs QuantLib
      IterativeBootstrap on a production swap grid. Outright-vs-spread is inherent in the curve DEFINITION
      (`CurveSpec.base`), so a curve quoted as `base + spread` calibrates jointly (or over a fixed base)
      with no solver change. *(Next: real FF-averaging-futures QuantLib helpers for a fully-faithful build
      benchmark; real-time cross-curve risk ladder.)*
- [x] **Stage 4 — generic instrument pipeline (see §7c) — COMPLETE (adopted + legacy deleted).**
      Design `docs/generic-instrument-pipeline.md`. The generic kernel (§2), the ONE `BundleFloatBatch`/
      `BundleFixedLegs` primitives (§4), the coupon-type-dispatch extractors (§5) and the `Instrument`/leg/
      role/quote model on the analytic W-cache (§3) are now the ONLY path. Every call site — reference
      market (`build_problem`/`build_square_problem`), `Portfolio`, single-curve `CalibrationProblem`,
      multi-curve `BundleProblem`, and ALL benchmarks — constructs generic `Instrument`s. The legacy
      `OisSwap`/`CompoundedFuture`/`AveragedFuture` structs, their pricing functions, their extractors and
      the compiled legacy adapters are DELETED. Byte-safety of the swap migration is pinned by
      `tests/migration_guard_test.cpp` (the `valueDates().front()/back() == accrual dates` date fact, which
      made the generic swap reprice the old form bit-for-bit). Gate: correctness 85/85 (the 6 pure legacy-vs-
      generic reduction tests were removed with the structs, their job done); perf re-baselined on the
      generic path (curve-build now measures the adopted pipeline). CLAUDE.md §7c is the current reference.
- [ ] **Stage 5 — B-spline curve type + moment integration — PARTIAL, branch `feat/bezier-and-moment-integration`.**
      Design `docs/bezier-and-moments.md`. **DONE & gate-verified (89/89, perf PASS):**
      - **B-spline curve type (Part A) — COMPLETE.** Control-point clamped cubic (`BSpline` region,
        `flat_bspline` layout): C² + convex-hull, `is_linear_map`, validated end to end — QuantLib OIS
        oracle 6.9e-17, calibration fit (identifiable), W-cache reprice 1.1e-16, `bspline_collocation`
        risk transform. See the interpolation bullet in §2.
      - **Moment-integrated averaging (Part B) — landed as a FAST APPROXIMATION, additive/opt-in.**
        `RateObservation.fixing_step > 0` selects it in the ordinary coupon path; the numerator is
        `∫f + ½·⟨τ²⟩·∫f² (+ ⅙·⟨τ³⟩·∫f³)` — closed-form curve moments instead of a day-by-day sum. **HONEST
        LIMIT:** it matches the exact daily average to ~7e-11 on UNIFORM daily fixings but floors at **~5e-9
        on a REAL calendar** (weekend 3-day accruals; the `f`-variation × day-structure correlation in the
        2nd-moment coefficient, which higher moments do NOT remove). So it is a fast approximation (~5e-5 bp,
        far below market relevance), NOT a 1e-10 replacement; the exact sub-period path (`fixing_step==0`)
        stays available and bit-for-bit unchanged. `tests/bspline_oracle_test.cpp` isolates
        moment-vs-exact-daily (~5e-9) from exact-daily-vs-QuantLib (~1e-16).
      **NOT done: observation shift & lookback/lockout — but the QL upgrade that UNLOCKS them is DONE**
      (branch `chore/quantlib-1.35-bump`). QL 1.34's `OvernightIndexedCoupon` had NO obs-shift/lookback/lockout
      API; **QuantLib 1.35 ADDED lookback days, lockout days and observation shift** to
      `OvernightIndexedCoupon`/`OvernightIndexedSwap` + helpers (Marcin Rybacki), so the oracle now EXISTS.
      `tools/bootstrap_deps.sh` is `QL_VER=1.35`, the full oracle suite is **green (94/94)** against 1.35, and
      perf is re-baselined (fingerprint `a8c9a844826e` UNCHANGED — QuantLib version is not in it — only the
      committed comparison timings were refreshed; still PASS). The bump was low-risk as scoped (narrow QL
      surface, no `-Werror`, none of 1.35's removed symbols used, `RelinkableHandle` default-ctor only). **The
      ONE behavioral change 1.35 introduced:** `OvernightIndexFuture::averagedRate()` (1M arithmetic future) was
      refined — the per-day fixing is now read at `calendar.adjust(d1, Preceding)` and the last day's accrual is
      capped at `min(d2, maturity)` — which shifted the realized-fixing current-month contract by ~2.4bp. That
      is oracle-side (our engine math is unchanged); `tests/reference_curve.hpp` `avg_future_obs` was aligned to
      1.35's loop exactly (weighted forecast sub-periods; interior days weight 1.0 so fully-forecast futures stay
      bit-identical). The macOS-14.5-SDK libc++ workaround (std::format ADL) is **still needed** for 1.35 (built
      clean WITH it; not tested without).
      **DONE: observation shift + lookback + lockout are now BUILT and gate-verified** (branch
      `feat/rfr-lookback-lockout-obsshift`). These RFR conventions break the telescoping that collapses a
      compounded coupon to one DF ratio, so `RateObservation` gained a **COMPOUNDED (product) mode**
      (`compounded` + `realized_factor`, `pricing/cashflows.hpp`): the numerator is
      `realized_factor·∏_k(1 + w_k(DF(s_k)/DF(e_k) − 1)) − 1`, each factor `1 + fixing(f_k)·dt_k`, with the
      per-fixing observation over the index's own overnight period so `w_k = dt_k/τ_k` (== 1.0 for a plain
      day, carrying the lookback/lockout day-count skew otherwise). `ql/extract.hpp`
      `extract_overnight_rfr_obs` builds it from the QL 1.35 coupon's `fixingDates()`/`dt()` when
      lookback/lockout/obs-shift are present; a **standard** compounded/averaged coupon keeps the telescoped
      path bit-for-bit (calibration untouched). Validated against QuantLib 1.35 as a composite oracle (our
      curve wired in as the term structure) to <1e-12 across lookback-no-shift, obs-shift, lockout, combined,
      and a partially-realized straddle (`tests/rfr_coupon_oracle_test.cpp`). Gearing on a compounded coupon
      multiplies the whole `(growth − 1)` (not per-factor), so geared RFR coupons are rejected (SOFR/SONIA
      FRNs are gearing 1). **Scope boundary:** the compounded product lives in the TEMPLATED kernel
      (`float_coupon_pv`/`rate`, AAD-safe) only — the arithmetic compiled W-cache batch (`push_obs`) throws on
      a `compounded` observation, because RFR coupons are pricing coupons, never calibration instruments.
      Obs-shift telescopes (the product still evaluates it exactly); lookback-no-shift/lockout genuinely need
      the product. NB the changelog warns lookback is **incompatible with the telescoping formula** unless the
      observation shift is also applied — we build coupons non-telescopic, sidestepping it.
      Also not done: a `scheme` selector on the problem structs so a real Bundle/CalibrationProblem *selects*
      B-spline or the new MonotoneCubic (the W-cache supports the linear ones; only the standalone
      `BSplineProblem`/`MonotoneCubicProblem` tests exercise a non-default scheme today).
- [x] **Stage 6 — bonds & bond asset swaps — DONE (pricing + universe sweep + asset swaps + API verbs + perf gate).**
      Merged to `main`. Design `docs/bond-pricing.md`. A bond is DATA (dated
      cashflows + a small yield convention), NOT a subclass — "US Treasury" is field values a builder fills
      in (semiannual, ACT/ACT ICMA, street f=2), never a type in engine code (§0/§1). **DONE & self-check
      green (QL-free `tests/bond_yield_test.cpp`; QL oracle `tests/bond_oracle_test.cpp` written, runs under
      the gate on a QL-built host):**
      - **Two pricing modes, both native shapes** (`pricing/bond.hpp`, templated on Scalar):
        (A) CURVE space — `bond_dirty_price = Σ amount·DF(pay)/DF(settle)`, LINEAR in `DF = exp(-Wx)`, so a
        universe rides the SAME W-cache as calibration/the swap book; `bond_z_spread` is a per-bond Newton
        against those DFs. (B) YIELD/street space — `dirty(y) = Σ CF·(1+y/f)^{−E}` with cumulative exponent
        `E_i = w + i` (ACT/ACT ISMA), which reproduces QuantLib's chained per-period discounting EXACTLY, so
        price/yield/accrued/`bond_risk` (modified & Macaulay duration, convexity) are penny-perfect vs
        `QuantLib::BondFunctions`.
      - **Universe sweep** (`portfolio/bond_universe.hpp`): `BondUniverse` is the price↔YTM cache. YTM has
        NO shared curve (each bond its own y), so `DF=exp(-Wx)` doesn't apply directly — but for a REGULAR
        bond the exponents are arithmetic (`E_i=w+i`), so `P=v^w·Σ CF_i v^i` is `v^w` × a coupon POLYNOMIAL
        in `v=1/(1+y/f)`. The cache is the coefficient matrix + `w` + `f` (structure-only); the hot loop is
        HORNER — `Q,Q',Q''` in one FMA sweep per cashflow column (synthetic differentiation), then one
        `pow(v,w)` per bond + chain rule — so a batched-Newton iteration is O(cashflows) FMAs + O(bonds) pows,
        NOT O(cashflows) transcendentals, and still exact/penny-perfect. `yields_from_clean` solves the whole
        universe in ONE batched Newton (converged bonds self-arrest, no masking); the REVERSE `dirty_prices`/
        `clean_prices` (yield→price) is the same cache run value-only, allocation-free (const-ref scratch), so
        re-marking a universe as yields move never allocates. ACCRUED is a pure schedule quantity
        (coupon·day-fraction, no yield/price/curve), computed once at build (`build::accrued_interest`, the one
        definition) and cached — `BondUniverse::accrued()` is O(1) and converts clean↔dirty both ways; it (and
        `w`) are linear in settlement, so a rolled settlement recomputes O(1) off the cached `BuiltBond` period
        bounds, no rebuild (a coupon crossing does need one). Irregular schedules fall back to a general
        per-cashflow `exp` path (`is_regular()` gates it). `CompiledBondBook` is the curve-space counterpart:
        build W once, reprice PV/dirty/clean + per-bond z-spread off `DF=exp(-Wx)`
        (z-spread/asset-swap/relative-value), no per-bond QuantLib pricing.
      - **Construction** (`build/bond.hpp`): `fixed_rate_bond`/`us_treasury` build both representations from
        bond terms; ACT/ACT ISDA (`year_frac`) + ACT/ACT ICMA (`act_act_icma`) added to `build/day_count.hpp`.
      - **When-issued (WI)** (`when_issued_bond`/`us_treasury_wi`): settle on the dated date (new issue => ZERO accrued; a reopening settles later within
        the first period => accrued from the ORIGINAL dated date), with a SHORT first coupon PRORATED to
        actual days (`coupon/f·(first_coupon−dated)/E`). Stays on the Horner fast path — the short coupon only
        changes the first coefficient, not the exponent spacing, so `is_regular()` holds. Long first coupon
        (spans >1 quasi-period) is rejected (follow-up). Gated by `BondWhenIssued.*`. **The PRORATION is
        31 CFR Part 356 App B; the DISCOUNTING is NOT** — see the convention note below.
      - **STREET vs TREASURY (31 CFR App B) discounting — a real convention gap, found AND now closed.**
        Every street convention agrees on the cashflows, on accrued and on the coupon polynomial `Q(v)`;
        they differ ONLY in how the FRACTIONAL first period `w` is discounted. `pricing::YieldConvention
        {freq, stub, final_period_simple}` is that one degree of freedom (rateslib factors the same thing
        as v1/v2/v3; interior periods are always regular, so two fields cover it):
          * `stub=Compound, final=false` — `Q(v)·v^w`. UK gilt / French OAT / Chinese GB. **Default**, and
            what the kernel always computed. Oracle: QuantLib `Compounded`.
          * `stub=Compound, final=TRUE` — compound, but SIMPLE once only one cashflow remains. **US
            Treasury STREET** and Bund (`build::us_treasury`). Before this landed we compounded to the end
            and were **~0.7 bp rich on every bond in its last six months** — invisible because no test had
            a final-period bond, and because our QuantLib oracle used `Compounded` too (a shared
            assumption, the same failure mode twice).
          * `stub=Simple` — `Q(v)/(1+w·y/f)`. **31 CFR Part 356 App B / Bloomberg Treasury method**
            (`build::us_treasury_tsy`, `us_treasury_wi_tsy`). The regulation is uniform: every Section II
            sub-case is written `P[1 + (r/s)(i/2)] = …`, never a compound `(1+i/2)^(r/s)`.
        **BOTH forms have a first-class QuantLib oracle** — `Compounding::SimpleThenCompounded` reproduces
        App B exactly (it applies simple interest precisely when the step `t <= 1/f`, i.e. the stub), so
        neither convention rests on a hand-rolled formula. ⚠️ but QuantLib's `BondFunctions::duration`/
        `convexity` are NOT the derivative of its own price under `SimpleThenCompounded` (npv chains
        STEPWISE factors; modifiedDuration branches on CUMULATIVE time then compounds purely): measured
        1.4e-4 relative vs a central difference of its own price, against 2.1e-11 under `Compounded`. Ours
        matches that finite difference to ~1e-10, so the oracle test checks prices/yields against
        QuantLib's analytic values and duration/convexity against an FD of QuantLib's PRICE.
        **Perf is untouched:** TIMING conventions stay baked into the exponent `E_i` (the Horner fast
        path); only the closing factor differs (`v^w` vs `1/(1+w·y/f)`, the latter needing NO `pow`), a
        mixed universe blends the two lanes by a 0/1 mask with no scalar remainder, and a compound-only
        universe early-returns through byte-identical arithmetic — `bond_sweep` unchanged.
        Gated by `BondOracle.{TreasuryMethodMatchesSimpleThenCompounded,
        StreetSwitchesToSimpleStubInTheFinalPeriod, MixedConventionUniverseMatchesTheScalarKernel}`.
        NOT done: a TIPS index-ratio mode (needs a builder for the 3-month lag + daily CPI interpolation,
        and the principal deflation floor is optionality, not data), and Gilt ex-div / BTP pay-adjust.
      - **Bond types are DB entries, not code (§0).** `conventions/conventions.json` gained a `bonds`
        section (codegen'd to `BondConv`/`kBonds`/`conventions::bond(id)`), so a bond TYPE is an id:
        `build::yield_convention(id)` / `bond_from_convention` / `wi_bond_from_convention` pull frequency +
        stub rule from the DB, and the `bonds` run_json verb takes `convention` (default "US-TREASURY").
        Catalogued: `US-TREASURY` (street) and `US-TREASURY-TSY` (App B). **Only conventions supported END
        TO END — builder AND QuantLib oracle — are listed.** UK gilts / OATs / Bunds are ONE entry away
        (the kernel already reproduces their yield math exactly: verified against Rateslib `uk_gb`/`fr_gb`
        to 12 digits on an annual bond) but gilts need ex-dividend + a GBP calendar and Bunds a EUR bond
        calendar, so they are deliberately ABSENT rather than listed half-supported; BTPs additionally need
        a payment-date adjustment. `Conventions.BondConventionsDriveTheNamedBuilders` pins DB == named
        builder == QuantLib's two compounding modes, so the JSON, the builders and the oracle cannot drift.
      - **WHEN-ISSUED is now reachable across the API seam** (it previously had no JSON path at all): the
        `bonds` verb takes `dated` + `first_coupon` in place of `issue`. Gated by
        `BundleApi.BondsVerbSelectsConventionAndHandlesWhenIssued`.
      - **Cross-validation beyond QuantLib** (`tests/bond_reference_test.cpp`, QL-free; `tools/bond_reference/`):
        Excel/OpenFormula PRICE/YIELD (reimplemented — different algebra than Horner) is checked in-code to
        1e-12, and the 31 CFR App B reimplementation now pins the coupon polynomial exactly AND the
        street↔Treasury factor above. `gen_golden.py` emits an external **Rateslib** golden in BOTH modes
        (`mode` column): `us_gb` asserted EQUAL to 1e-9, `ust_31bii` asserted equal after the convention
        factor — so `BondReference.ExternalGoldenIfPresent` would catch drift in EITHER convention. It still
        skips if the CSV is absent. **Rateslib is source-available, NOT open-source** (an earlier note here
        and in `gen_golden.py` said "MIT" — wrong): without a registered licence, use is non-commercial only
        (<https://rateslib.com/licence>). Generating or committing that golden is a use of it; the QuantLib
        oracle + the two QL-free reimplementations need no third-party code. A single oracle can hide a
        shared convention assumption — which is exactly what happened here.
      - **Asset swaps + API verbs — DONE.** Par-par ASW spread vs a `QuantLib::AssetSwap::fairSpread` oracle
        (`tests/bond_asset_swap_oracle.cpp`, 5e-5); the stateless street-space `bonds` run_json verb; the
        curve-space `asset_swap` verb off a bundle.
      - **PERF GATE — DONE** (`bench/bond_sweep_bench.cpp`, metrics `bond_sweep` + `bond_book`). 5,000
        seasoned treasuries off the SAME QuantLib/compiler/flags. Ours 3.51 ms vs `BondFunctions::yield`
        bond-for-bond 2,500 ms = **712×**; curve-space book 414 µs vs a per-bond `DiscountingBondEngine`
        loop 27.9 ms = **67×**. **Never quote the 712× alone:** measured with a counting solver in the same
        `CashFlows::yield<Solver>` template, ONE bond costs 34 npv + 29 duration = **63 leg walks**, because
        `IrrFinder::derivative` returns `modifiedDuration = −P′/P` where the objective needs `−P′ = P·modDur`
        — in the 100-face basis QuantLib normalizes to, that is a ~99× under-scaled Newton step and the
        safeguarded solver mostly bisects. `BM_BondSweep_QuantLibTuned` re-runs the SAME QuantLib pricing
        from a correctly-scaled Newton (~4 iterations): 459 ms, i.e. **125×** — the honest kernel-vs-kernel
        number, and the `bond_sweep` threshold (50×) sits below BOTH. A leg walk is ~11 µs / 45 cashflows,
        ~215 ns of the ~245 ns per cashflow being `ActualActual(ISMA)::yearFraction`; we bake that structure
        into the exponents once. The fixture ABORTS unless both paths agree (1.0e-14 on yields, 5.9e-15 rel
        on curve prices) so a mis-set-up universe cannot quote a speedup. Size sweep (opt-in
        `SWAPS_BOND_SCALE=1`): no small-universe crossover — 680/600/736/738 ns per bond at 100/1k/5k/10k
        vs a flat ~500 µs per bond.
      **PENDING (next):** curve-space key-rate DV01 beyond PV; TIPS (index ratio + real yield); long-
      first-coupon when-issued; non-treasury bond types (Gilt/Bund/corporate/FRN) as new builders filling
      the SAME structs — the yield-convention slots they need now exist, so most are a one-liner over
      `fixed_rate_bond` plus their own accrual/ex-div rules.
