# CLAUDE.md — SwapsEngine

Guidance for Claude Code when working in this repository. Read this first, every session.

## 1. What this project is

SwapsEngine is a high-performance **extension of QuantLib** that replaces its two slowest workflows
— swap-curve calibration and bulk swap analytics — with a globally-calibrated, AAD-differentiated,
vectorized implementation, while **reusing** QuantLib for everything else.

**QuantLib IS a linked dependency of the shipped engine.** We reuse its base functionality wholesale
— calendars, day counters, `Schedule`, instrument definitions (`OvernightIndexedSwap`, SOFR futures,
rate helpers), quotes, conventions. We do **not** reimplement calendars or schedule generation. Our
value-add is narrow and deep: the calibration solve and the bulk-pricing math.

QuantLib is **also** still the correctness **oracle** and the speed **baseline** — its native
`IterativeBootstrap`/`GlobalBootstrap` and its per-swap `NPV()` loop are exactly what we must beat,
compiled with the same compiler and flags (§3 perf-gate integrity).

### The division of labour (the core architectural rule)
- **Reused from QuantLib (setup, done once, not differentiable):** calendars, day counts, schedules,
  instrument/cashflow definitions. Extract dates and accrual factors from QuantLib objects here.
- **Ours (hot path, templated on `Scalar`, differentiable, vectorized):** the curve interpolation,
  the discount/forward math, the residual vector, and the batched portfolio kernels. QuantLib is not
  templated, so AAD cannot flow through `OvernightIndexedSwap::NPV()`; the differentiable kernel must
  be our own code consuming QuantLib-extracted schedules.

### North-star capabilities
1. **Our calibrated curve is a `QuantLib::YieldTermStructure`.** `make_calibration_curve<Scalar>`
   (`MultiRegionCurve<Flat, Hermite>`) is the templated math core; the generic
   `CurveTermStructure<Curve>` wrapper exposes it to QuantLib pricing engines for validation and reuse.
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

### Interpolation is a compile-time, multi-region policy (linear schemes keep the fast path)
The curve is `MultiRegionCurve<Scalar, Regions...>` (`curve/multi_region_curve.hpp`): an arbitrary
compile-time sequence of region policies (`curve/regions.hpp`), each **linear in its knot values**,
stitched with a C⁰ (level) `Boundary` handoff by default (optional C¹ where both regions support it).
The **shipped** curve is `make_calibration_curve` = `MultiRegionCurve<Flat, Hermite>` (flat meeting-date
front, **local C¹ Hermite** back) — used everywhere: calibration, pricing, risk, streaming, the bundle.
The old `TwoRegionForwardCurve` (Flat+NaturalCubic) wrapper was retired; `NaturalCubic`/`Linear`/`BSpline`
remain as available region policies. Region ctors reject duplicate/unsorted knots (a zero-length segment
is a 0/0 → silent NaN; caught at construction).
- **`BSpline` (clamped cubic, CONTROL-POINT) is an alternative back end** (`make_bspline_curve` =
  `MultiRegionCurve<Flat, BSpline>`; branch `feat/bezier-and-moment-integration`). Free vars are B-spline
  control points (P₀ pinned to the front boundary for a C⁰ join), giving **C²** and the **convex-hull**
  property (forwards can't overshoot; positivity enforceable) that Hermite's C¹ does not. de Boor eval,
  2-pt-Gauss `integral` (exact for the cubic). Fully validated: QuantLib OIS oracle 6.9e-17, calibration
  fit (identifiable), W-cache reprice 1.1e-16, and a `bspline_collocation` risk transform (control-point ↔
  forward-at-knot deltas, invertible). Control points don't lie on the curve, so risk is reported in the
  forward basis via that transform. `integral_weight_matrix(..., BackScheme::BSpline)` puts it on the fast path.
- **Rule: the interpolation must be a LINEAR MAP of the knot values to keep the microsecond path.**
  Only linear schemes (`Flat`, `Linear`, `NaturalCubic`, `Hermite`, `BSpline`) preserve `integral(t)=w(t)·x`,
  hence the `W`-cache, analytic Jacobian and warm update. `is_linear_map` (AND over regions) gates that tier.
- **Value-dependent schemes drop to the AAD tier — and `MonotoneCubic` is the first one BUILT and
  gate-verified** (`make_monotone_curve` = `MultiRegionCurve<Flat, MonotoneCubic>`). It is a C² natural
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
  makes the streaming Jacobian *refresh* cheap. AAD is now used in exactly one place: producing `W`
  once (`integral_weight_matrix`), which works generically for any linear region policy.
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

### Performance gate (Google Benchmark, `bench/`)
- Curve-build time, batch-analytics time, and AAD-risk time are compared against QuantLib baselines
  stored in `baselines/baselines.json`.
- The gate **fails if we are not measurably faster than QuantLib** by the agreed thresholds, and also
  fails on self-regression vs our own committed baseline.
- **Rule: every commit that changes engine code must include or refresh a benchmark proving the
  speedup. If the performance gate fails, do not commit the change as an improvement.**

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
  as our engine** (`-O3 -march=native`). Never benchmark our tuned build against a generically-
  compiled QuantLib package/bottle — that measures compiler flags, not our algorithm, and inflates
  the speedup. `third_party/` builds QuantLib with our flags for exactly this reason.
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

### Hard constraints of the dev machine — do not re-litigate these
- **MacBook Pro 15" 2017 (`MacBookPro14,3`), i7-7820HQ Kaby Lake, 4 physical cores / 8 threads, 16 GB.**
- **SIMD *on this host*: AVX2 + FMA, no AVX-512 → 4 doubles/register.** This is a *fact about this
  machine*, **not a design constant**. The ISA is detected automatically at configure time by
  `cmake/DetectISA.cmake` (AVX-512 / AVX2 / AVX / NEON / SSE2), so the project builds optimally on
  any host. See the no-hard-coded-width rule in §5.
- **macOS 13.7.8 (Ventura) is the final supported OS for this Mac.** No macOS upgrade is possible.
- **Cap build parallelism at 4 (physical cores). Never use ninja's default (`logical+2` = 10).**
  Ten concurrent clang processes on this 4-core/16 GB box exhausted memory and tripped an
  **APFS kernel panic** (`OSMetaClassBase::_RESERVEDOSMetaClassBase6`, panicking task `clang`),
  plus non-deterministic clang segfaults that look like compiler ICEs but are not — the same
  file compiles fine when run alone. `bootstrap_deps.sh` caps via `SWAPS_BUILD_JOBS`.
- **Never run benchmarks while anything else is compiling.** Perf numbers taken under load are
  garbage (we measured a load average of ~17 during a build).
- **Homebrew is "Tier 3" on macOS 13 → it ships NO prebuilt bottles.** `brew install` compiles
  everything from source and drags in `go`/`rust`/`llvm` build deps. **Do not use Homebrew for
  project dependencies.** Vendor them into `third_party/` instead.
- Toolchain is **Apple clang 15** via Command Line Tools for Xcode 15.4 (Apple clang 12, which
  shipped with this machine, cannot compile C++20 — it rejects `-std=c++20`).
  Ensure `xcode-select -p` → `/Library/Developer/CommandLineTools`, **not** the stale `Xcode.app` (12.4).

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

- **C++20. Links QuantLib.** The engine reuses QuantLib types for calendars/schedules/instruments.
  The *hot, differentiable kernels* (curve, pricing, residuals, batched analytics) are **header-only
  and templated on the scalar type** (`double` for pricing, `AutoDiffScalar<…>` for AAD) — never
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
- Hot paths: **no heap allocation in inner loops**, no `virtual` dispatch, no UB. Prefer Eigen fixed/
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
include/swaps/curve/         multi_region_curve.hpp + regions.hpp (Flat/Linear/NaturalCubic/Hermite;
                             region ctors reject duplicate/unsorted knots), calibration_curve.hpp
                             (make_calibration_curve = MultiRegionCurve<Flat,Hermite> -- the SHIPPED curve),
                             curve_module.hpp (runtime ModularCurve: build from CurveModule{knots,scheme}),
                             spread_curve.hpp, ql_term_structure.hpp (generic CurveTermStructure<Curve>)
include/swaps/calibration/   problem.hpp (CalibrationProblem + the GENERIC Instrument/FloatLeg/FixedLeg/
                             QuoteKind model -- see §7c), lm.hpp, risk.hpp, warm.hpp (cached-Jacobian +
                             linear update), streaming.hpp (exact frozen-Newton live feed),
                             residual_engine.hpp (per-problem residual/Jacobian engine trait driving
                             warm/streaming),
                             compiled_residual.hpp (single-curve = 1-curve delegate to compiled_bundle.hpp),
                             compiled_bundle.hpp (CompiledBundleResidual: multi-curve W-cache residual),
                             bundle_problem.hpp + bundle_stage.hpp (Stage 3 multi-curve bundle)
include/swaps/pricing/       templated, QuantLib-free pricing kernel (cashflows.hpp: the GENERIC
                             RateObservation/FloatCoupon/FixedCoupon model (§7c) ALONGSIDE the still-live
                             legacy OisSwap/CompoundedFuture/AveragedFuture structs; single- AND
                             multi-curve OIS: ois_par_rate/basis_par_spread with forecast != discount);
                             compiled.hpp (integral_weight_matrix W primitive) + compiled_book.hpp
                             (CompiledCurveSet + role-aware batches incl. BundleFloatBatch, the ONE
                             float primitive: the multi-curve W-cache engine)
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
- The fixed point is `r = 0` **for any invertible `M`**, so every tick lands on the EXACT solution
  (round-trip `‖model_rates(x) − q‖∞ ≈ 1e-12`, machine-exact) regardless of `M`'s accuracy. `M` only
  sets the convergence *rate* `ρ = ‖I − M·J(x)‖` (measured: ~0.04 at 10 bp of drift → ~3 steps/tick).
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
- **PLANNED — branch detection + parallel SCC solve (do AFTER the single-solve hot-path optimization is
  exhausted).** The staged solver already condenses the dependency graph into SCCs solved in dependency
  order. When that DAG BRANCHES (a star/forest topology — several curves each spread directly off the
  base, mutually independent), the independent SCCs can be solved CONCURRENTLY on a thread pool. This is
  the ONE architecture-aligned intra-calibration parallelism that stays DETERMINISTIC (each SCC is a
  separate independent solve — no shared FP reduction, so §5 reproducibility holds), and it drops
  straight out of the existing Tarjan decomposition: add a topological-level grouping (SCCs with all
  dependencies already solved form a parallel wave) and dispatch each wave across threads. NOT started —
  it is deliberately gated behind finishing the single-solve wins (compiled staged engine so each block
  is fast; analytic `W_all` build; leaner LM linear algebra), because those shrink the SERIAL fraction
  that otherwise caps any parallel speedup (Amdahl). Process-level parallelism is NOT the target here —
  a solve is 1–22 ms, far below process/IPC overhead; threads over independent SCCs is. (Across-problem
  parallelism — many independent calibrations for a scenario grid / multi-currency book — is already
  trivially available given the single-threaded, allocation-light engine; that needs no new code.)
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
      `TwoRegionForwardCurve<Scalar>` validated against QuantLib's BackwardFlat and natural-cubic
      interpolators to ~1e-16, with a 1bp negative control. Hull–White futures convexity matches
      `HullWhite::convexityBias` exactly. `ql_adapter.hpp` exposes our curve to QuantLib as a
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
      *(`swaps/curve/spread_curve.hpp`: `forward = base + spread`, `DF = base_DF·exp(-∫spread)`, spread
      interpolated with the same two-region scheme so it stays linear/AAD-differentiable; base fixed.
      `SpreadCalibrationProblem` reuses the instrument set + pricing of a `CalibrationProblem` and
      duck-types the interface, so the LM/AAD/`calibrate`/`aad_jacobian` code — now templated on the
      problem type — drives spread calibration unchanged. Decomposition exact to 1e-16; recovers a
      known spread to 1.5e-13; spread AAD Jacobian matches bump to 1e-7.)*
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
        `make_bspline_curve`): C² + convex-hull, `is_linear_map`, validated end to end — QuantLib OIS
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
