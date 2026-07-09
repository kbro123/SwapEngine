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
1. **Our calibrated curve is a `QuantLib::YieldTermStructure`.** `TwoRegionForwardCurve<Scalar>` is the
   templated math core; the `YieldTermStructure` wrapper exposes it to QuantLib pricing engines for
   validation and reuse.
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
    This is the workhorse oracle: it needs no QuantLib equivalent of our two-region interpolator.

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
include/swaps/curve/         two-region interpolation + discounting; ql_term_structure.hpp wrapper
include/swaps/pricing/       templated, QuantLib-free pricing kernel (plain-data cashflow schedules)
include/swaps/ql/            QuantLib -> plain-data schedule extractors (the one QL-touching layer)
include/swaps/ad/            AAD scalar typedefs / dual helpers
include/swaps/calibration/   residual vector (problem.hpp), LM solver (lm.hpp), IFT risk (risk.hpp)
include/swaps/portfolio/     portfolio NPV kernel (portfolio.hpp) + vectorized reprice (compiled.hpp)
src/                         non-header impl / example drivers
tests/                       GoogleTest correctness gate  (tests/golden/ = committed reference data)
bench/                       Google Benchmark performance gate
baselines/baselines.json     committed baseline timings (ours vs QuantLib)
tools/                       bootstrap_deps.sh, verify.sh, checkpoint.sh, gen_golden, etc.
third_party/                 Eigen, GoogleTest, Google Benchmark, Boost headers, QuantLib
                             (gitignored; fetched + built locally by tools/bootstrap_deps.sh)
```

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
- [ ] **Phase 4** — Spread curves (forward-spread interpolation to a base curve).
- [x] **Phase 5** — Vectorized portfolio analytics + analytic bucketed delta.
      *Analytic bucketed delta (`swaps/calibration/risk.hpp`): AAD `d(NPV)/dx` + IFT
      `dx/dq = (JᵀJ)⁻¹Jᵀ` → full ladder from one calibration. Matches bump-and-recalibrate to ~2e-8;
      **39× vs QuantLib bump-and-reprice**.*
      *Vectorized book reprice (`swaps/portfolio/compiled.hpp`): `W` (cashflow-time × knot integral
      weights) built once via one AAD pass; reprice = `DF = exp(-Wx)` + gathered elementwise coupon
      math + sparse per-swap reductions, no scalar loop. Matches the scalar kernel to ~1e-15;
      **~126× vs QuantLib's per-swap `NPV()` loop** on a 1000-swap book (0.19 ms vs 23.9 ms,
      fingerprint `52e94be82bc4`). All three perf baselines now populated.*
- [ ] **Phase 6** — Perf-gate hardening, SIMD/layout tuning, checkpoint/backup automation.
