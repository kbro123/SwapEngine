# OPTIMIZATION.md — how the swap engine got fast (and the playbook for doing it again)

The clean record of the SwapEngine calibration/streaming optimization journey: the techniques, the commits,
the results, and the repeatable **method**. Read `ARCHITECTURE.md` first for the object graph; this is the
performance layer on top of it. When we optimize the **options** pricing path, this is the template.

---

## The problem

Calibrate a multi-curve bundle (N curves, tens of knots) to market quotes, then **re-calibrate to a live
drifting market every tick** at microsecond latency for streaming. Two regimes with different bottlenecks:

- **Cold calibrate** — solve `x` from scratch (Levenberg–Marquardt). Cost = iterations × (residual + Jacobian).
- **Warm stream** — re-solve as the market drifts a little each tick. Cost is dominated by *avoiding* the
  expensive parts (curve rebuild, Jacobian factorization) that barely change tick to tick.

## The measurement discipline (do this first, always)

You cannot optimize what you do not measure, and a "faster" change that trips the oracle is a regression.

- **Perf gate** (`tools/verify.sh` → `bench/`): four benchmarks — `curve_build`, `risk_full_jacobian`,
  `portfolio_analytics`, `warm_recalibration` — each with a `need>=` speedup floor and a `regr` factor vs a
  recorded baseline, gated on a **CPU fingerprint** (so baselines are comparable). Runs on every change.
- **Engine-stamped timing** (`379ed54`): the session stamps `last_solve_us` + a streaming running average
  *inside* the engine, around the pure compute — so we measure the kernel, not Python/marshalling overhead.
- **A/B across the commit arc via git worktrees** (`tools/…` bench driver): run one fixed driver over a
  range of commits in parallel worktrees to pin *which commit* moved tick-time.
- **Correctness is the gate**: every perf change is proven against the QuantLib oracle in a *separate* test
  binary. `-fno-math-errno`/`noalias` etc. are only taken where they're oracle-safe.

Rule of thumb learned the hard way: **the benchmarks are load-sensitive** — re-run on a quiet machine before
trusting a `regr` reading; concurrent builds/agents inflate it by 10–20%.

---

## The techniques (grouped by the idea behind them)

### 1. Don't recompute what's constant — the **W-cache** (the single biggest win)
For a **linear-map** curve, the log-discount is an affine function of the knot forwards: `DF = exp(−W·x)`,
where `W` is a constant weight matrix. Precompute `W` once; then every calibration iteration and every
streaming tick is a cheap `W·x` + `exp`, with an **analytic Jacobian** that falls straight out — no curve
rebuild, no autodiff sweep. This is the "compiled" fast path and the reason streaming is µs.
- `integral_weight_matrix` / `bspline_collocation` build `W`.
- **Generic any-region W-cache** (`f225b8e`): extended from the shipped Flat+Hermite to *any* linear region
  scheme (Linear/NaturalCubic/Hermite/BSpline/Tension, `bb6f64b`) — only the value-dependent MonotoneCubic
  falls off it.
- **Memoize DF, dedup the Jacobian gather, reuse scratch** (`c19e8ec`); **fused fast path for the plain OIS
  coupon** (`cda378b`).

### 2. Use the right derivative tool — **analytic AAD, never bumping**
The engine is templated on `Scalar`, so the *same* kernel prices with `double` and yields the exact Jacobian
`J = dq/dx` with `Scalar = ad::Dual` (Eigen `AutoDiffScalar`) in **one** differentiated pass. Finite-difference
bumping (O(n_knots) re-prices) is never used for calibration or risk.
- **Two-tier residual engine** (`residual_engine<Problem>` trait): compile-time pick between the compiled
  W-cache path (linear) and the AAD tier (non-linear / regularized).
- **Analytic risk operator** `M = (JᵀJ + RᵀR)⁻¹Jᵀ` and analytic `cross_jacobian`/`transform_matrix`
  (`6b3e073`): risk ladders and cross-bundle transforms with **no bumping** — one AAD pass + linear algebra.

### 3. Streaming — don't redo per-tick work that barely changed
- **Frozen-Newton streaming** (`e68aa13`, `7eaf904`): re-solve each tick with a *frozen* Jacobian/factorization
  and Newton steps, instead of a full recalibrate. Banded (soft-least-squares) and mixed FX/MtM bundles
  stream this way. A `start_streaming` **step-tolerance knob** (`453aeed`) trades accuracy for speed.
- **Speculative background Jacobian** (`497d105`): recompute the Jacobian on a background thread so the
  refresh-tick latency spike is hidden from the hot path.
- **Reuse curve objects across ticks** (`e3afebf`): the topology is fixed, so `BundleCurveSet` holds the
  `double` + `Dual` handles once and overwrites knot forwards **in place** (`CurveHandle::set_forwards`) —
  zero per-tick handle/curve allocation.
- **Vectorized discount cache** for the FX/MtM streaming block (`72e38ce`) — **~3× tick**.
- **Fold the smoothness regulariser into the streaming operator** (`b786868`).

### 4. Bring the exotics onto the fast path (don't let one trade drop the book to AAD)
- **Hybrid residual engine** (`dc9b64a`): cacheable rows on the W-cache **plus** a **width-reduced AAD block**
  for only the non-cacheable rows. The AAD block seeds only the knots those instruments *touch* (dynamic-width
  `AutoDiffScalar` shrinks every gradient).
- **W-cache the FX forward** (`ba07fbf`, affine residual) and the **MtM xccy basis** (`44dc0e1`, par funding
  leg → ParSpread quotient), and **Portfolios of W-cacheable components** (`2955a57`). Result: a standard
  FX+MtM cross-currency book is now *fully* W-cacheable — the AAD block is empty, **~13µs/tick vs ~130µs**.

### 5. Low-level: allocation-free, vectorized, parallel
- **Allocation-free reprice/streaming loop** (`2596233`, proven by `alloc_free_test` compiling Eigen with
  `EIGEN_RUNTIME_NO_MALLOC`) — pre-sized workspace, zero heap in the loop.
- **Hand-written auto-vectorized gather loops** in the compiled kernel (`0e0572b`); **SoA batched portfolio
  analytics** (`simd::packet_size`, `compiled_book.hpp`).
- **AVX2 (256-bit) default over AVX-512** on Intel + rebaseline (`2981c6d`) — AVX-512 down-clocking lost.
- **`-fno-math-errno` + `noalias` the DF-refresh GEMV** (`ba2bf1d`), oracle-safe.
- **Persistent thread pool** for parallel calibrate + reprice (`20aff72`); **lock-free async pricer/calibrator
  split** for the live feed (`665f936`).

---

## Results (perf gate, current baseline)

| benchmark | speedup vs naive | notable |
|---|---|---|
| `curve_build` | ~30× | cold calibrate |
| `risk_full_jacobian` | ~36× | analytic vs bump |
| `portfolio_analytics` | ~279× | SoA + SIMD batched reprice |
| `warm_recalibration` | ~60× | frozen-Newton streaming |

FX + MtM cross-currency streaming: **~130µs → ~13µs per tick** once fully W-cacheable.

---

## The playbook (the repeatable method)

1. **Measure first** — engine-stamp the real cost; find the dominant term (build vs Jacobian vs solve vs
   reprice). Profile, don't guess. A/B across commits if a regression appeared.
2. **Classify the cost, then apply the matching tool:**
   - *Recomputing something constant?* → **precompute/compile it** (W-cache, memoize, fuse).
   - *Bumping for derivatives?* → **analytic AAD** (templated Scalar) or the **implicit-function theorem**
     for solver sensitivities (don't differentiate the solver).
   - *Re-factorizing every tick?* → **freeze** (frozen-Newton) + refresh in the background.
   - *Allocating / not vectorizing in the loop?* → **alloc-free workspace + SoA + SIMD**, reuse objects.
   - *One exotic dropping the whole book off the fast path?* → **hybrid**: keep the bulk compiled, isolate the
     exotic in a width-reduced block.
3. **Gate it** — prove correctness against the oracle, then lock the speedup with a perf-gate baseline so it
   can't silently regress.

---

## Applying this to the OPTIONS pricing path

**Analytic layer — done (web `/options` smile + ATM grid), as an ENGINE hot path.** The per-call analytic
math (Bachelier/SABR/swaption/CMS) was already cheap; the cost was **redundant calibration** — each page paint
calibrated the curve *four* times (smile and grid each probed for the forward, then priced). A swaption cell's
`(forward, annuity, expiry_years)` depends only on the **curve**; the vol + price are a *pure* function of
them — the W-cache split exactly.

A first cut cached the forwards in Python and repriced the smile with a *pure-Python* Bachelier/SABR. That
worked (bit-identical, ~70× on a slider move) but put pricing math in the web layer — a second implementation
to keep in sync, against the principle that **the engine is the one optimised core and every client is thin**.
So it was **redone in C++** as the options analogue of `price_portfolio`:
- **`vol_cube` verb + `BundleSession::price_vol_cube`** price a whole expiry×tenor×strike surface in one pass:
  each cell's forward/annuity from a **single `sample()` over the union of schedule times**, then a pure
  Bachelier/SABR pass into a flat SoA. Strikes as absolute / moneyness / ATM; delta/vega/gamma analytic.
- **Two tiers, like the swap path:** the stateless verb (Excel/one-shot — calibrate + price in one C++ call)
  AND the **held session** (`swapengine.Session.price_vol_cube`) that reuses an already-calibrated/streaming
  session so a live vol surface reprices with **no recalibration**. The web `/options` page holds one over a
  websocket; the Python lib exposes `swaption_grid`; both call the same kernel.
- **Result:** the whole 17-cell / 37-point page — first paint ~16 ms (4 calibrations) → ~4.3 ms (1 verb call);
  a **SABR-slider move on the held session ~16 ms → ~0.6 ms** (engine-stamped `price_us` ~0.45 ms), all in C++,
  no recalibration. Freeze the curve, tick the vol — the streaming W-cache split, now on the options path.
- **Gated** like the swap benches: `bench/vol_cube_bench.cpp` (`BM_VolCube_Warm`/`Cold`) is wired into
  `tools/check_perf.py` as an **ours-only self-regression metric** (a swaption reprice has no expensive
  QuantLib baseline the way risk-bumping does, so it gates on its own committed `ours_ns`, not a speedup floor).

**A QuantLib head-to-head that caught a real bug (`bench/vol_cube_ql_bench.cpp`).** Pairing the surface
reprice against QuantLib's own analytic path (its `DiscountCurve` + `bachelierBlackFormula`, schedules
precomputed) exposed that `price_vol_cube_json` was **rebuilding every cell's date schedule on every call** —
the calendar/holiday walk for ~190 pay dates, identical for each reprice. Result: **490µs vs QuantLib's 10µs,
~40× SLOWER**. The measurement discipline paid off exactly as intended — a "faster than before" story hid a
gross inefficiency the oracle-paired bench surfaced. Two fixes, in order:
1. **The W-cache idea applied *within* the reprice** — memoize the curve-independent schedule per cell, and
   cache `(forward, annuity)` against the curve state `x` so a vol-only reprice (SABR-slider tick) samples
   nothing, just the Bachelier pass. **490µs → ~50µs.**
2. **A native (non-JSON) entry point.** The residual was an interface artifact: `vol_cube` had *only* a
   `price_vol_cube_json` method, so parse + compute were fused and the bench measured our JSON marshalling
   against QuantLib's native loop. Split it — `price_vol_cube(const VolCubeSpec&)` does the pure compute,
   `price_vol_cube_json` is a thin parse-then-call wrapper (matching `price_portfolio` / `price_portfolio_json`)
   — and reserve the SoA output. The bench now calls the native method: **~11µs.**

**Result: at parity with QuantLib.** Native surface reprice **~10.8µs (flat) / ~11.3µs (SABR) vs QuantLib's
~10.1µs** — and ours *builds the full 12-array Greeks SoA* (price + delta/vega/gamma + metadata for 37 points)
that QuantLib's loop doesn't even store, so on equal work we are at least even. The pricing kernel is
QuantLib-exact (the oracle proves it to 1e-12); the batched-reprice throughput now matches it too. The whole
arc: **574µs → 11µs (~50×)**. Lessons re-learned: benchmark against an independent reference, not just your
own past self — and give every op a native entry point, never route a performance test through a serialization
layer. (The JSON verb still exists for the web/Excel seam; it now costs ~14µs of parse on top of the ~11µs
compute — fine for a web request, and off the hot path for native/bench callers.)

**Desk-scale portfolio** (`bench/swaption_portfolio_bench.cpp`, ours-vs-QuantLib scaling probe): a random book
of thousands of swaptions across the standard grid prices at **~2M swaptions/sec, at parity-to-slightly-ahead
of QuantLib's lean per-instrument loop** (~1.1–1.17×, 2k→32k) — *and* ours builds the full Greeks SoA for every
trade that QuantLib's scalar-accumulate loop doesn't. The per-cell forward/annuity cache dedups by node (N
positions across ~81 grid nodes → 81 forward/annuity builds), so the edge is largest on small books and
converges as the N Bachelier evals + SoA writes dominate. Confirms desk-scale robustness; the >QuantLib win
here is bounded by the same analytic-math ceiling as the surface reprice.

The next surface is the **batched vol cube across many curves** and the future **Monte-Carlo** path, where the
same ideas map directly:
- **Analytic AAD → reverse-mode AAD tape** for MC Greeks (many inputs → one price, ≤4× one price): the
  templated-Scalar discipline is *identical*, the mode flips from forward to adjoint.
- **W-cache/precompute → path pre-computation**: Sobol + Brownian-bridge factors, model `evolve`
  coefficients, and the discount/numeraire curve are constant across paths — compute once.
- **Frozen-Newton/refresh → freeze the LSM exercise boundary** (envelope theorem) for first-order Greeks.
- **Alloc-free + SoA + SIMD + thread pool → the path loop verbatim** (per-path record→backprop→wipe tape,
  thread-local buffers on the existing `ThreadPool`, SoA path batching, checkpointing at observation dates).
- **Perf gate → add options benchmarks** (a swaption reprice, a vol-cube build, an MC price+Greeks) with
  baselines, same discipline.
