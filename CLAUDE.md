# CLAUDE.md — SwapsEngine

Guidance for Claude Code when working in this repository. Read this first, every session.

## 1. What this project is

SwapsEngine is a high-performance interest-rate **swap-curve calibration engine** and
**vectorized swap-analytics library**. It is designed to be *provably faster and more accurate*
than stock QuantLib on the specific workflow it targets. QuantLib is **not** a dependency of the
shipped engine — it is used only as (a) the correctness **oracle** and (b) the speed **baseline** to beat.

### North-star capabilities
1. **Global curve calibration.** All knot forwards solved **jointly** with Levenberg–Marquardt,
   not sequential 1-D bootstrapping.
2. **Analytic Jacobian via AAD.** The LM Jacobian `J[i][k] = d residual_i / d knot_k` is computed
   by **forward-mode "vector-dual" automatic differentiation** (Eigen `AutoDiffScalar` carrying a
   length-M derivative vector), in one differentiated evaluation — not by bump-and-reprice.
3. **Analytic bucketed risk.** Via the implicit-function theorem, `dx/dquote = -(dr/dx)^{-1} (dr/dquote)`,
   reusing the calibration Jacobian, so delta ladders are analytic (no bumping).
4. **Vectorized portfolio analytics.** A portfolio of P swaps is priced as batched Eigen matrix–vector
   algebra (par rate, NPV, PV01/DV01, bucketed delta, convexity) — **no per-swap loop**.

## 2. Curve model (the math the code must implement)

- Free variables `x = (f_1 … f_M)` are **forward rates at knot points**. Discounting is
  `DF(t) = exp(-∫_0^t f(u) du)`.
- **Spread curves:** if a curve is defined as a spread to a base curve, the free variables are
  **forward spreads** `s_k` and `forward(t) = forward_base(t) + spread(t)`; the base curve is held
  fixed (jointly-calibrated base is a later extension).
- **Interpolation is two-region:**
  - **Front end (up to the last central-bank meeting date):** instantaneous forward is
    **piecewise-flat**, breakpoints at **CB meeting dates**. Forwards jump only at meetings.
    Knot (meeting) dates are **decoupled** from instrument maturities — this is why calibration is a
    global least-squares solve, not 1:1 bootstrapping.
  - **Back end (beyond the last meeting date):** **smooth forwards** with **C¹ and C² continuity**
    across the long knots (cubic-spline-class interpolation on the forwards).
  - **At the join** (last meeting date): enforce **level continuity** of the forward into the spline;
    do **not** impose C¹/C² across the join (the front end is intentionally discontinuous).
- A cubic spline's coefficients are a **linear** map of the knot forwards, so the whole interpolator
  stays cleanly differentiable — preserve that property; do not introduce non-differentiable kinks in
  the back end.

## 3. THE TWO GATES (non-negotiable)

Every checkpoint must pass **both** gates. `./tools/verify.sh` runs them and prints a pass/fail table.

### Correctness gate (GoogleTest, `tests/`)
- Our discount factors and par rates match QuantLib within tolerance.
- Our AAD Jacobian matches a QuantLib bump-and-reprice Jacobian within tolerance.
- Batched portfolio analytics match per-swap QuantLib pricing within tolerance.
- **Rule: never regress correctness. A failing correctness test blocks the checkpoint. No exceptions.**

### Performance gate (Google Benchmark, `bench/`)
- Curve-build time, batch-analytics time, and AAD-risk time are compared against QuantLib baselines
  stored in `baselines/baselines.json`.
- The gate **fails if we are not measurably faster than QuantLib** by the agreed thresholds, and also
  fails on self-regression vs our own committed baseline.
- **Rule: every commit that changes engine code must include or refresh a benchmark proving the
  speedup. If the performance gate fails, do not commit the change as an improvement.**

### Numerical tolerances (single source of truth: `tests/tolerances.hpp`)
- Discount factors / par rates vs QuantLib: `rel <= 1e-10`.
- AAD Jacobian vs bump-and-reprice: `rel <= 1e-6` (bump noise dominates; tighten if we refine the bump).
- Update tolerances only with a documented reason in the commit message.

## 4. Build / test / bench commands

```bash
# Configure + build (Ninja + ccache)
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
- Prefer the Homebrew LLVM toolchain over Apple clang (better auto-vectorization, C++20).

## 5. Coding standards

- **C++20.** The engine is **header-only and templated on the scalar type** (`double` for pricing,
  `AutoDiffScalar<…>` for AAD). Never hard-code `double` in engine math — use the template scalar.
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
- Commit trailer (both gates green): `Verified: correctness+perf gates passing`.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Major checkpoints** get an annotated tag `checkpoint-NN` and are **pushed to the web backup**
  (GitHub private repo `origin`) **only after both gates pass**. Use `./tools/checkpoint.sh "message"`.
- **Do not push to the web backup automatically or on every commit.** Pushing publishes outside the
  machine — only at major checkpoints, gates green, and surface what is being pushed.

## 7. Directory layout

```
include/swaps/curve/         interpolation (meeting-date flat + smooth spline), discounting
include/swaps/ad/            AAD scalar typedefs / dual helpers
include/swaps/calibration/   LM solver wrapper, residuals, implicit-function-theorem risk
include/swaps/portfolio/     vectorized swap analytics
src/                         non-header impl / example drivers
tests/                       GoogleTest correctness gate  (tests/golden/ = committed reference data)
bench/                       Google Benchmark performance gate
baselines/baselines.json     committed baseline timings (ours vs QuantLib)
tools/                       verify.sh, checkpoint.sh, gen_golden, etc.
third_party/                 QuantLib + Eigen source (gitignored; built locally)
```

## 8. Phased roadmap (update the checkbox as phases land)

- [ ] **Phase 0** — Toolchain, repo, CLAUDE.md, CMake skeleton, QuantLib baseline builds.
- [ ] **Phase 1** — Correctness harness + golden reference curve from QuantLib.
- [ ] **Phase 2** — Core engine: two-region interpolation + discounting + global LM with numerical Jacobian.
- [ ] **Phase 3** — AAD Jacobian (forward-mode vector-dual); verify vs bump; verify speed.
- [ ] **Phase 4** — Spread curves (forward-spread interpolation to a base curve).
- [ ] **Phase 5** — Vectorized portfolio analytics + analytic bucketed delta.
- [ ] **Phase 6** — Perf-gate hardening, SIMD/layout tuning, checkpoint/backup automation.
