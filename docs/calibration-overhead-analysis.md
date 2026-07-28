# Calibration pipeline — overhead-reduction analysis

Research doc (branch `research/calibration-overhead`, cut from `main` @ 453aeed). Deliverable is
analysis + a ranked plan; **no engine source is changed here**. Everything below is grounded in the
code as read, plus micro-benchmarks compiled with the project's real flags (`-O3 -march=native
-DEIGEN_ENABLE_AVX512`) on the build host (Intel Xeon W-3223, Cascade Lake: AVX2 + AVX-512F/DQ/CD/BW/VL).

Raw benchmark numbers are in the appendix; the sources are in the session scratchpad
(`ubench*.cpp`), not committed (they `-I` the main checkout's `third_party/eigen`).

---

## TL;DR — the measurements overturn the obvious hypotheses

Three "obvious" optimizations were tested and **refuted**; documenting them saves the next session the effort:

- **"Hand-vectorize the DF `exp` with AVX-512 intrinsics."** Eigen's `array().exp()` already vectorizes
  (2.9× faster than a scalar `std::exp` loop). At the working-set sizes it is **memory-bandwidth bound**:
  AVX-512 (`EIGEN_ENABLE_AVX512`) and AVX2 measure **identical** (≈570 ns for 300 DFs). Wider SIMD buys nothing.
- **"Hand-write the `W·x` GEMV kernel."** Same story — the `W·x` DF layer is bandwidth-bound on the 240 KB
  `W` matrix; AVX-512 == AVX2 (≈3.4 µs). Intrinsics/asm cannot beat DRAM/L2 bandwidth.
- **"Devirtualize `CurveHandle::discount` in the hot loop."** The W-cache fast path (the one that runs
  every tick and every cold-LM iteration) contains **no virtual calls at all** — it is `DF = exp(-Wx)` +
  dense/sparse array math. Virtual `CurveHandle` dispatch exists only in the AAD tier, and a direct
  micro-benchmark shows the indirect-call cost there is in the noise (dominated by the `lower_bound` +
  cache misses, exactly as `curve_module.hpp:14` already notes).

What is *actually* worth doing, ranked by (impact × safety), is below. The headline real levers are
**(a) compiler flags** (free, safe) and **(b) killing the per-operation heap allocation inside the AAD
`Dual`** (the one hot-path cost the code comments explicitly defer). The single largest *compute* cost in
the pipeline is the **cold-calibrate Jacobian GEMM (~82 µs/iteration)**, but it is compute-bound dense
linear algebra already near Eigen's efficiency, and the naive sparse rewrite measures *slower* — so it is
a medium-payoff, medium-risk item, not a quick win.

---

## Where the time goes (measured, realistic bundle dims: n_knots≈100, n_times≈300, n_res≈80)

| Kernel | Where | Cost | Bound by | Runs |
|---|---|---|---|---|
| `W·x` GEMV (DF layer) | `compiled_book.hpp` `CompiledCurveSet::df_into` | ~3.4 µs | **memory bandwidth** (240 KB W) | every residual eval |
| DF `exp` | same, `(-out.array()).exp()` | ~0.57 µs | memory bandwidth | every residual eval |
| Frozen-Newton `M·r` matvec | `streaming.hpp:93` | ~0.5 µs | compute (tiny) | every Newton step |
| **Cold Jacobian `-(G·diagDF)·W`** | `compiled_bundle.hpp:181` | **~82 µs** | **compute (dense GEMM, ~58 Gflop/s)** | every LM iteration |
| AAD `Dual` op (per arithmetic op) | `ad/dual.hpp` `AutoDiffScalar<VectorXd>` | heap alloc/op | allocator | FX/MtM tick refresh; non-linear cold |

Interpretation:
- **Streaming per tick** (the "microsecond" money path) ≈ `W·x` + `exp` + `M·r` ≈ **~4.5 µs/Newton step**,
  1–2 steps/tick on a smooth feed, dominated by the bandwidth-bound `W·x`. The Jacobian refresh (rare)
  is the only place the 82 µs GEMM appears in streaming.
- **Cold calibrate** ≈ ~10 LM iterations × (residual + **82 µs Jacobian**) ≈ **~1 ms**, dominated by the
  Jacobian GEMM.

---

## Ranked recommendations

### 1. Compiler flags: add `-fno-math-errno`, LTO on the pybind TU; keep FMA at default; forbid `-ffast-math`
**Safety: very high. Impact: modest but free. (Highest impact×safety.)**

- **WHERE:** `CMakeLists.txt:15` (`CMAKE_CXX_FLAGS_RELEASE = "-O3 -DNDEBUG"`) + `cmake/DetectISA.cmake`
  (`-march=native`). No math or LTO flags are set today.
- **MECHANISM:**
  - `-fno-math-errno` — tells the compiler `exp`/`log`/`sqrt` do not set `errno`, so they become pure and
    can be hoisted/vectorized and don't spill around every call. Directly helps the **scalar** `std::exp`
    in `curve_forward_sq/cube_integral` and `obs_*` (AAD/templated path) and the `log()` in the FX
    residual (`compiled_bundle.hpp:110`). It does **not** change results (only the errno side effect).
  - **LTO** (`-flto=thin`) on `api/` (the pybind boundary `bundle_api.cpp` + `libswapengine`) so the
    JSON→`BundleProblem`→solve path inlines across TU boundaries. The engine is header-only (already
    inlined), so LTO's payoff is confined to the API/bench object files — small but free.
  - **FMA contraction:** leave clang's default (`-ffp-contract=on`, contract within a single expression).
    Eigen's GEMV/GEMM already emit FMA via intrinsics, so `on` vs `fast` barely moves the vectorized
    kernels (measured: `-ffp-contract=fast` changed the DF layer by <2%).
- **RISK TO ORACLE:** This is the crucial nuance. **Do NOT enable `-ffast-math` / `-funsafe-math-optimizations`
  / `-fassociative-math` / `-freciprocal-math`.** Reassociation and `x/y → x·(1/y)` change rounding by
  many ULP and would (a) risk the ~1e-10 QuantLib oracle gate and (b) **break the dozens of internal
  bit-identical gates** the code relies on ("bit-identical", "byte-for-byte", `x*1.0` exactness in
  `compiled_book.hpp`, `cashflows.hpp`, `compiled_bundle.hpp`). `-fno-math-errno` and `-ffinite-math-only`
  are the safe subset — but note `-ffinite-math-only` assumes no NaN/Inf, and the codebase deliberately
  reasons about NaN (`regions.hpp:29` rejects duplicate knots *because* they produce NaN), so **skip
  `-ffinite-math-only` too.** Recommend exactly: `-fno-math-errno` (global, safe) + ThinLTO (api/bench).
  - **Even `-ffp-contract=fast`** (contraction *across* statements) is risky here: two code paths that are
    algebraically identical but written differently (e.g. the fused `cpn_is_plain` path vs the general
    k-form in `compiled_book.hpp`) can contract to *different* FMA schedules and diverge in the last bit,
    tripping the internal bit-exact tests. If ever wanted, scope it per-file and re-run the bit-exact
    suite. Default `on` is fine and is likely already in effect.
- **FIRST STEP:** append `-fno-math-errno` to `CMAKE_CXX_FLAGS_RELEASE`; run `tools/verify.sh` in the
  engine repo (oracle + bit-exact gates) to confirm zero gate movement; then add `-flto=thin` to the
  `api`/`bench` targets only and re-time `calibration_bench`.

### 2. Kill the per-operation heap allocation in the AAD `Dual` (pooled / fixed-width gradient)
**Safety: medium. Impact: large — but only on the AAD tier (FX/MtM streaming refresh, non-linear MonotoneCubic cold solve).**

- **WHERE:** `ad/dual.hpp` — `using Dual = Eigen::AutoDiffScalar<Eigen::VectorXd>`. Every arithmetic op on
  a `Dual` allocates a fresh `VectorXd` gradient on the heap. Consumed by `aad_block.hpp`
  (`jacobian_vs_into`/`jacobian_into`, `instrument_residual<ad::Dual>`) and by `aad_jacobian`
  (`jacobian.hpp:16`) for any non-linear curve (`MonotoneCubic`). The `aad_block.hpp:16-18` header
  already flags this as the deferred cost: *"The per-operation gradient allocations inside Eigen's
  AutoDiffScalar remain — eliminating those needs a pooled Dual type, a separate future change."*
- **MECHANISM:** The block already does **width reduction** (seed only the touched knots, `aad_block.hpp`
  `touched_`), so gradients are already short (~10–20, not 100). The remaining cost is `malloc`/`free`
  churn — one allocation per `+`, `*`, `/` in the differentiated sweep. Two options, in ascending
  invasiveness:
  1. **Fixed-size gradient when the touched width is small and known.** `Eigen::AutoDiffScalar<Matrix<double,K,1>>`
     stores the gradient inline (no heap) for compile-time `K`. Instantiate the block's kernel for a few
     bucketed widths (`K = 8, 16, 32`, else fall back to dynamic). Removes 100% of the per-op allocation
     for the common FX/MtM case (touches 2 curves ≈ ≤32 knots).
  2. **Arena/pool allocator** behind a custom dual (or a monotonic buffer resource wired into the
     `VectorXd`), reset per residual eval. Keeps dynamic width, removes the `malloc` traffic.
- **EXPECTED:** For the FX/MtM streaming **refresh** (the AAD `jacobian_vs_into`) and any MonotoneCubic
  cold solve, allocation is the dominant per-op cost once width is already reduced — plausibly a **2–5×**
  reduction in the AAD sweep. No effect on the pure W-cache path (which never touches `Dual`).
- **RISK TO ORACLE:** Low-to-medium. The *arithmetic* is unchanged — a pooled/fixed-width dual computes
  the identical value and derivatives, so the oracle and bit-exact gates are untouched **if** the fixed-K
  path pads/masks correctly (a K larger than the true width, with zeroed tail, gives identical results).
  The risk is purely implementation correctness (width bucketing, fallback), not numerical.
- **FIRST STEP:** micro-benchmark `AadBlock::jacobian_vs_into` on a 2-FX-trade bundle with the current
  dynamic `Dual` vs `AutoDiffScalar<Matrix<double,32,1>>` to confirm the allocation share, before
  committing to the bucketed-instantiation machinery.

### 3. Cold-calibrate Jacobian: remove the per-iteration dense temporary; treat the GEMM as the real cost
**Safety: medium. Impact: medium (cold path only, which is infrequent). Honest caveat: the naive sparse rewrite is slower.**

- **WHERE:** `compiled_bundle.hpp:181`, `return -((G * DF.asDiagonal()) * cs_.W());`, plus the
  per-call `std::vector<double> qb` and the `num_pos/num_neg/num/ann` `VectorXd`s (`:129,143-146`).
- **MECHANISM (the safe part):** `G` is already a reused member (`G_`), but the expression allocates a
  fresh `n_res×n_times` temporary for `G·diagDF` and a fresh result matrix every LM iteration
  (measured ~192 KB temp). Give the engine a reused `GD_` member and a reused `J_` member and split with
  `.noalias()` (`GD_.noalias() = G_ * DF.asDiagonal(); J_.noalias() = -(GD_ * cs_.W());`). This removes
  the per-iteration allocations; it does **not** touch the ~82 µs GEMM compute.
- **MECHANISM (the compute part — investigate, don't assume):** the GEMM `G·W` (80×300×100) measures
  ~82 µs at ~58 Gflop/s — roughly 50% of AVX-512 peak, typical for Eigen at these non-cache-blocked dims.
  `G` is *structurally* sparse (each residual row is nonzero only on its own instrument's coupon DF
  columns), so in principle the product should cost `nnz(G)·n_knots` not `n_res·n_times·n_knots`. **But a
  naive row-AXPY scatter (`J.row(r) += s·W.row(c)` per nonzero) measured *slower* than the dense GEMM for
  ≥40 nonzeros/row** — column-major Eigen makes per-row strided access cache-hostile, and long swaps have
  many coupons (near-dense rows). So the sparse idea only wins at very low density (~1.6× at 10 nnz/row,
  a loss beyond that). A genuine win would need a *column-oriented* accumulation (walk `W` rows once,
  scatter into the J columns of every instrument that references that DF) or a real sparse `G`
  (`SparseMatrix`) times dense `W`, which Eigen can schedule cache-friendly.
- **EXPECTED:** allocation removal: negligible time but cleaner; GEMM restructuring: uncertain, *test
  before trusting* — could be 1.5–3× on the cold Jacobian **or** a regression, per the measurements.
- **RISK TO ORACLE:** allocation split is bit-identical (same ops, `.noalias()` only reorders storage).
  A sparse reformulation must reproduce the exact accumulation order the `+=` scatters rely on
  (`compiled_book.hpp:300` notes structural aliases `e_k == s_{k+1}` and `pay == e_k` that **must**
  accumulate) — get that wrong and residual/Jacobian consistency breaks. Medium risk; gate on the
  Jacobian-vs-bump test.
- **FIRST STEP:** land the `.noalias()` member-buffer split (trivially safe), re-time `calibration_bench`;
  only then prototype a `SparseMatrix<double> G` × dense `W` and compare — abandon if not clearly faster.

### 4. Streaming `W·x` GEMV — accept it is bandwidth-bound; the only safe lever is W's structure
**Safety: medium. Impact: medium (this is the dominant per-tick cost). Do NOT reach for intrinsics or float32.**

- **WHERE:** `compiled_book.hpp:85` `df_into` (`out.noalias() = W_ * x; out = (-out.array()).exp();`).
- **MECHANISM / measured reality:** `W` is dense `n_times×n_knots` (≈240 KB at 300×100), streamed from L2
  every eval; the GEMV is DRAM/L2-bandwidth bound (AVX-512==AVX2, confirmed). Compute-side tricks (SIMD,
  FMA) cannot help. The only bandwidth levers:
  - **Exploit `W`'s near-triangular structure.** For local schemes (flat front + Hermite/Tension back),
    `integral(t)=∫₀ᵗf` is cumulative, so `W`'s row for time `t` is nonzero only on knots up to `t` —
    roughly lower-triangular ⇒ ~half the bytes ⇒ ~1.5–2× on the GEMV. Caveats: it is **not** strictly
    triangular (spread curves add the base's full rows via `logdf_weight` recursion; a global
    `NaturalCubic` back end is fully dense), so this needs a per-build "is W banded/triangular?" check and
    a triangular-`W` GEMV path, falling back to dense otherwise.
  - **Rejected: store `W` as float32** (halves bandwidth). `Wx ~ O(1)`, so float32 gives ~1e-7 relative
    error in the log-DF ⇒ ~1e-7 in rates — **breaks the ~1e-10 oracle.** Not viable.
- **EXPECTED:** ~1.5–2× on the DF layer for local-scheme bundles (i.e. ~1.7 µs off a ~4.5 µs step, so
  ~1.3–1.4× per Newton step); zero for dense/global back ends.
- **RISK TO ORACLE:** none numerically (triangular GEMV computes the same sums, same order per row). Risk
  is purely correctly detecting when `W` qualifies. Medium implementation risk, low numerical risk.
- **FIRST STEP:** instrument a real bundle build to measure `W`'s actual fill fraction (`(W!=0).count() /
  W.size()`) for the shipped `flat_hermite` layout; if it is ≲55%, prototype an `Eigen::TriangularView`
  or a hand banded-GEMV and compare against the dense `noalias`.

### 5. Small per-tick allocation in the FX/MtM `refresh_curves`
**Safety: high. Impact: small (only the hybrid FX/MtM streaming path).**

- **WHERE:** `aad_block.hpp:269`,
  `disc_df_[c] = (-(disc_M_[c] * x + disc_b_[c])).array().exp();`
- **MECHANISM:** `disc_M_[c] * x` materializes a temporary `VectorXd` per cached curve **every tick** on
  the FX/MtM streaming path (the surrounding `_vs` loop otherwise allocates nothing). Add a reused
  per-curve scratch and `.noalias()` the GEMV into it, then `exp` in place — matching the allocation-free
  discipline the rest of the streaming loop already follows (`streaming.hpp:189`, `compiled_book.hpp:374`).
- **EXPECTED:** removes a handful of small heap ops per tick on the FX/MtM path; negligible on smooth
  feeds but tidy and consistent. No effect on the pure W-cache majority.
- **RISK TO ORACLE:** none — `.noalias()` into scratch is bit-identical.
- **FIRST STEP:** add `mutable std::vector<Eigen::VectorXd> disc_wx_;` sized in `build_df_cache`, and
  `disc_wx_[c].noalias() = disc_M_[c]*x; disc_df_[c] = (-(disc_wx_[c]+disc_b_[c])).array().exp();`.

### 6. (Anti-recommendation) Do not add SIMD intrinsics / inline asm to the DF `exp` or `W·x`
**Documented so the explicitly-authorized "intrinsics/asm" option is closed on evidence, not left open.**

- The user authorized considering hand-vectorized intrinsics/asm here. The measurements say **don't**, for
  the two candidate kernels: the vectorized `exp` and the `W·x` GEMV are **memory-bandwidth bound** (AVX-512
  buys nothing over AVX2), so a hand-written AVX-512 `exp` or GEMV kernel would add a maintenance burden,
  an ISA-fingerprint dispatch branch, and a scalar fallback to maintain — for ~no speedup. The third
  candidate, the frozen-Newton `M·r` matvec, is only ~0.5 µs and already Eigen-vectorized.
- The **one** place hand-tuned data-layout work could pay off is inside a pooled AAD `Dual` (item 2) — but
  that is an allocator change, not SIMD. If AVX-512 intrinsics are ever revisited, the honest target is a
  fused *strided-gather + divide* in `BundleFloatBatch::num/pv` (`compiled_book.hpp:202,216`) — but those
  loops already auto-vectorize to hardware gathers (the code comments confirm, and the `__restrict`
  pointer loops are written precisely to let the compiler do it), so the ceiling is low.

---

## The five requested audit dimensions — verdicts

1. **Virtual calls in hot loops** — *No action.* `CurveHandle` (`bundle_problem.hpp:37`,
   `OutrightHandle`/`SpreadHandle`) and the region `RegionIface` (`curve_module.hpp:30`) dispatch live
   **only** in the AAD/templated residual and at W-cache *setup*. The hot loop (streaming tick, cold LM
   iteration) is `DF=exp(-Wx)` + array math — zero virtual calls. Micro-benchmark: an opaque-`Handle*`
   virtual `discount` vs a devirtualizable `final` call is **within noise** (both dominated by the
   `lower_bound` + memory), matching the existing note at `curve_module.hpp:14`. `RegionHolder` is already
   `final`. CRTP/templating the handles would remove setup-time dispatch only, where AAD gradient
   allocation dominates anyway — not worth the loss of the runtime-composable curve.

2. **Redundant memory allocations** — *Mostly already eliminated; two small residual items.* Mapped:
   - **Per-tick frozen-Newton, pure W-cache (common case): allocates NOTHING after warm-up.** `streaming.hpp`
     reuses `x_/r_/dx_`; `df_at` recomputes into `df_` (member); `model_rates`/`residuals_vs` write into
     `out_/res_` (members); batch `num/pv/rate/annuity` return refs into per-batch scratch. Confirmed clean.
   - **Per-tick, FX/MtM hybrid path:** one temporary per cached curve in `refresh_curves` → **item 5**.
   - **Per cold-LM iteration:** the Jacobian's `(G·diagDF)` temp + `qb` + `num*/ann` vectors + result
     matrix → **item 3** (the `.noalias()` member-buffer split).
   - **Per Jacobian refresh (rare):** `HybridBundleResidual::jacobian_vs` allocates `J`/`Jc`; `set_anchor`
     forms an `n_res×n_res` inverse. Rare enough to leave.
   - **The big one:** per-operation `Dual` gradient allocation → **item 2**.

3. **Loops to unroll / vectorize** — *Already done where it helps; the rest is bandwidth-bound.* The DF
   `exp` (`array().exp()`) and the `W·x`/`M·r`/`G·W` products are Eigen-vectorized; the `BundleFloatBatch`
   gathers are `__restrict` pointer loops written to auto-vectorize into hardware gathers
   (`compiled_book.hpp:208,232,281`). Measured: exp is 2.9× over scalar and bandwidth-bound; GEMV/GEMM are
   bandwidth/compute bound at Eigen's efficiency. No un-vectorized hot loop was found that SIMD would
   rescue. The scalar `std::exp` in `curve_forward_sq/cube_integral` (moment path, AAD) is the only scalar
   transcendental loop — helped modestly by `-fno-math-errno` (item 1), not worth manual vectorization
   (it is a small correction term, `subdiv=32`, setup/analytics not the tick).

4. **Inline asm / intrinsics** — *Recommend against for the named kernels (item 6), on measured evidence.*
   The authorized candidates (vectorized `exp` for DF, the `W·x` GEMV, the `M·r` matvec) are respectively
   bandwidth-bound, bandwidth-bound, and trivially small. Portable scalar fallbacks + ISA-fingerprint
   dispatch would be pure cost. The real hot-path lever is the AAD allocator (item 2), which is not SIMD.

5. **Compiler flags** — *The concrete safe win (item 1).* Today: `-O3 -DNDEBUG -march=native`, no math/LTO
   flags. Safe to add: **`-fno-math-errno`** and **ThinLTO on `api`/`bench`**. Keep FMA contraction at
   clang default `on`. **Forbid** `-ffast-math`, `-funsafe-math-optimizations`, `-fassociative-math`,
   `-freciprocal-math` (reassociation breaks the ~1e-10 oracle **and** the internal bit-identical gates),
   and **also** `-ffinite-math-only` (the code deliberately reasons about NaN, `regions.hpp:29`). If
   `-ffp-contract=fast` is ever wanted, scope it per-file away from the bit-exact-gated kernels.

---

## Today's tension code — hot-loop impact check (branch `feat/tension-splines`)

Confirmed **no hot-loop overhead added**:

- `regions.hpp` `class Tension` is `is_linear_map = true`. All the expensive work — the `sinh/cosh`
  hyperbolic coefficient assembly (`tension_detail::p_coef/q_coef/Phi/Psi` with their Taylor-series
  small-σh branches) and the tridiagonal curvature solve `A(h,σ)z = B(h,σ)y` — happens **once in
  `Tension::build()`**, exactly like every other region's coefficient build. Because it is a linear map of
  the knot values, the W-cache builds its weight rows once at setup (`integral_weight_matrix`,
  `regularize.hpp`→`compiled.hpp:33`) and the differentiated hot loop stays `DF = exp(-Wx)` — it never
  calls `Tension::forward/integral`.
- **One honest note (setup cost, not hot loop):** `Tension::forward/integral` evaluate `Phi/Psi`, i.e.
  `sinh/cosh/exp`, *per call* — unlike the other cubics' cheap Horner polynomial eval. Since `integral()`
  is called once per registered `(curve,time)` when **building `W`**, a tension curve's **W-build (setup)
  is measurably slower** than a Hermite curve's (transcendentals vs Horner, ~5–10× on that one setup
  pass). This is a one-time per-recalibration-structure cost and is acceptable; it is **not** on the
  per-tick or per-LM-iteration path. Worth a comment if W-build latency ever shows up in cold-compose
  timing.
- `regularize.hpp` `tension_energy_operator` runs a `SelfAdjointEigenSolver` (dense eigendecomposition of
  the `nk×nk` stiffness) to factor `K = RᵀR` — an `O(nk³)` **setup** cost, paid once when constructing the
  streaming regularizer, not per tick. The streaming loop consumes the resulting `R`/`RᵀR` as the constant
  `B·x` curvature-pull term (`streaming.hpp:94`), which is one extra GEMV per Newton step — the same cost
  class as the existing smoothness regularizer, no new per-tick overhead beyond that documented term.

---

## Appendix — micro-benchmark results (Xeon W-3223, `-O3 -march=native -DEIGEN_ENABLE_AVX512`)

```
dims: n_knots=100 n_times=300 n_res=80

DF layer (per call):
    W*x GEMV                 : ~3400 ns   (AVX2 and AVX-512 identical -> bandwidth-bound on 240KB W)
    exp (Eigen array)        :  ~570 ns   (300 elems; AVX2==AVX-512 -> bandwidth-bound)
    exp (scalar std::exp)    : ~1660 ns   (2.9x slower than Eigen's vectorized exp)
frozen-Newton M*r (80x80)    :  ~510 ns

W*x bandwidth scaling (fixed n_knots=100):
    n_times=40  (31KB, L1)   :  313 ns  (7.8 ns/row)
    n_times=150 (117KB)      : 2027 ns  (13.5 ns/row)
    n_times=300 (234KB, L2)  : 3437 ns  (11.5 ns/row)
    n_times=800 (625KB)      : 6184 ns  (7.7 ns/row)     -> cost tracks bytes, not flops

cold Jacobian GEMM -((G*diagDF)*W):
    pure G*W (80x300x100), prealloc+noalias : ~82 us  (58.5 Gflop/s, ~50% of AVX-512 peak)
    code form (fresh alloc each iter)        : ~96 us  (the ~14us delta is the per-iter temporaries, item 3)
    naive sparse row-scatter, 10 nnz/row     :  1.6x FASTER than dense
    naive sparse row-scatter, 40 nnz/row     :  0.4x (SLOWER — strided col-major access)
    naive sparse row-scatter, 120 nnz/row    :  0.2x (much SLOWER)

virtual dispatch (discount() x300/call):
    virtual vs devirtualized : within noise (dominated by lower_bound + memory), per curve_module.hpp:14
```
