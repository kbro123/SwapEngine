# SwapEngine Unified Hot-Path Core — Design Blueprint

Build spec for the vol/XVA performance session. One compiled evaluation kernel that vol pricing, greeks,
cube reprice, MC exposure, and XVA all ride. File:line anchors are verified insertion points. Pairs with
`VOL_XVA_ROADMAP.md` (feature roadmap) and `ARCHITECTURE.md`.

## 1. The unified core

One templated kernel — `Scalar=double` prices, `Scalar=Dual` yields d(pv)/d(inputs) from identical source,
`Scalar=Tape` (later) flips to reverse mode for XVA with no code change:

```cpp
template<class Scalar>
void reprice(const CompiledUniverse& U,   // immutable, built ONCE
             const State<Scalar>& s,       // {x (knots), v (sabr/vol params)} — the ONLY per-cell input
             Out<Scalar>& out);            // pv, and (if Scalar=Dual) gradients fall out
```

**`CompiledUniverse` (SoA, built once, const, shared):** `union_times` (sorted/deduped union of every
reset/pay/expiry); **`W_all`** (rows=union_times × cols=knots; `DF = exp(-W_all·x)`, derived once by one AAD
pass over a LINEAR curve — NEVER re-derived per path/tick/greek); integer topology `subS/subE/payIdx`,
`R_cpn/R_sub` gather-reduce; yield-independent scalars `tau/konst/accrued`; vol-cell topology
`cellStartIdx/cellPayIdx/cellExpiry/cellStrike` and **precomputed `sqrtT`**.

**Per-cell hot path (everything else is algebra):**
1. `DF = exp(-W_all · x)` — one GEMV + one vectorized exp
2. `fwd = DF[subS]/DF[subE] - 1` — fixed gather
3. `pv = R_cpn · (DF[payIdx] · fwd)` — sparse reduce
4. (vol cell) `NodeCtx = sabr_node(F,T,sp)` once per node
5. (vol cell) `BachCtx = {d, pdf(d), cdf(d), stddev, sqrtT}` — ONE sqrt/exp/erfc per point
6. `{price,vega,delta,gamma,vanna,volga} = algebra(BachCtx)` — zero further transcendentals

Non-linear schemes (`is_linear_map()==false`) take the `BundleCurveSet` update-in-place fallback, not the
matvec. Every linear scheme (Flat/Linear/Hermite/BSpline/Tension) gets the `W_all` fast path.

## 2. Repeated-calc elimination map (R1–R12)

| # | Redundancy | Fix |
|---|---|---|
| R1 | 6-greek Bachelier transcendental fan (6√+5exp+2erfc/pt) recomputed every tick (`bachelier.hpp:27-97`) | `BachCtx` once/pt → all 6 greeks algebraic; √T from precomputed `sqrtT`. ~4–5× fewer transcendentals |
| R2 | `price_vol_cube` per-tick `std::map`+string keys (`options.cpp:242-244,292`) | route cube reprice through `VolSurface` SoA; re-point `price_vol_cube_json` |
| R3 | DF re-integrated per cashflow via region-walk (`bundle_api.cpp:606-609`) | `DF=exp(-W_all·x)` one matvec; pricing+exposure share the sample |
| R4 | J rebuilt 2–3× on same `x` (`generate_risk.cpp:105-107`) | memoize J on session keyed by `x_` (J is reg-independent) |
| R5 | book repriced as Dual 2–3× for NPV/PV01/ladder (`bundle_api.cpp:609,619-624,655-663`) | value+all sensitivities in ONE sweep |
| R6 | ladder materializes full M via explicit inverse (`bundle_api.cpp:593-594`) | lean IFT: `a=JtJ.ldlt().solve(dnpv_dx); ladder=J*a` (`risk.hpp:37-38`) |
| R7 | CMS Simpson `pow` bloat + `std::function` (`cms_replication.hpp:26-76`) | cache `a,aN,aD` per node; template `vol_at` to inline SABR |
| R8 | SABR vol recomputed per strike; ATM via full smile (`options.cpp:115`) | `sabr_node()` memoizes F/T part; closed-form ATM |
| R9 | `strip_caplet_vols` O(n²) (`cap_stripping.hpp:87-95`) | carry prefix PV as running sum → O(n) |
| R10 | `sabr_calibrate` evals strip 2× / iter (`sabr_calibration.hpp:57,72,86`) | recover RMS from AAD residual pass |
| R11 | per-op Dual gradient heap-allocs (`aad_block.hpp:16-18`) | pooled fixed-width Dual → allocation-FREE (also what the reverse tape wants) |
| R12 | exposure would re-derive `W_all` per path | derive once; whole grid = one `W_all·X` matmul; node-aging via live-flow union / zero-amount masking |

## 3. Ranked performance backlog

Metric legend: **SRL** surface-recal latency · **FBR** full-book reval throughput · **ESW** exposure-sim
wall-clock · **AGS** AAD greeks/sec.

| Rank | Item | Metric | Est. | Effort | Proving benchmark |
|---|---|---|---|---|---|
| 1 | R11 pooled fixed-width Dual (enabler for reverse tape) | AGS | 3–10× greeks-heavy | L | `bench_dual_book_greeks` |
| 2 | R1 Bachelier BachCtx fan collapse (streaming vol path) | SRL | 4–5× fewer transc. | S | `bench_vol_cube_reprice` |
| 3 | R3+R12 W-cache reprice AS the exposure kernel | FBR+ESW | O(N·M·cf)→O(1) setup | M | `bench_book_reprice`, `bench_exposure_grid` |
| 4 | R5 value+PV01+ladder one Dual sweep | FBR+AGS | 2–3× risk verb | S | `bench_generate_risk` |
| 5 | R4 memoize J on session | FBR | drop 2/3 J builds | S | `bench_risk_operator` |
| 6 | R2 kill cube map/string churn | SRL | N string + 2N map/tick | S | `bench_cube_json_seam` |
| 7 | R7 CMS Simpson pow-cache + template vol_at | FBR(CMS) | >2× per rate | M | `bench_cms_replication` |
| 8 | R6 lean IFT ladder | AGS | vec solve vs inverse | S | `bench_bucketed_delta` |
| 9 | R9 caplet strip O(n²)→O(n) | SRL | ~n× | M | `bench_caplet_strip` |
| 10 | R10/R8 SABR calib single-eval + closed ATM | SRL | ~m evals/iter | S | `bench_sabr_calibrate` |

Top 3 convert "overnight batch → interactive," which the roadmap foregrounds as THE differentiator.

## 4. Build + benchmark order (each stage lands a native C++ google-benchmark — NEVER through JSON)

- **Stage 0 — Core + micro-benchmarks:** `CompiledUniverse` SoA + templated `reprice<Scalar>`; land R1, R2, R4.
  `bench_book_reprice` **< 1 µs / reprice** (100-instrument single-curve book off x); `bench_vol_cube_reprice`
  **< 50 µs** (20×20 swaption cube on a vol tick).
- **Stage 1 — Live vol surface (Phase A, no MC):** VolSurface as the only cube path; SABR calib (R10), cap
  strip (R9), CMS (R7) on the core. `bench_vol_surface_stream` **< 100 µs / vol-slider tick (> 10k ticks/s)**;
  `bench_sabr_calibrate` **< 10 µs / smile**.
- **Stage 2 — Pooled Dual + one-sweep risk:** R11, R5, R6. `bench_dual_book_greeks` **full delta ladder at ≤ 4×
  base valuation, alloc-free**; `bench_generate_risk` **> 1000 bundles/s**.
- **Stage 3 — MC exposure profiles:** evolve `x(t,path)` under LGM/HW1F; `reprice(double)` per node; ThreadPool
  + SoA + Sobol/Brownian-bridge; R12. `bench_exposure_grid` **10k paths × 100 nodes EPE/ENE/PFE in < 1 s
  (> 1M node-reprices/s)**; `bench_collateral_transform` < 5% overhead.
- **Stage 4 — Bermudan / AMC exposure:** LSM early-exercise; analytic-European control variate + Andersen–
  Broadie dual as validation. `bench_amc_exposure` **< 2× European-exposure wall-clock**.
- **Stage 5 — XVA + reverse-mode tape (capstone):** flip Scalar to the reverse tape; CVA/DVA off a hazard
  ModularCurve; netting-set-native; FVA; SIMM; MVA. `bench_xva_greeks` **full XVA sensitivity vector (thousands
  of Greeks) in ≈ 4× one XVA valuation** (vs overnight bump-and-run); `bench_cva` **< 2× the EPE-profile cost**.

**Cross-cutting rules:** never re-derive `W_all` (setup-only); never allocate in the per-cell/per-path loop
(reuse SoA scratch + pooled Dual); one templated kernel (Scalar is the only axis); benchmarks are native C++;
non-linear-scheme books take the `BundleCurveSet` fallback (gate on `is_linear_map()`).
