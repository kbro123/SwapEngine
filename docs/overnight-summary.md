# Overnight results summary

_Generated for morning review. Branch: `feat/tension-splines` (engine), `feat/tension-web` (web). Nothing was deployed._

## 1. TL;DR

- **Track A (Tension interpolation region) — COMPLETE.** New linear hyperbolic-spline region landed as `0eaa71d`; all gates green (σ→0 vs NaturalCubic at machine precision, integral vs Gauss quadrature ~1e-14, C¹/C² continuity, QuantLib OIS oracle rel ~7e-17), 194/194 tests pass.
- **Track B (tension-energy regularizer) — COMPLETE.** Constant pseudo-residual smoothness penalty landed as `9d0f76d`; drops the ill-conditioned EUR-trio normal-matrix condition number from **~1e18 (rank-deficient) to ~1.2e10** and damps null-space wander ~24x. Honest caveat: it smooths but does not fully pin the long-end null.
- **Web wiring — COMPLETE (local only).** Both tracks wired into the composer on `feat/tension-web` (`0295b4d`, `dbe9710`); Python-level verification passed. Browser UI check and `fly deploy` deliberately left for you.
- **Fast-path audit — COMPLETE, no fix needed.** Both changes are off the hot loop by construction; measured per-tick cost is unchanged (Tension-back B/A=0.998, tension-reg C/D=1.007 vs baselines). One optional cosmetic micro-opt noted, not applied.
- **Separate pipeline-overhead research — DID NOT RUN.** No `docs/calibration-overhead-analysis.md` and no `research/calibration-overhead` branch exist; that stage produced no output. See §6/§7.

## 2. Track B — tension-energy regularizer

**Status: COMPLETE. Committed, full gate green.**

Implemented the tension-energy **regularizer** (research note §5) — a smoothness penalty on the LM, **not** a new spline type.

**What shipped**
- `tension_energy_operator(prob, weight, sigma, curves)` in `include/swaps/calibration/regularize.hpp`: returns the constant pseudo-residual block `R = weight·L` where `LᵀL = K2 + σ²·K1`. K1 = ∫Φ'Φ'ᵀ (membrane), K2 = ∫Φ''Φ''ᵀ (bending) are the structure-only stiffness matrices of the curves' actual knot-forward shape functions. σ=0 ⇒ pure curvature ∫(f'')²; σ>0 adds the membrane term. Computed in closed form per knot interval, factored once via a symmetric eigendecomposition (SPSD, constant/linear null dropped — no plain-Cholesky failure). All spline algebra is one-time setup; the differentiated hot loop is untouched.
- `LinearRegularizedProblem` + `linearly_regularized()`: appends `R·x` as pseudo-residual rows (calibrate-side twin of folding `RᵀR` into the streaming operator), duck-typing the problem interface like `SmoothedProblem`.
- `RegSpec` gained `tension` (bool) + `sigma`; calibrate / risk_operator / start_streaming and the `run_json` dispatcher route to the tension operator when `tension=true`, else the existing `second_difference` path is unchanged. Default-off, opt-in.

**Gate results (all pass; `tools/verify.sh` fully green — 184 tests + perf gate at this commit)**
- `tests/tension_regularizer_test.cpp` (non-QL): affine forward ⇒ zero bending energy + exact membrane energy β²·T at machine precision; operator bending energy matches ∫(f'')² of the built curve; energy(σ)=bend+σ²·membrane exact; weight scales as weight².
- `tests/tension_regularizer_oracle_test.cpp`: on the ill-conditioned basis-only EUR trio the penalty drops the normal-matrix condition number from **~1e18 (rank-deficient) to ~1.2e10**, lifting the near-null eigenvalue to +9.6e-10.
- Same file: the regularized solve is first-order optimal (‖Jᵀr‖∞ ~ 1e-15), preserves the observable fit (data residual ~1.7e-7), and damps null-space wander **~24x (0.73 → 0.030)**.

**Honest finding.** Pure bending energy weights each interval by 1/h³, so it penalizes curvature only weakly at the widely-spaced long end — exactly where this bundle's null sits (EUR3M 30y knot). Tension therefore damps the wander an order of magnitude but does **not** drive x→x_true to machine precision the way the uniform second-difference penalty does (consistent with note §7: energy-min is a partial smoother; pinning fully closes a long-end null). Gate (c) asserts this true behaviour rather than over-claiming recovery.

**Commit:** `9d0f76d feat(reg): tension-energy regularizer (note §5) as a constant pseudo-residual block`. `ARCHITECTURE.md` + `tests/ORACLE_TESTS.md` updated in the same commit.

**Cond-number improvement (measured):** ~1e18 → ~1.2e10 on the EUR-trio normal matrix.

## 3. Track A — Tension interpolation region

**Status: COMPLETE.**

Added the `Tension` interpolation region (spline under tension, note §1/§3/§6), mirroring how the BSpline region landed.

**What shipped**
- `include/swaps/curve/regions.hpp`: new `template<class Scalar> class Tension` alongside Flat/Linear/NaturalCubic/Hermite/BSpline/MonotoneCubic, satisfying the identical contract (n_values, t_end, build, forward, integral, out, `is_linear_map=true`). f ∈ span{1, t, sinh(σt), cosh(σt)} per interval; knot curvatures z solve a tridiagonal A(h,σ)z=B(h,σ)y depending only on spacings h and σ (never on y), so z = A⁻¹B·y is a constant matrix × y ⇒ forward/integral are linear in the knot forwards. All sinh/cosh + Thomas-tridiagonal work is in `build()`; hot loop stays DF=exp(−Wx). Added a `tension_detail` namespace of numerically-stable structure functions (built from minus-linear helpers sinhm1/coshm2/xcoshm; Taylor |x|<0.5, scaled-exp asymptotics for σh>20) so σ→0 reproduces NaturalCubic exactly with no cancellation and large σ cannot overflow.
- `include/swaps/curve/curve_module.hpp`: `Scheme::Tension`, `CurveModule::sigma`, `TensionHolder`, add() case, and the `flat_tension(meeting, back, σ)` named layout. Router treats it as linear via the runtime `is_linear_map()` check — W-cache unchanged.
- `api/bundle_api.cpp`: scheme_from_str/to_str + JSON `sigma` round-trip.

**Gate results** (`tests/tension_test.cpp` + `tension_oracle_test.cpp`, both registered in CMake):
- σ→0 vs NaturalCubic: forward **7e-18** / integral **1e-16** (machine precision).
- integral(t) vs high-order Gauss quadrature: **~1e-14**.
- C¹/C² continuity via FD; linearity + W-cache exp(−Wx) reproduces discounts exactly.
- Large-σ taut limit: no NaN.
- QuantLib OIS oracle over σ∈{0.05, 1, 5, 25}: rel **~7e-17** (< 1e-10 tol).
- `tools/verify.sh` fully green: **194/194** correctness tests (12 new Tension unit tests + 1 new oracle test), oracle guard PASS, conventions sync PASS, perf gate PASS.

**Commit:** `0eaa71d feat(reg): Tension interpolation region (note §1,§3,§6) as a linear hyperbolic scheme`. `ARCHITECTURE.md` updated in the same commit.

## 4. Web wiring

**Status: COMPLETE (local only, NOT deployed). Branch: `feat/tension-web`.**

**Setup.** Pushed engine branch `feat/tension-splines` to origin (`kbro123/SwapEngine`), bumped the web's `engine/` submodule `453aeed → 0eaa71d` via `tools/sync_engine.sh`, and rebuilt the pybind11 binding.

**Track A (region):** added "Tension" to `POLICIES` in `web/engine.js` and `server/compile.py`. `renderRegions` shows a per-region sigma input only when the region policy is Tension (default σ=1 on switch, coerced non-negative). `compile.py`'s `_curve_layout` emits `sigma` on the region module only for the Tension scheme. The Flat→Hermite classic fast-path check is unaffected.

**Track B (penalty):** added an opt-in "Tension-energy penalty" checkbox + sigma input beside the Forward-smoothing control (`spec.tension_reg` / `spec.tension_sigma`). `compile_spec` carries them through; `reg_spec()` appends `tension:true` + `sigma`. The binding's `reg_from()` (`binding/swapengine_py.cpp`) now forwards `tension` and `sigma` to `api::RegSpec` (previously dropped).

**Verification (local Python-level; browser not possible autonomously):** a scratchpad script drove `server.compile` + `swapengine.Session` on the `sofr_ff_prime` seed — (1) Tension back region σ=3 calibrates rms 2.15e-16 and streams via the frozen-Newton fast path; (2) Tension σ=0 reproduces NaturalCubic (rms 1.99e-16); (3) tension-energy reg (σ=0.5) verified to carry tension+sigma, solve first-order optimal (stationarity 1.43e-13). Existing server tests pass (test_calendars 53 assertions, test_conventions_web 39). xccy 5-curve example still compiles.

**Commits:** `0295b4d chore(engine): bump submodule to tension branch (0eaa71d)`, `dbe9710 web: wire Tension region + tension-energy regularizer into the composer`.

**What a human should check**
- In the browser: the per-region sigma input renders when a region is set to Tension; the Tension-energy checkbox + sigma row toggles correctly; calibrate + stream a Tension bundle in-app.
- **Weight calibration:** the tension-energy penalty's useful λ range is orders of magnitude smaller than the second-difference operator's (K2 is much larger in magnitude). `reg_spec` currently reuses the same off/light/strong = 0/0.5/5.0 presets for both — consider a per-penalty preset table or auto-scaling to a target penalty magnitude so "light" means the same visual smoothness in both modes.

## 5. Fast-path audit

**Status: COMPLETE. No engine code change warranted — nothing committed, working tree clean, gate green.**

**Conclusion:** neither Track A nor Track B introduced any redundant allocation or loop on the fast pipeline; both are off the hot path by construction.

**Structural evidence**
1. Track A touched only regions.hpp, curve_module.hpp, api, tests; Track B touched only regularize.hpp, api, tests. None of the hot-path residual-engine files (streaming.hpp, aad_block.hpp, hybrid_residual.hpp, compiled_residual.hpp, compiled_bundle.hpp) were modified.
2. Tension sets `is_linear_map=true` → routes through `integral_weight_matrix`, built once per curve at `CompiledCurveSet::finalize()`. All sinh/cosh + Thomas-tridiagonal work is in `Tension::build()`; no such work leaked into the per-tick path.
3. `tension_energy_operator` returns a constant R (eigen-factored once at setup) plugged into the same streaming slot the pre-existing `second_difference_operator` used: `RtR_` formed once in ctor, `B_` once per refresh; per-step application is a single `noalias B_*x` GEMV — no new per-tick code or heap alloc.
4. The frozen-Newton hot loop (`streaming.hpp update_exact`) is allocation-free; the only Eigen temporaries live in `jacobian_vs`, which runs on refresh only, off the per-tick path.

**Before/after tick timings** (`scratchpad/tension_tick_repro.cpp`, 4000 correlated ticks, 23 knots/23 instruments, median ns/tick):

| Case | ns/tick | ratio |
|------|---------|-------|
| A — Hermite-back, no reg | 10225 | — |
| B — Tension-back, no reg | 10206 | B/A = 0.998 |
| C — Tension-back + tension reg | 10816 | C/D = 1.007 |
| D — Hermite-back + 2nd-diff reg | 10739 | — |

Interpretation: the Tension region and the tension regularizer add no per-tick cost beyond the identical W-cache / existing-regularizer hot path they ride.

Perf gate (baseline → current, ns): curve_build 812,742 → 657,274; risk_full_jacobian 825,542 → 701,769; portfolio_analytics 97,838 → 101,756; warm_recalibration 20,578 → 21,933 — all PASS.

**Optional micro-opt (not applied):** `streaming.hpp update_exact` does `r_ = engine_.residuals_vs(x, q_new); dx_.noalias() = M_ * r_;` — the assignment into member `r_` is a same-size vector copy (reused scratch, no heap alloc) that could be fused to `dx_.noalias() = M_ * engine_.residuals_vs(x, q_new);` to drop an O(n_res) memcpy per Newton step. Sub-1% on a ~23-element vector; cosmetic. Reproducer preserved at `scratchpad/tension_tick_repro.cpp`.

## 6. Separate pipeline-overhead research

**Status: DID NOT RUN — no output produced.**

The task expected a research deliverable at `docs/calibration-overhead-analysis.md` on branch `research/calibration-overhead` with top-ranked recommendations. **Neither exists.** The research stage's input was `null`, and I confirmed on disk:

- `docs/calibration-overhead-analysis.md` — **not present** (only `bezier-and-moments.md`, `generic-instrument-pipeline.md`, `tension-spline-research.md` are in `docs/`).
- branch `research/calibration-overhead` — **not present** (no branch matching `overhead`).

There are therefore **no recommendations to report**. This item is carried into §7 as not-completed.

## 7. What did NOT complete + exact next steps

1. **Pipeline-overhead research (§6) — not started.** No doc, no branch, no findings.
   - Next: run the research stage explicitly, targeting `docs/calibration-overhead-analysis.md` on a fresh `research/calibration-overhead` branch; produce ranked recommendations on where calibration wall-clock is spent (setup vs solve vs binding marshalling).
2. **Browser UI verification of the web wiring — not done** (requires interactive auth).
   - Next: open the composer, set a region to Tension (confirm the σ input appears), toggle the Tension-energy penalty, then calibrate + stream a Tension bundle and confirm the frozen-Newton fast path in the UI.
3. **Tension-energy weight/preset calibration — not done.**
   - Next: add a per-penalty preset table (or auto-scale weight to a target penalty magnitude) so the web light/strong presets map to tension weights ~1–2 orders of magnitude smaller than the second-difference presets.
4. **Nothing merged; no PR opened.** Engine branch `feat/tension-splines` is pushed but not merged; web branch `feat/tension-web` is local only.
5. **Optional engine follow-ups (not required):** expose `flat_tension`/σ region choice more prominently in the composer; unify the σ→0 tension operator with the Tension region behind one series-fallback path (note §6); adaptive/variable-σ tension (note §4) remains future work.

## 8. Deployment status — READ THIS

**Nothing was deployed to fly.** The live site (`swapengineweb.fly.dev`) is unchanged and still runs the pre-tension build. This was deliberate — awaiting your review.

**Branches to review**
- Engine: `feat/tension-splines` (pushed to `kbro123/SwapEngine`), commits `9d0f76d` (Track B) and `0eaa71d` (Track A).
- Web: `feat/tension-web` (local only), commits `0295b4d` (submodule bump) and `dbe9710` (composer wiring).

**Recommended deploy/verify sequence (only after review)**
1. Review the two engine commits and the two web commits; open PRs / merge as you see fit.
2. Ensure the web `engine/` submodule is checked out at the pushed sha `0eaa71d` (`git submodule update --init` on a fresh clone) — an uninitialised submodule means an empty `engine/` and a build failure.
3. From the web repo: `fly deploy`.
4. Confirm: `curl -s https://swapengineweb.fly.dev/healthz` (reports the asset version).
5. Smoke-test a Tension bundle in the live composer (region σ input + tension-energy penalty).

_Note: the engine branch is pushed but NOT merged; the web submodule already points at a pushed sha, so a deploy build will resolve it._
