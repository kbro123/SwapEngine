## PART 3 — Branching under record → coarse kernels

### 3.1 Every branch on the pricing/interpolation path, classified

Paths are relative to the engine's `include/swaps/`.

#### (A) Structural — fixed per compiled structure; folds to constants at record/lowering time

Changing any of these is a structure change, so it means a recompile (the CompiledModel rule).

| where | branch | depends on |
|---|---|---|
| `curve/regions.hpp:83,90-101` Flat | `t <= t0_`, `t >= m_.back()`, `lower_bound` segment | query time vs knot times |
| `:125-130,196-200,314-318,434-438,982-986` | `lead = !in.has_predecessor` (a leading vs following region) | region order |
| `:146-174,258-288,366-396,495-525,1039-1079` | pre/post-segment `t <= xs_.front()`, `t >= xs_.back()`, `upper_bound` segment | time |
| `:216,336,1003` | `nseg >= 2` end-tangent formula | knot count |
| `:553,559,568,575` Hyman | `i == 0`, `i == N−1`, `i > 1`, `i < N−2` | node index |
| `:635,657,684-707,720-727,736,746,752-756,767,774,785,794-805` BSpline | domain clamps, de Boor span search, `den > 0` knot-multiplicity guards, `first` accumulator seeds | knot / time |
| `:836-940` Tension shape functions | `abs(x) < 0.5` series vs closed form; `x > 20` asymptotic | σ·h, σ·v (constants) |
| `curve/curve_module.hpp:141,154,207,214,220` | σ default, linearity query, `integral(t <= 0) = 0`, `zero(t <= 0)`, region `locate` | spec / time |
| `pricing/curve_handle.hpp:96,137,143,202-223` | turn window membership, outright vs spread, turns present, linear-horizon routing | time / spec |
| `pricing/cashflows.hpp:141-145` | `mtm_coupon_is_seasoned` (`reset_fx` set, FX fixed, start or reset in the past) | fixings / eval date |
| `:190-213` | weighted vs unweighted sub-periods | observation shape |
| `:226-234` | moment path (`fixing_step > 0`, `step3`, weight) | observation shape |
| `:250-255,291-311` | compounded, empty sub-periods, plain fast path | observation shape |
| `:320,344,392` | empty legs | leg shape |
| `:350-381` | accrual set, reset time, settled exchanges `e >= 0` / `s >= 0`, fixed `reset_fx`, `fx_spot_time != 0` | dates / fixings |
| `:472-482,500,509,517,526,537` | quadrature order, panel split at curve pieces, empty windows | structure |
| `calibration/problem.hpp:374,395,436,445,448` | `fx_spot_time == 0` spot roll, FX log residual, unbanded Rate association, banded vs unbanded | instrument spec |
| `pricing/compiled_book.hpp:467-627,683,729-790` | reduce layout, `sub_is_identity`, `cpn_is_plain`, `has_mtm_`, moment coupons, `rN >= 0` | batch shape (decided at `finalize`) |
| `curve/inflation.hpp:50,64-77,104` | seasonality active, phase, month index clamp | time / spec |
| `curve/parametric.hpp:42,49,70,76` | `t <= 0` | time |
| `pricing/fixings.hpp:137-187` | past vs future fixing days, compounded accumulation, all-ones weights | eval date + fixing table |

#### (B) Value-dependent min/max/abs/sign/limiter

- **The only one on the curve calibration path:** the Hyman monotonicity filter, `curve/regions.hpp:544-586`. It consists of:
  - `smin`/`smax` (`:544-545`);
  - `abs` and the sign `m/|m|` (`:555,561,583`);
  - the sign-agreement predicates `m·S > 0` (`:554,560,582`);
  - the monotone-run predicates `(S_{i−1}−S_{i−2})(S_i−S_{i−1}) > 0` and `pm·pd > 0 ∧ pm·(…) > 0` (`:569-579`);
  - the no-op test `correction != m` (`:558,564,586`).
- **Recorded guard count:** 78 for a 7-node region.
- **No floors** on the curve path. There is no zero floor on rates, no `max(0,·)` in the curve or instrument kernels, and hazard
  positivity is by reparameterisation, not a clamp. `max(V, 0)` exists only in XVA exposure, off the calibration path.

#### (C) Value-dependent formula selection

- **The only one:** the Huber band residual `band_residual` (`calibration/problem.hpp:219-224`), its double/slope twins (`:228-241`), and
  the streamer's active-set machinery over it (`calibration/streaming.hpp:530-700`: `side_of`, `breakpoint`, `apply_pins`,
  `track_bands`, `verify_pins`). There is one formula per side (below, inside, above) with slope `decay` or 1.
- The MonotoneCubic clamp is (B), not (C): both arms are the same formula family. The FX log residual has no branch.

#### (D) Iterative / implicit

**None inside a curve-calibration quote transform.** Every bundle quote is a closed form in DFs. Iterative solves elsewhere:

| where | what | calibration path? |
|---|---|---|
| `pricing/bond.hpp:90-104` | a spread-to-price Newton | a bond-fit quote, if bonds are ever fitted in yield/spread terms |
| `pricing/bond.hpp:228-234` | yield from clean price, Newton on dirty price | same |
| `vol/fx_vol_surface.hpp:110-135` | strike from delta: bisection on monotone branches around an interior maximum | FX vol-surface calibration |
| `vol/fx_black.hpp:264-276` | GK implied vol, safeguarded Newton-bisection | FX vol-surface calibration |

`calibration/bond_fit.hpp` fits on price (closed form), so no root-find is on the bond-fit path today.

#### (E) Discrete argmin

`pricing/bond_future.hpp:151-154` `select_ctd` takes the argmax of implied repo over deliverables. It is **not on the curve
calibration path**; it is on the bond-future pricing/RV verbs. Nothing on the bundle path picks a discrete argmin.

**Summary:** outside (A), the curve calibration path has exactly **one (B) site** (Hyman) and **one (C) site** (Huber bands,
already owned by the streamer). (D) and (E) exist only in adjacent products.

### 3.2 Representations compared per branch class (qualitative; measured numbers in 3.3)

| representation | value / derivative correctness | replay cost | SIMD inside a coarse kernel | ties / boundaries (hysteresis) |
|---|---|---|---|---|
| **(1) predicate / select** (both arms, mask) | exact value; derivative = the selected arm's (a one-sided subgradient at the kink, the same side AAD picks today: `smin` ties → first arg, `abs'(0) = +1`) | both arms every eval: cost = Σ arms + mask | yes: mask blend (`blendv`) across lanes/scenarios; no data-dependent control flow | none: the predicate is re-evaluated every eval, so the kink is re-decided at every iterate (the source of 2-cycles if the streamer does nothing) |
| **(2) guard-and-replay** | exact on the recorded pattern; **silently wrong after a flip until the guard is checked** (the check is part of every replay) | taken arm only, and under a fixed pattern the limiter **collapses into the linear rows** (no filter work at all) + O(#guards) compares; a flip → re-record + re-lower | the replay kernel is branch-free; the guard check is a vectorisable compare + OR | natural hysteresis point: the pattern can be HELD across iterates (a policy choice, not a numerical accident) |
| **(3) per-pattern cache** | as (2) | as (2); a flip to a seen pattern = a hash lookup (µs), a new pattern = (2)'s rebuild | as (2) | as (2); the cache makes ping-pong between two patterns cheap, which is also what hides a 2-cycle if nothing else detects it |
| **(4) implicit-function op** (D only) | exact value at convergence; derivative −F_x⁻¹F_p analytic (no iteration unrolling, no tape growth with iteration count, no dependence on the solver's path) | one converged solve (done anyway) + one small linear solve per derivative | the solve is scalar/serial; the derivative is a dense small solve | tolerance-dependent: the node is only as good as the convergence; kinks in F (e.g. bond price kinks) need (1)/(2) inside F |

Per class:
- **(A)** Fold. No representation needed; a change is a recompile.
- **(B) Hyman**:
  - (1) is correct and simple, but must avoid NaN in the discarded arm (study (a));
  - (2)/(3) are faster per eval because the whole filter folds into W under a pattern;
  - the break-even is set by the flip rate (study (b)).
- **(C) Huber bands**: keep in the streamer as today. Per row, the formula selection is (1) (a 3-way select on q) and costs nothing. The
  *active-set logic* is the streamer's, and should not be hidden inside a kernel.
- **(D)** (4), when a yield/implied quote is ever calibrated. Never unroll a Newton loop into a tape: that gives a long tape, a
  solver-path-dependent derivative, and iteration-count guards.
- **(E) CTD**: (2)/(3) with the argmin index as the guard (a discrete pattern), or (1) as a select over the candidates' values when
  the candidate count is small. The derivative is the chosen deliverable's (one-sided at a switch), the standard market convention.

### 3.4 The streamer interaction: desk_mixed's false stall IS a Hyman guard flip (measured)

**Probe.** `coarse/guardtrace.cpp`, counts only, output in `coarse/guardtrace_out.txt`.
- A `BundleProblem` subclass selects a logging engine through `residual_engine<>`, so the streamer runs unmodified. Breakeven is
  pinned at 64, as in `audit/probes/probe_dm_trace.cpp`.
- Every x the streamer evaluates is replayed through desk_mixed's tape, recorded at the committed state. The tape has 94 guards, all
  in SOFR's MonotoneCubic region; the count depends on the pattern, because nested predicates only record when their outer one holds.
- Guard outcomes are printed between the streamer's own trace lines. Ticks run q0 ↔ q_small from the committed state.

**q_small tick** (repeats identically on every q_small tick: 5 steps, 1 refresh):

```
[r eval 1: != anchor 0]
step 1 |dx|=4.69e-03
[r eval 2: != anchor 14, != previous 14]                          <- the big first step crosses 14 Hyman predicates
step 2 |dx|=1.31e-04
[r eval 3: != anchor 10, != previous 4: g86 (-6.0e-05->+2.2e-05) g88 (-1.7e-10->+4.1e-10) g89/g90 (-3.0e-05->+1.1e-05)]
kink 2-cycle: half step                                           <- the FLK2 reversal detector fires
step 3 |dx|=1.31e-04  adaptive: rho 1.000 -> REFRESH              <- compares the UNDAMPED reversing |dx| with the previous
[J eval 4: != previous 4: g86 g88 g89 g90 flip back]              <- the half step lands on the other side again
step 4 |dx|=1.01e-05 ; step 5 |dx|=4.89e-11 converged
```

**q0 tick** (repeats on every q0 tick: 10 steps, 1 refresh):

```
step 1: 6 guards flip back (g28 g61 g86 g88 g89 g90); pattern now "8 != anchor"
steps 2..8: NO guard changes, |dx| contracts linearly at rho = 0.577 (the frozen operator is the q_small side's)
refresh (the max_frozen = 8 hard cap, not the adaptive stall), step 9: the last 8 guards flip to the anchor pattern with margins 1e-12 -> 6e-18
```

**What this shows.**
1. **The 2-cycle is exactly one group of Hyman predicates** (four guards at the region's far end) flipping between consecutive
   Newton iterates. The Jacobian differs on the two sides, so each side's frozen step points back across the kink.
2. **The committed solution sits ON a kink.** On the q0 tick the final flips happen at margins of 1e-12 to 1e-18, i.e. at rounding.
   That is why the q0 walk contracts only linearly (rho 0.577): it approaches a kink point with the other side's operator.
3. **The refresh is forced by construction, not by staleness.** After the half step the adaptive stall computes
   `rho = |dx_undamped| / |dx_prev|`, and the 2-cycle test itself required those two to be the same size
   (`last_step_.squaredNorm() < 1.21·n2`). So rho ≈ 1 → REFRESH, every tick (`streaming.hpp:411-459`).
   - Two cheap fixes: exclude a damped step from the contraction history (set `dx_prev = -1` when `damp < 1`), or feed it
     `damp·|dx|`. Either removes the false refresh. The iterate path changes at rounding level, and the fixed point does not.
4. **Only 4 distinct guard patterns occur over 51 evaluations.** The oscillation is a small active set, not chaos.

**Treating interpolation kinks as an active set, like band edges.** The mask/guard bits give, per evaluation and for free:
- which predicates changed since the previous iterate (the flip set), and
- their signed margins, i.e. how far the iterate is from each kink.

The streamer already runs exactly this machinery for Huber band edges: `breakpoint` (step length to the nearest edge), `switch_row`,
the pin-on-two-flips rule (`flips_[k] >= 2`), `verify_pins` (a KKT release test). The kink version would be:

| band machinery (today) | kink machinery (proposed) |
|---|---|
| a row's side of its band | a predicate's truth value (guard bit) |
| `breakpoint`: α to the nearest band edge along dx | α to the nearest predicate zero along dx: each margin is a smooth function of x, so α ≈ −margin / (∂margin/∂x · dx). The partial is one row of the same forward/tangent pass, or a finite difference between the two iterates' margins (free: both margins are already computed) |
| `switch_row`: rescale the row's slope | switch the pattern: under (2)/(3) swap in the pattern's W rows, under (1) nothing to swap. The Jacobian rows of the instruments reading the region change, which is a low-rank update of M (those rows only), like `rescale_row` |
| pin on two flips in one tick | **HOLD the pattern within a walk:** when a predicate flips twice in one tick, pin it at its edge (a kink optimum), exactly the band rule |
| `verify_pins`: KKT release | release if the pinned side's one-sided directional derivative says the objective decreases into the other side (the same λ-sign test) |

**Would it replace the half-step heuristic?**
- **Yes, for Hyman kinks**, and more exactly. The half step is a blind bisection that detects the cycle only after it has happened
  (one wasted step + a forced refresh). A breakpoint walk stops *on* the kink in one step, knows which predicate it is, and pins it.
- The half step would stay as a generic safety net for kinks the kernel does not expose (for example a future scheme written
  without masks).

**Cost.**
- **Per evaluation:** the guard/mask compare, 78-94 compares ≈ 0.1 µs, plus keeping the previous bits.
- **Per step:** α computation over the flipped/near predicates only.
- **Per switch:** a low-rank update of M restricted to the rows reading the region: desk_mixed has 18 AAD rows on SOFR's long end,
  cheaper than the full refresh the heuristic triggers today.
- **Engineering:** the streamer's band code generalises (a "constraint" = band edge or kink predicate). Risk: rows can straddle
  several kinks; the pin/release logic must handle multiple simultaneous predicates, as it already does for multiple bands.
- **Prerequisite:** the pricing kernel must export the bits and margins. That falls out of (1) predicate/select (the mask *is* the
  bit, the predicate value *is* the margin) and of (2) (the guard check).

### 3.5 Main-line q_small variants: V1 flips real Hyman kinks, V4 only moves degenerate ties (measured)

**Probe.** `coarse/guardtrace2.cpp`, counts only, output in `coarse/run_part3b.txt`.
- `variant()` is copied verbatim from `scratchpad/dm/probe_dm_stall_qsmall.cpp`. Protocol as there: calibrate, break-even pinned 64,
  6 warm ticks alternating q0 ↔ V1.
- Every iterate goes through a **named** predicate evaluator for SOFR's MonotoneCubic region: the engine's `hyman_filter` predicate by
  predicate, giving node, knot interval, truth, margin, and whether the engine evaluates it at all. It is cross-checked against the
  recorded tape's 94 guards.
- desk_mixed's only value-dependent region is **SOFR (curve 0), knots 6..11, 12Y..30Y**, with the join at 10Y. Every other curve is linear.

**Committed-state differences.**

| from x(q0) to | knot move | Hyman predicates that change |
|---|---|---|
| **x(V1)** (5 steps, **1 refresh**/tick) | \|dx\|∞ 2.0e-4 at knot 54, **not** on SOFR | **Real kinks:** node 5 (t = 25Y, tangent over [20Y, 30Y]) **clamp `M < \|m\|` false → true**, margin −1.58e-4 → +2.54e-5; node 6 (30Y end, [25Y, 30Y]) **end sign `m·S > 0` true → false**, margin +6.33e-9 → −2.40e-10 (the end tangent is zeroed); node 2 (12Y, [10Y, 15Y]) min-switch `\|pu\| < \|pm\|` true → false, +1.99e-5 → −3.49e-5. **Degenerate ties:** nodes 3, 4, 5 monotone-run tests with margins **2e-20 / 3e-23** at x(q0) |
| **x(V4)** (2 steps, **0 refreshes**) | \|dx\|∞ 3.0e-5 at knot 10 (SOFR) | **only the same degenerate ties** at nodes 3, 4, 5 (margins 1e-19..1e-22 → −7e-10), plus predicates that stop being evaluated below them. No clamp or sign kink changes |

**Per iterate of one V1 tick** (identical on the repeat tick):

```
eval 1  x(q0)                                   (no change)
step 1  |dx| 4.69e-3
eval 2  node 2 min-switch, node 5 CLAMP on, node 6 END-SIGN off, + the node 3-5 ties     (14 tape guards change)
step 2  |dx| 1.31e-4
eval 3  node 5 clamp OFF (+6.0e-5 -> -2.2e-5), node 6 end-sign ON (-1.7e-10 -> +4.1e-10)  (4 tape guards)
        kink 2-cycle: half step ; adaptive rho = 1.000 -> REFRESH
J eval  node 5 clamp ON (-2.2e-5 -> +1.9e-5), node 6 end-sign OFF (+4.1e-10 -> -2.0e-10)   <- THE REFRESH is taken exactly here
step 4, step 5: no further change -> converged
```

**Answers.**
1. **Hypothesis confirmed.** V1's alternating per-row noise moves SOFR's 25Y/30Y tangents across two genuine Hyman kinks: the 25Y clamp
   activates, and the 30Y end tangent loses sign agreement with the last secant and is zeroed. That is a **Jacobian jump**: the rows
   reading 20Y..30Y see a different linear map on each side, which explains the frozen map's spectral radius 1.365 after a 2e-4 knot move.
2. **The flip coincides with the refresh.** The same two predicates alternate on iterates 2 → 3 → the half-step Jacobian point. The refresh
   is taken at a point on the V1 side; after it, no predicate changes and the tick converges in two steps.
3. **V4 crosses only degenerate ties.** At x(q0) the monotone-run products `(S_i − S_{i−1})(S_{i+1} − S_i)` are ~1e-20, because the
   15Y/20Y/25Y/30Y knots are equally spaced and the fixture's x_true is linear in knot index, so consecutive secants are equal.
   Crossing such a tie switches which formula computes M, but **both arms are equal at the tie**, so value and Jacobian are continuous
   and the frozen map stays contractive (radius 0.000, 2 steps). A predicate flip is not automatically a Jacobian jump; the margin *and*
   the arm gap matter.
4. **The end-sign predicate at node 6 is hypersensitive.** Its margin is ~6e-9 in m·S units; a 0.1 bp noisy tick moves it by ~6e-9. Any
   noisy q_small (V0, V1) crosses it, and the fixture change to V1 will **not** remove desk_mixed's per-tick refresh.

**Implications for the active-set idea (3.4).**
- The exported mask should carry the **arm gap** at each flipped predicate, not only the flip. For example, at the node 5 clamp the gap is
  |m| − M, and at node 6 it is the zeroed tangent |m|.
- The streamer could then ignore tie flips (gap ≈ 0: V4's case) and treat only real kinks (node 5 clamp, node 6 end sign) as active
  constraints. Both quantities are by-products of the select kernel: the predicate value and the two arm values are computed anyway.
