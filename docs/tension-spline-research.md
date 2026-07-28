# Tension Splines — research notes

**Status:** pure research, no code yet. Branch `claude/tension-spline-research-v8eyc2`.
**Scope:** (1) first-principles math of tension splines, (2) the lowest-energy proof, (3) how to
implement so it stays OFF the AAD hot path, (4) whether minimum energy suppresses arbitrageable
forward artifacts. Ties throughout to the engine's linear-map / W-cache architecture (CLAUDE.md §2).

---

## 0. TL;DR / decisions to carry forward

- A **fixed-tension** tension spline is a **LINEAR MAP of the knot forwards** → `is_linear_map = true`
  → it rides the existing W-cache and analytic Jacobian exactly like `NaturalCubic` / `BSpline`.
  **AAD touches it exactly once** (building `W`); the differentiated hot loop never sees a `sinh`.
  So "slow on the AAD path" is a non-issue *if we keep σ a fixed hyperparameter*.
- Implementation shape = a new `Tension` region in `curve/regions.hpp` + a `flat_tension(meeting, back, σ)`
  layout. Additive, `BSpline`-style. Inherits calibration/risk/streaming/bundle with no hot-path change.
- The killer argument vs `MonotoneCubic`: tension gives overshoot control **without data-dependent
  branches**, so it stays on the fast path that Monotone had to leave (Hyman's filter = value-dependent
  `min/max/sign` → `is_linear_map = false`).
- **Adaptive/variable tension** (σ chosen from the data to guarantee monotonicity) goes value-dependent
  → AAD tier. Keep it fast with a **frozen-tension** scheme (freeze σ within an LM iteration, refresh
  between), the direct analogue of frozen-Jacobian streaming.
- Best synthesis: carry tension energy as an **analytic quadratic regularizer** on the over-determined
  LM — a *constant* Jacobian block that also fixes the cond≈636 ill-conditioning flagged in Phase 3.
- On arbitrage: minimum energy is only a **partial** defense (good noncommittal prior in unpinned
  regions; L2-average control of the butterfly-detectable quantity). It does NOT prevent overshoot,
  negative forwards, or phantom-hedge leakage. What actually closes the pick-off is **tension +
  positivity constraint + locality + pinning**, not a smaller energy number.

---

## 1. What a tension spline is (variational first principles)

Natural cubic spline minimizes **bending energy** over `C²` interpolants:

    J₀[f] = ∫ (f″)² dt        s.t. f(tᵢ)=yᵢ

Spline under tension (Schweikert 1966, Cline 1974) adds a first-derivative ("membrane") penalty:

    Jσ[f] = ∫ [ (f″)² + σ² (f′)² ] dt

Euler–Lagrange for a functional in (f, f′, f″):

    ∂F/∂f − d/dt(∂F/∂f′) + d²/dt²(∂F/∂f″) = 0
      0    −   (−2σ² f″)   +     (2 f⁗)     = 0
    ⇒  f⁗ − σ² f″ = 0   on each knot interval

So `g = f″` solves `g″ = σ²g` → `g = A cosh(σt) + B sinh(σt)`, and on each interval

    f ∈ span{ 1, t, sinh(σt), cosh(σt) }      (hyperbolic instead of cubic)

Limits:
- **σ → 0:** sinh/cosh Taylor-expand to {1,t,t²,t³} → recovers the natural cubic. (Numerically switch to
  the series near 0; see §6.)
- **σ → ∞:** hyperbolics become boundary layers → **piecewise linear**; tension pulls the curve taut,
  kills overshoot.

σ is a continuous knob: smooth-but-overshoots (cubic) ↔ taut-no-overshoot (linear).

---

## 2. The lowest-energy proof (what was asked for)

**Holladay's theorem (1957).** Among all `C²` interpolants of `(tᵢ,yᵢ)`, the *natural* cubic spline `s`
uniquely minimizes `∫(f″)²`.

*Proof (orthogonality — this is the part that generalizes).* Let `f` be any `C²` interpolant, write
`f = s + e` with `e(tᵢ)=0`. Then

    ∫(f″)² = ∫(s″)² + 2∫ s″e″ + ∫(e″)²

Cross term, integrate by parts twice:

    ∫ s″e″ = [s″e′]ₐᵇ − ∫ s‴e′

Natural BC `s″(a)=s″(b)=0` kills the boundary term. On each interval `s‴ = cᵢ` (const, cubic), so

    −∫ s‴e′ = −Σ cᵢ (e(tᵢ₊₁) − e(tᵢ)) = 0     (e vanishes at knots)

Hence `∫(f″)² = ∫(s″)² + ∫(e″)² ≥ ∫(s″)²`, equality iff `e″≡0` ⇒ `e` linear & zero at knots ⇒ `e≡0`.
Existence + uniqueness. ∎

**Tension case, for free.** The EL operator `L = D⁴ − σ²D²` is **self-adjoint**. The tension spline is
the interpolant with `Lf=0` between knots + natural BCs `f″(a)=f″(b)=0`. The identical `f=s+e`
expansion on `Jσ` gives cross term `2∫[s″e″ + σ²s′e′]`, which collapses via `Lf=0` and the natural BCs
to `Σ cᵢ(e(tᵢ₊₁)−e(tᵢ)) = 0`. So `Jσ[f] = Jσ[s] + Jσ[e] ≥ Jσ[s]` — **the tension spline is the unique
minimum-tension-energy interpolant.** Same theorem, one operator up.

**General family (L-splines; de Boor–Lynch 1966, Schultz–Varga 1967).** For any energy
`∫ Σₖ μₖ(f⁽ᵏ⁾)²` the minimizer is the L-spline solving `Lf=0`, `L = Σ(−1)ᵏμₖ D²ᵏ`, with natural BCs
`μₖ f⁽ᵏ⁾ = 0` at the ends:
- `∫(f″)²`            → cubic
- `∫(f″)²+σ²(f′)²`     → tension / hyperbolic
- `∫(f‴)²`            → quintic

"Lowest energy for a given construction" = the natural spline of the matching order.

**Curve-construction-specific result — cite alongside Holladay.** Adams & van Deventer (1994),
*Fitting Yield Curves and Forward Rate Curves with Maximum Smoothness*: minimize `∫₀ᵀ(f″)²` over the
*forward* curve subject to **discount-bond repricing (integral) constraints**. Because the constraints
are integral (not point values on `f`), the EL problem raises the order → the max-smoothness forward
curve is a **quartic** spline. In our engine we chose knot forwards as the free variables (point
interpolation of `f`), so we're in the Holladay/L-spline regime → cubic/tension is the minimizer here.

---

## 3. Why fixed tension stays a LINEAR MAP (the architecture fit)

Engine thesis (CLAUDE.md §2): `integral(t) = ∫₀ᵗ f = w(t)·x`, linear in the knot forwards, weights
depend only on knot *times*. That gives the W-cache, analytic `J = −(∂r/∂DF·diag(DF))·W`, warm/stream.
Qualifying schemes: Flat, Linear, NaturalCubic, Hermite, BSpline. `MonotoneCubic` does NOT (Hyman filter
= data-dependent branches).

**Fixed-tension qualifies.** With σ a fixed number (or a per-interval array `σᵢ` chosen from geometry),
assemble the spline like the natural cubic: solve a **tridiagonal system** for knot curvatures
`z = (f″(tᵢ))`:

    A(h, σ) z = B(h, σ) y

`A`, `B` depend only on spacings `hᵢ = tᵢ₊₁−tᵢ` and σ (through `sinh(σhᵢ)`, `cosh`, `coth`, `csch`),
**never on the values `y`**. Standard rows (up to normalization):

    A_{i,i−1} = 1/h_{i−1} − σ/sinh(σh_{i−1})
    A_{i,i}   = σ coth(σh_{i−1}) + σ coth(σhᵢ) − 1/h_{i−1} − 1/hᵢ
    A_{i,i+1} = 1/hᵢ − σ/sinh(σhᵢ)
    B_i       = usual divided-difference of y   (linear in y)

So `z = A⁻¹B y` = **constant matrix × y**, and on each interval

    f(t) = [linear in yᵢ, yᵢ₊₁] + [ zᵢ sinh(σ(tᵢ₊₁−t)) + zᵢ₊₁ sinh(σ(t−tᵢ)) ] / (σ² sinh(σhᵢ))

is linear in (y, z) hence **linear in y**. `∫₀ᵗ f` is linear too (`∫sinh = cosh/σ`, elementary).

Therefore:
- `is_linear_map = true`; `integral(t) = w(t)·x` with `w(t)` structure-only → **W-cache applies**,
  analytic Jacobian unchanged.
- **AAD touches it exactly once:** `integral_weight_matrix` extracts `W` in one forward-mode pass;
  because the map is linear, that gradient is exact & constant — one sweep, never again. All the
  `sinh/cosh` + tridiagonal work is **setup cost, off the hot path**. Hot path stays `DF = exp(−Wx)`.
  **No `sinh` in the differentiated hot loop.**

**Implementation = additive `Tension` region** (mirror how `BSpline` landed):
- ctor takes knots + σ (or `σᵢ`), builds & factorizes `A`, precomputes `Z = A⁻¹B`.
- `forward(t)`, `integral(t)`: evaluate hyperbolic shape functions, linear in `x`.
- `is_linear_map = true`.
- add `flat_tension(meeting, back, σ)` named layout; rides calibration/risk/streaming/bundle unchanged.

**Caveat — locality.** Global-tension spline is **non-local** (dense `W` ⇒ non-local deltas), like
natural cubic. If hedging locality matters, use a **fixed-tension Hermite** variant (tension on a local
`C¹` Hermite basis, structure-only tangents) — still linear, still fast, local deltas, at the cost of
`C¹` instead of `C²`. Same locality/continuity axis already noted between natural-cubic and Hermite.

---

## 4. When it goes data-dependent — and how to keep it fast

A single global σ is a smoothness knob but does **not guarantee** monotonicity/positivity. To *guarantee*
no overshoot you need **variable tension `σᵢ(y)`** (Schumaker 1983, Rentrop 1980): crank σ locally where
the cubic would overshoot. That makes coefficients value-dependent → `is_linear_map = false` → routes to
`AadResidualEngine`, no W-cache. Existing guard catches this automatically (the `static_assert` in
`integral_weight_matrix`, `residual_engine_t` routing, runtime `is_linear_map()`).

**Frozen-tension** keeps even adaptive tension cheap (analogue of frozen-Jacobian Newton):
- Within an LM iteration, hold `σᵢ` fixed at the value implied by the current `x` → residual is
  linear-map for that iteration → full analytic W/J, microsecond step.
- Recompute `σᵢ(x)` only between iterations (or when a monotonicity violation is detected) — same
  "recompute on genuine staleness" pattern as streaming.
- Error vs true joint differentiation is the `∂σ/∂x` coupling, which is 2nd-order near a monotone
  solution (σ usually sits at a corner/plateau of its selection rule ⇒ `∂σ/∂x ≈ 0`). Semismooth-Newton
  argument; exact AAD path is always the fallback.

---

## 5. The synthesis to actually build: tension as an analytic regularizer

Phase-3 note: calibration Jacobian ill-conditioned (cond≈636), "smoothness/Tikhonov regulariser is the
eventual fix." Tension energy **is** that regularizer and drops in analytically.

Energy is a **quadratic form in the knot forwards.** With `f(t) = Φ(t)·x` (structure-only shape fns):

    ∫(f″)² dt = xᵀ K₂ x ,   ∫(f′)² dt = xᵀ K₁ x       (K₁ = ∫Φ′Φ′ᵀ, K₂ = ∫Φ″Φ″ᵀ, structure-only)
    tension energy = xᵀ (K₂ + σ²K₁) x

Add `μ` × this to the LM objective:

    minₓ  ‖r(x)‖²  +  μ · xᵀ(K₂ + σ²K₁) x

Consequences:
1. **Just extra pseudo-residuals.** Factor `K₂+σ²K₁ = LᵀL` (Cholesky, once); append `√μ·L x` as extra
   residual rows. Extra Jacobian block = the **constant matrix `√μ L`** — no AAD, no per-iteration cost.
   Normal-equation matrix becomes `JᵀJ + μ(K₂+σ²K₁)` (SPD, better-conditioned) → kills cond≈636 and the
   weakly-identified knot direction directly.
2. **Makes "lowest energy" literal for the over-determined fit.** We don't interpolate — we LS-fit an
   over-determined market (min‖r‖², non-zero residual by design). Adding tension energy makes the
   calibrated curve the **minimum-tension-energy curve consistent (in LS) with the market** — a
   *smoothing tension spline*. μ→0 = current fit; μ→∞ = taut/linear; between = controlled trade of
   repricing residual for forward roughness. Principled way to damp the join and the long-end wiggle.

Same object `xᵀ(K₂+σ²K₁)x` is (a) the interpolation scheme if enforced exactly via EL, or (b) the
regularizer if added softly. Both analytic, both off the AAD hot path, both structure-only.

---

## 6. Numerical stability (confined to the once-only W build)

- **σh large:** `sinh(σh)` overflows; `sinh/sinh` ratios lose precision. Use scaled forms
  `sinh(σs)/sinh(σh)`, and `coth`, `csch` directly; for `σh ≳ 20` switch to exponential/linear
  asymptotics (`coth→1`, `csch→0`) — that interval is effectively taut anyway.
- **σ→0 / σh small:** naive hyperbolic form is `0/0` and cancels catastrophically. Switch to the Taylor
  **series** in σh below ≈1e-3; leading terms reproduce natural-cubic coefficients exactly ⇒ σ→0 limit is
  continuous. Could even unify `Tension` + `NaturalCubic` behind one region with a series fallback.
- Neither touches the differentiated hot path ⇒ no AAD accuracy/speed impact; only the one-time build of
  `A`, `B`, `W`.
- **Gate** like the others: composite QuantLib `YieldTermStructure` oracle on a hand-specified `x`
  (§3(a) workhorse) to 1e-10; a σ→0 test asserting bit-agreement with `NaturalCubic`; a first-order
  optimality calibration test.

---

## 7. Does minimum energy suppress arbitrageable forward artifacts? (partial — and naive reading is backwards)

**What the HF trades.** A spurious local hump in the instantaneous forward = a locally mean-reverting
forward, and it's the 2nd difference of the curve → a **butterfly**:
- calendar spread isolates `f′` (spurious slope/kink)
- butterfly (long wings, short belly) isolates `f″` (spurious curvature hump)

So "minimize `∫(f″)²`" = minimize (in L2 average) the exact quantity a butterfly trades → obvious partial
yes. But it's an *average* control, not *pointwise* — that's where it leaks.

**Two arbitrage types, different defenses:**
1. **Static/replication arb (hard):** needs `f(t)<0` somewhere (DF non-monotone). Genuine free lunch.
   **Energy min does NOT prevent it** — a low-energy cubic dips negative in overshoot.
2. **Mark-to-model pick-off (soft):** off-strip quote (5y7y FRA, off-IMM date) deviates from replication
   with your on-strip instruments; HF locks the difference. This is the "mean-reverting artifact" case.

**The genuine YES — minimum curvature as maximally-noncommittal prior.** In a region with **no
calibration instrument spanning it**, any curvature is curvature the market never asked for, and every
unforced wiggle is a free option written to whoever can replicate around it. Minimum-curvature = asserts
no unsupported view = least pick-off-able (a butterfly struck there has ~0 model PV). Legit argument FOR
the energy objective. Qualifiers: (a) holds cleanly only when **not forced through knots** — i.e. a
*smoothing* fit, which our over-determined LS already is, and the §5 regularizer is exactly what relaxes
unpinned regions to the noncommittal curve; most spurious humps are overfitting, not signal. (b) only
where the true curve *is* smooth (see FOMC below).

**The NO — three failures of energy min:**
- (a) **Min-bending-energy *interpolant* is the canonical overshooter.** Holladay's minimizer = natural
  cubic = exactly the scheme Hagan & West (*Interpolation Methods for Curve Construction*, 2006 — THE
  reference for this question) show gives oscillatory, non-local, sometimes-negative forwards. Forcing a
  global-`C²` curve through knots near a fast-changing region redistributes curvature into **overshoots
  adjacent to the feature** and the global coupling **rings**. Low global energy, local humps.
- (b) **No positivity.** Nothing in `∫[(f″)²+σ²(f′)²]` keeps `f≥0`. The hard static arb survives energy
  min. Needs a *constraint*, not a smoothness penalty.
- (c) **Silent on locality.** Equal-energy curves can have wildly different delta structure. Non-local
  interpolation ⇒ a trade at `T` carries phantom hedge sensitivity to far instruments ⇒ dynamic pick-off
  when they move. Energy says nothing; locality (Hermite / high-tension / B-spline) does.

**Where energy min is actively WRONG — our own front end.** True forward is expected to **jump at FOMC
meetings** — the discontinuity is *information*. Minimizing energy across a meeting smooths the jump away
and *that* curve is the arbitrageable one (misprices a FRA straddling the meeting). Our piecewise-flat
front with level-only continuity at the join is the anti-artifact choice precisely because it **refuses**
to minimize energy where the real curve is discontinuous. Operative principle: **match the true
information structure; minimize energy only in the residual freedom that leaves.** Meetings discontinuous,
between/back smooth.

**What actually defeats the sharp HF (ranked by what closes real arb):**
1. **Positivity / DF-monotonicity (`f≥0`)** — closes static calendar arb. A *constraint*, not energy.
   B-spline convex-hull/positivity route already available.
2. **Shape preservation (monotone/convex)** — kills spurious extrema (the butterfly-pickable humps).
   What adaptive tension buys as σ grows; guaranteed by a monotone-convex method (Hagan–West).
3. **Locality of deltas** — closes phantom-hedge pick-off. Tension (σ→∞) and local bases give it.
4. **Pinning** — add calibration instruments so humps aren't unpinned. No interpolation defends a region
   the market may disagree with you about.

Tension is a continuous knob along 2 and 3. So: **it's not the energy *minimum* that suppresses the
artifact — it's the *tension* (+ positivity constraint) — and the energy minimum is the smoothest curve
consistent with that tautness.**

**Clean synthesis — constrained variational problem (both smooth AND arb-free):**

    minₓ  ‖r(x)‖²  +  μ·xᵀ(K₂+σ²K₁)x            (fit + tension energy)
    s.t.  f(t) = Φ(t)x ≥ 0    ∀t                 (positivity — closes static arb)
          [+ optional monotone/convex where warranted]

- energy term → unpinned-region noncommitment (the genuine yes)
- σ tension → damps ring, localizes deltas (mark-to-model + hedge pick-off)
- positivity constraint → the ONLY thing that closes the hard static arb; turns it into a small obstacle
  problem, composes with the B-spline convex-hull machinery.

**Fast-path status of the constrained version:** energy term + fixed-σ tension stay on the fast path
(constant Jacobian block, linear map). Positivity is an inequality ⇒ active-set/QP, NOT a plain linear
solve — but the active set changes rarely tick-to-tick ⇒ **frozen-active-set / warm-start**, parallel to
frozen-Jacobian streaming: freeze the active set, fast linear step, refresh the set only on a
constraint activation. Steady state keeps the microsecond path.

---

## 8. Open threads / next steps (pick up here)

1. **Derive the `Tension` region concretely:** exact tridiagonal `A(h,σ)`/`B` rows and closed-form
   integral weights `w(t)`; the σ→0 series unification with `NaturalCubic`.
2. **Closed form for `K₁`, `K₂`** (the stiffness matrices) on the flat-front + tension-back layout, and
   wire the `√μ L` pseudo-residual block into the LM.
3. **Positivity on the B-spline control-point basis:** `f≥0` reduces to control points ≥ 0 by the
   convex-hull property — cheap and elegant; work out the active-set warm-start.
4. **Size the actual pick-off:** how much unpinned curvature can a butterfly of our *real* on-strip
   instruments detect? Quantify the exposure rather than argue qualitatively.
5. **Frozen-tension convergence** argument in full rigor (semismooth Newton, `∂σ/∂x` order).
6. Decide default: global fixed σ (simple, fast, non-local) vs fixed-tension Hermite (local, `C¹`) vs
   adaptive (guaranteed monotone, AAD/frozen tier).

## 9. Key references

- Schweikert (1966), Cline (1974) — spline under tension.
- Holladay (1957) — minimum-curvature theorem.
- de Boor & Lynch (1966); Schultz & Varga (1967) — L-spline theory (general energy minimizers).
- Adams & van Deventer (1994) — maximum-smoothness forward curve (quartic under repricing constraints).
- Schumaker (1983); Rentrop (1980); Pruess; Späth — variable/adaptive tension, shape preservation.
- Hagan & West (2006), *Interpolation Methods for Curve Construction* — the reference on oscillation,
  non-locality, negativity, and arbitrage in curve interpolation; monotone-convex method.
