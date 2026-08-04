# VOL_XVA_ROADMAP.md — vol-book & XVA feature roadmap

An engineering roadmap, not a marketing list. It surveys the most differentiating **vol-book** and **XVA**
capabilities from the leading quant libraries (QuantLib, ORE, finmath, tf-quant-finance/JAX, and commercial
refs where public docs illuminate a feature), ranks them by impact, and — the part that matters — maps each
one **concretely onto this engine's primitives**: the `Scalar`-templated kernel, the W-cache compiled curve,
the batched `vol_cube`, the two-tier residual engine, forward-mode `ad::Dual`, and the **planned** MC path
engine + reverse-mode AAD tape + LGM/Cheyette models.

Read `ARCHITECTURE.md` (object graph) and `OPTIMIZATION.md` (the perf playbook + the options/vol section)
first; this sits on top of both. The options-engine consensus design and build order live in the
`options-engine` memory and its plan artifact — this roadmap extends that spine into the vol-book and
adds the XVA layer that does not exist yet.

---

## Where we are today (the honest baseline)

**Built** (`include/swaps/vol/`, QuantLib-free, sits on top of the calibrated curve, never touches the frozen
calc kernel):

- `normal.hpp` / `bachelier.hpp` — normal (Bachelier) price / vega / delta / gamma + safeguarded
  Newton–bisection implied vol. `Scalar`-templated.
- `sabr.hpp` — β=0 normal SABR smile (Hagan), closed-form ATM. `Scalar`-templated.
- `swaption.hpp` — forward swap rate + annuity from DFs → Bachelier/SABR European swaption.
- `cms.hpp` — first-order linear-TSR CMS convexity (Hagan), single optionlet. NOT full replication.
- `api/options.cpp` — the `swaption` verb and the batched **`vol_cube`** (`BundleSession::price_vol_cube`):
  a whole expiry×tenor×strike surface repriced off the calibrated `x` in one pass, `(forward, annuity)` cached
  against curve state, delta/vega/gamma SoA, **at parity with QuantLib** (~11 µs native surface reprice).

**Planned but NOT built** (the load-bearing dependencies for almost everything below):

- **LGM / Hull–White-1F**, then **Cheyette (1F quasi-Gaussian)** with a local-vol skin. `ad/` today is only
  `dual.hpp` (forward mode).
- **PDE oracle** (1F lattice) for early-exercise validation.
- **Reverse-mode AAD tape** (Savine-style block-list, thread-local, record→backprop→wipe) — the big XVA
  differentiator. Does not exist; `ad::Dual` is forward-mode only.
- **Monte-Carlo path engine** (`Model → PathGenerator → Product → Aggregator`), Sobol + Brownian bridge.
- **LSM (Longstaff–Schwartz) Bermudan** with analytic-European control variate; Andersen–Broadie dual bound.
- **No XVA layer at all**, and **no exposure simulation** — there is no netting set, no CSA/collateral
  object, no EPE/ENE/PFE, nothing.

### The dependency spine (read this before the rankings)

Most of the high-impact items are gated on three planned pieces, in this order:

```
LGM/HW1F model  ─┐
                 ├─► MC path engine ─► LSM Bermudan ─► American-MC exposure ─► XVA aggregation
reverse-AAD tape ┘                                              │
                                                                └─► AAD XVA sensitivities (the differentiator)
```

Anything tagged **[needs MC]** or **[needs tape]** below cannot start until that piece lands. The vol-book
section is deliberately front-loaded with items that need **neither** — they extend the analytic `vol_cube`
and are shippable now — so there is real work to do before the heavy machinery exists.

**Effort key:** S ≈ days · M ≈ 1–2 wks · L ≈ 3–6 wks · XL ≈ multi-month / needs a new subsystem first.
Efforts assume the listed dependencies already exist.

---

## Section 1 — VOL-BOOK features, ranked by impact/impressiveness

### 1.1 — Arbitrage-free SABR + smile calibration to a strip  **[no MC] · effort M**

**What it is.** Fit `(α, ρ, ν)` (β fixed for SOFR normal) to a whole strike strip per (expiry, tenor) node,
and guarantee the fitted smile is **arbitrage-free** — Hagan's own expansion generates negative densities in
the wings, so you either apply Hagan's arbitrage-free PDE correction or fit the density directly.

**Why it's impressive / who does it best.** QuantLib ships `SabrSmileSection` + `ZabrSmileSection` and the
**Andreasen–Huge** arbitrage-free one-step local-vol interpolator (`AndreasenHugeVolatilityInterpl`) — the
cleanest published route to a strictly arbitrage-free surface from sparse quotes. finmath and commercial desks
(Murex, Numerix) treat "no butterfly/calendar arbitrage, ever" as table stakes; it is what separates a demo
smile from a book-quality one.

**Concrete mapping.** We already have `sabr_normal_vol` and the `vol_cube` batched reprice. This is a
**calibration** on top:
- Reuse the **Levenberg–Marquardt** solver in `calibration/` (it is not curve-specific) with the residual
  `Σ_k w_k · (σ_SABR(K_k; α,ρ,ν) − σ_market(K_k))²`. Because `sabr_normal_vol` is `Scalar`-templated, seed it
  with `ad::Dual` and you get the **analytic 3×N Jacobian for free** — no bumping — exactly the discipline the
  curve calibration already uses.
- Add an **arbitrage gate**: compute the risk-neutral density `∂²(call price)/∂K²` from the fitted smile on a
  fine strike grid and assert non-negativity; reject/penalise fits that violate it. This is a pure `double`
  post-check reusing `bachelier_price`.
- The clean-room win is **Andreasen–Huge** as a *second* smile representation next to SABR: one implicit
  finite-difference step of Dupire's forward PDE, calibrated so it is arbitrage-free by construction. It slots
  in as an alternative `SmileSection`-like object that `vol_cube` can read a vol from at any strike.
- **Reuses:** `vol_cube` fwd/annuity cache, `Scalar` AAD Jacobian, the LM solver. **Depends on:** nothing new.
  Ordering: do SABR-strip calibration first (M), Andreasen–Huge as a follow-on (M–L).

### 1.2 — Full vol-cube construction & no-arbitrage interpolation  **[no MC] · effort M–L**

**What it is.** Turn the discrete `vol_cube` grid into a **queryable surface**: interpolate in expiry, tenor,
and strike with controls that prevent calendar arbitrage (variance non-decreasing in expiry) and butterfly
arbitrage (convex in strike), and fill missing nodes.

**Why it's impressive / who does it best.** ORE/QuantLib's `SwaptionVolatilityStructure` /
`SwaptionVolCube1/2` (SABR-parametrised cube with ATM interpolation and smile spreads) is the reference. This
is the object a whole desk queries; getting the interpolation *arbitrage-aware* is the differentiator over a
naïve bilinear grid.

**Concrete mapping.** `vol_cube` already produces the SoA at grid nodes. Build a thin **surface object** over
it that (a) interpolates ATM vol in (expiry, tenor) using the same region schemes the curve layer already owns
(`curve/` has Linear/NaturalCubic/MonotoneCubic/Tension — reuse `MonotoneCubic` in the total-variance
dimension to keep it monotone → no calendar arbitrage), and (b) attaches a per-node SABR smile (§1.1) for the
strike dimension. The result is a `SwaptionVolCube`-equivalent that any pricer samples. **Reuses:** the
`curve/` interpolation schemes (they are generic 1-D interpolators, not curve-specific!), `vol_cube`, SABR.
**Depends on:** §1.1 for the smile parametrisation. This is the natural "productionise the demo" step.

### 1.3 — Cap/floor & swaption vol stripping  **[no MC] · effort M**

**What it is.** Markets quote **flat** cap vols (one vol per cap maturity); you must **strip** them into
**caplet** (forward) vols to price anything term-structured. Symmetrically, bootstrap a swaption vol surface
from co-terminal/diagonal quotes.

**Why it's impressive / who does it best.** QuantLib's `OptionletStripper1/2` +
`StrippedOptionletAdapter` is the canonical implementation; it is fiddly (overlapping caps, the last-caplet
problem, interpolation of the stripped vols) and every rates desk needs it. It is unglamorous but it is what
makes cap/floor books real.

**Concrete mapping.** A **bootstrap**: caplet `i`'s vol is solved so that the sum of caplet prices reproduces
the quoted cap price, marching out the maturity axis — structurally the same shape as the curve bootstrap we
already do, but on vols not forwards. Reuse the safeguarded Newton in `bachelier_implied_vol` per caplet and
the `vol_cube` machinery to get each caplet's forward/annuity from the calibrated curve. Extend the `build/`
schedule layer (already there) to roll the caplet fixing/pay schedule. **Reuses:** `bachelier_implied_vol`,
`vol_cube` fwd cache, `build/schedule.hpp`. **Depends on:** nothing new. A caplet is just a 1-period swaption,
so this largely falls out of the swaption path.

### 1.4 — CMS & CMS-spread replication (full Hagan static replication)  **[no MC] · effort M–L**

**What it is.** Replace the current *first-order linear-TSR* CMS adjustment (`cms.hpp`) with **full static
replication**: integrate a strip of swaptions against the second derivative `G''(K)` of the annuity mapping
over the whole SABR smile. Then CMS-spread options via a bivariate (copula/Gaussian) integration.

**Why it's impressive / who does it best.** This is the textbook Hagan "Convexity Conundrums" replication —
QuantLib has it (`CmsCoupon` + `GsrProcess`/`LinearTsrPricer` + `NumericHaganPricer`), and it is the honest
way to price CMS caps/floors with smile. CMS-spread (steepeners) needs the joint smile and is a genuine
differentiator; ORE and every structured-rates desk carry it.

**Concrete mapping.** `cms.hpp` already has the linear-TSR skeleton and the `CmsForward` struct; the comment
even flags full replication as the labelled follow-up. Build the replication integral as a **strike quadrature
over `bachelier_price` off the SABR smile** (Gauss–Kronrod or a fixed fine grid) weighted by `G''(K)` from the
terminal-swap-rate model. It is a pure analytic pass on top of §1.1's smile — no MC. CMS-**spread** adds a 2-D
integration with a correlation input; keep it Gaussian-copula for the first cut. **Reuses:** `sabr.hpp`,
`bachelier_price`, `swaption.hpp` fwd/annuity. **Depends on:** §1.1 (a good smile is the integrand). This is
one of the most "quant-shop" credible items and needs nothing heavy.

### 1.5 — Vega / skew / vanna-volga risk representations  **[no MC] · effort S–M**

**What it is.** Present vol risk the way a trader hedges it: **bucketed vega** (∂P/∂σ per expiry×tenor node),
**skew/vanna/volga** (∂P/∂ρ, cross ∂²P/∂F∂σ, ∂²P/∂σ²), and SABR-parameter risk (∂P/∂α, ∂P/∂ρ, ∂P/∂ν).

**Why it's impressive / who does it best.** Bloomberg MARS and Murex are known for the *representation* — the
same risk projected onto tradeable hedge instruments. Vanna-volga itself (a market-standard smile-consistent
adjustment) is a neat, cheap, recognisable technique.

**Concrete mapping.** This is almost free given the `Scalar` discipline: seed the SABR params and the forward
with `ad::Dual` through `sabr_normal_vol → bachelier_price` and read ∂P/∂{α,ρ,ν,F,σ} directly — the `vol_cube`
already emits vega/delta/gamma this way, so extend the same SoA with the SABR-parameter and second-order
columns. Bucketed vega = project cell vegas onto the surface nodes (reuse the §1.2 surface). Vanna-volga is a
3-instrument (ATM + two wings) closed-form overlay. **Reuses:** `ad::Dual`, `vol_cube` SoA, SABR. **Depends
on:** §1.1/§1.2 for buckets. Lowest effort, high "looks like a real risk system" payoff.

### 1.6 — Local vol (Dupire) + stochastic-local-vol (SLV)  **[needs MC for SLV] · effort L (LV) / XL (SLV)**

**What it is.** **Dupire local vol** `σ_LV(K,T)` extracted from the arbitrage-free call surface — the unique
1-factor diffusion reproducing all European prices. **SLV** = local-vol skin on top of a stochastic-vol
backbone (e.g. SABR/Heston), calibrated so the model reprices vanillas *exactly* while retaining realistic
forward smile dynamics.

**Why it's impressive / who does it best.** SLV is the crown jewel of an equity/FX vol engine and increasingly
rates; Numerix and Murex market it heavily. tf-quant-finance and QuantLib both have local-vol; the
**Andreasen–Huge** route (§1.1) is specifically prized because it yields a *stable* local vol from sparse
quotes. Getting the **particle/leverage-function calibration** right (Guyon–Henry-Labordère) is what makes it
hard and impressive.

**Concrete mapping.** Dupire LV from the arbitrage-free surface (§1.1/§1.2) is an analytic derivative
computation — do it once the surface is solid; the natural engine home is the **planned Cheyette local-vol
skin** the options-engine plan already pre-wired ("1F quasi-Gaussian, local-vol skin pre-wired"). SLV is
genuinely **XL and gated on the MC engine**: the leverage-function calibration is a forward-Fokker–Planck or
particle method that needs the path engine + a PDE oracle to validate. Flag it honestly as a *late* item.
**Reuses:** Andreasen–Huge surface, the planned Cheyette skin, MC engine. **Depends on:** §1.1, MC, PDE oracle.

### 1.7 — Term-structure-of-vol / Bermudan-consistent model calibration  **[needs LGM/PDE] · effort L**

**What it is.** Calibrate the **LGM/Hull–White** (later Cheyette) model's piecewise `σ(t)` and mean reversion
to the relevant European swaptions (co-terminal diagonal for a Bermudan), so the *model* — not just a smile —
reprices the vanillas that hedge an exotic. This is the bridge from the vanilla vol-book to the exotics book.

**Why it's impressive / who does it best.** ORE/QuantLib's `Gsr`/`MarkovFunctional` + `Swaption` engines and
finmath's LMM calibration are the references. Jamshidian's decomposition gives an **analytic** co-terminal
calibration for 1F Gaussian — elegant and fast, and it is exactly what the options-engine plan specifies
("analytic Jamshidian calibration to the co-terminal swaption diagonal").

**Concrete mapping.** This *is* the planned LGM/HW1F milestone. Reuse the LM solver + the `Scalar`/`ad::Dual`
Jacobian discipline to fit `σ(t)` to the diagonal; use Jamshidian to make each swaption an analytic function
of the model params (no MC needed for the *calibration*, only for later exotic pricing). The PDE oracle
validates. **Reuses:** LM solver, `ad::Dual`, `vol_cube` (supplies the target European prices/vols).
**Depends on:** the LGM/HW1F model existing. This is the linchpin between Section 1 and the exotics/XVA half.

---

## Section 2 — XVA features, ranked by impact/impressiveness

> **Reality check:** there is **no XVA layer and no exposure engine today**. Everything here is net-new and
> the exposure-simulation items are gated on the **planned MC path engine**. The single biggest differentiator
> — AAD XVA sensitivities — additionally needs the **reverse-mode tape**. Sequence accordingly.

### 2.1 — Monte-Carlo exposure simulation: EPE / ENE / PFE profiles  **[needs MC] · effort XL (first XVA build) then L**

**What it is.** Simulate the netting set's mark-to-market forward in time under a risk-neutral model, on a
time grid, to produce **exposure profiles**: EE/EPE (expected positive exposure), ENE, and PFE (a high
quantile). This is the substrate *every* XVA sits on.

**Why it's impressive / who does it best.** **ORE (Open Source Risk Engine)** is the canonical open reference:
its whole architecture is "simulate market paths → reprice the portfolio at each (path, time) node → aggregate
into exposure → integrate into XVA." finmath has a clean pedagogical version. This is the foundational,
must-exist-first capability.

**Concrete mapping.** This is where the **planned MC path engine** earns its keep, and where our perf DNA
transfers directly:
- The `Model → PathGenerator → Product → Aggregator` skeleton from the options plan **is** the exposure
  engine: paths of the short-rate/curve state, then **reprice the book at each node**. Crucially, our
  **W-cache is the reprice kernel** — a simulated forward curve is just a new knot vector `x`, and
  `MultiCurveBook` / `Portfolio` already reprice a book off `x` as a cheap `double` pass (`price_portfolio`).
  So exposure at a node = evolve `x(t, path)` under the model, then run the existing compiled reprice. That
  reuse is the differentiating story: *we already have a microsecond book reprice; MC exposure is that reprice
  in a path loop.*
- Reuse `parallel::ThreadPool`, the SoA/alloc-free discipline, and Sobol+Brownian-bridge from the plan.
- **Reuses:** MC path engine, `MultiCurveBook`/`Portfolio` W-cache reprice, `ThreadPool`, SoA loop.
  **Depends on:** MC engine + LGM model. The *first* profile is XL (stands up the whole subsystem); each later
  metric is incremental.

### 2.2 — CVA / DVA  **[needs 2.1] · effort M (once exposure exists)**

**What it is.** **CVA** = expected loss from counterparty default = `LGD · ∫ EPE(t) · dPD(t)` (discounted);
**DVA** the symmetric own-default benefit off ENE. The headline counterparty-risk number.

**Why it's impressive / who does it best.** ORE computes CVA/DVA as a thin integration over the exposure
profile against a hazard-rate/survival curve bootstrapped from CDS. It is *conceptually* simple once §2.1
exists — which is exactly why exposure is ranked first.

**Concrete mapping.** A quadrature over the EPE/ENE profile (§2.1) against a **survival curve** — and we
already have a curve bootstrapper: a hazard/survival curve is *another `ModularCurve`* calibrated to CDS
spreads via the existing LM machinery. So CVA reuses (a) the exposure profile and (b) the curve layer for
survival probabilities. **Reuses:** §2.1 profiles, `curve/` + LM for the survival curve. **Depends on:** §2.1.

### 2.3 — AAD-based XVA sensitivities  **[needs tape + 2.1] · effort XL (tape) then L**

**What it is.** Thousands of XVA Greeks (sensitivity to every curve pillar, vol, CDS spread, FX) from **one
adjoint (reverse-mode) pass** at ≈4× the cost of one XVA valuation, instead of thousands of finite-difference
re-simulations.

**Why it's impressive / who does it best.** **This is THE differentiator** and the reason the whole
options-engine plan foregrounds a reverse-mode tape. It is finmath's headline (stochastic AAD, Capriotti's
work), Numerix/Murex sell it, and Capriotti–Giles "15 Years of AAD" is the canonical account. A bump-and-run
XVA sensitivity run is a compute-farm overnight batch; AAD makes it interactive. Doing it *well* over a Monte
Carlo exposure simulation (path-wise adjoints, tape checkpointing at observation dates, handling the LSM
regression boundary via the envelope theorem) is genuinely hard and genuinely impressive.

**Concrete mapping.** This is the payoff of two planned pieces working together:
- The **reverse-mode AAD tape** (planned, Savine-style, thread-local record→backprop→wipe + checkpointing).
  Our `Scalar` templating means the *same* exposure/XVA code compiles under the tape type — the mode flips
  from forward `Dual` to adjoint, exactly as OPTIMIZATION.md's "analytic AAD → reverse-mode AAD tape" note
  anticipates.
- Applied over §2.1's exposure sim: one backward pass yields dXVA/d(every input). The subtlety is the LSM
  exercise boundary (§2.5) — **freeze it (envelope theorem) for first-order** Greeks, as the plan already
  states; differentiate through only where required.
- **Reuses:** the planned tape, §2.1 exposure, `Scalar` discipline. **Depends on:** reverse-mode tape (XL to
  build once) + §2.1. Rank-1 differentiator, but honestly the deepest lift in this document.

### 2.4 — CSA / collateral & margin: VM/IM, thresholds, MTA  **[needs 2.1] · effort M–L**

**What it is.** Model the CSA: **variation margin** (collateralised exposure = uncollateralised minus posted
VM, with **threshold**, **minimum transfer amount**, **margin period of risk**), and **initial margin**. This
transforms the raw exposure profile before any XVA integral.

**Why it's impressive / who does it best.** ORE models CSAs in detail (thresholds, MTA, MPoR, rounding); it is
what makes CVA/FVA *realistic* rather than a gross-exposure toy. The MPoR lag (exposure over a 10-day close-out
window) is the subtle, credible bit.

**Concrete mapping.** A **transform on the exposure paths** from §2.1: at each node, `collateralised_exposure =
max(0, MtM − C)` where `C` evolves under the CSA rules (threshold/MTA/rounding) with an MPoR time-shift. It is
path post-processing, not new pricing — so it reuses §2.1 wholesale and adds a `CollateralModel`/`CsaSpec`
object. **Reuses:** §2.1 paths. **Depends on:** §2.1. Needed before FVA/MVA mean anything.

### 2.5 — Backward / American Monte-Carlo (AMC) exposure  **[needs MC + LSM] · effort L (on top of LSM)**

**What it is.** For portfolios with **early exercise** (Bermudan swaptions, callables), exposure requires the
continuation value at every node — computed by **regression (Longstaff–Schwartz)** so the exercise decision
and the post-exercise exposure are consistent across paths.

**Why it's impressive / who does it best.** ORE's AMC framework is a marquee feature; it is what lets a bank
put callable/exotic trades into the *same* exposure/XVA engine as the vanillas. finmath's LSM is the clean
reference implementation.

**Concrete mapping.** This *is* the planned **LSM Bermudan** milestone, reused for exposure rather than just
price: the regressed continuation functions define both the exercise boundary and the pathwise exposure after
exercise. The analytic-European control variate + Andersen–Broadie dual bound (both in the plan) validate it.
**Reuses:** MC engine, planned LSM, analytic swaption as control variate. **Depends on:** §2.1 + LSM Bermudan.

### 2.6 — FVA (funding valuation adjustment)  **[needs 2.1 + 2.4] · effort M**

**What it is.** The cost/benefit of funding the *uncollateralised* portion of exposure at the bank's funding
spread over the risk-free/OIS rate. Integrates the (collateral-adjusted) EPE/ENE against a funding-spread
curve.

**Why it's impressive / who does it best.** ORE computes FVA (FCA/FBA) directly off the collateralised
exposure and a funding curve; the interesting modelling choice is the overlap/double-counting with DVA, which
a credible engine surfaces explicitly.

**Concrete mapping.** Another quadrature over the §2.4 collateral-adjusted profile, this time against a
**funding-spread curve** — again *another `ModularCurve`*. Mechanically it is CVA (§2.2) with a different
weighting curve and the collateral transform applied. **Reuses:** §2.4 profiles, `curve/` for the funding
curve. **Depends on:** §2.1 + §2.4.

### 2.7 — Netting sets & aggregation  **[needs 2.1] · effort M**

**What it is.** The bookkeeping that makes XVA correct: trades group into **netting sets** (exposure nets
*within* a set, is gross *across* sets), each set maps to a CSA and a counterparty. Aggregation must happen
*before* the `max(0, ·)` in exposure.

**Why it's impressive / who does it best.** Unglamorous but load-bearing — ORE's whole data model is
netting-set-centric. Getting the aggregation order right (net MtM within set, *then* floor at zero) is a
classic correctness trap.

**Concrete mapping.** A structural layer over `MultiCurveBook`/`Portfolio`: a `NettingSet` = a list of book
positions + a `CsaSpec` (§2.4) + a counterparty (survival curve, §2.2). The per-node reprice already produces
per-trade MtM; netting is a grouped sum before the exposure floor. **Reuses:** `MultiCurveBook`, §2.1 node
reprice. **Depends on:** §2.1. Do this alongside §2.1 — the exposure engine should be netting-set-native from
day one rather than retrofitted.

### 2.8 — Wrong-way risk (WWR)  **[needs 2.1 + 2.2] · effort M–L**

**What it is.** Correlation between exposure and counterparty default (exposure tends to be *high exactly when*
the counterparty is likely to default). Modelled via a correlated hazard rate or a stochastic-intensity model.

**Why it's impressive / who does it best.** A recognised sophistication gap in simpler engines; ORE supports
it, and it is a known regulatory/quant talking point. The clean approach couples the default intensity to the
simulated market state.

**Concrete mapping.** Couple the §2.2 survival curve's hazard rate to the §2.1 simulated state (e.g. a Hull
"h(t) = h0·exp(b·r(t))" or a Gaussian-copula tilt of default times toward high-exposure paths). It is a
modelling overlay on the existing exposure + survival machinery, not new infrastructure. **Reuses:** §2.1 +
§2.2. **Depends on:** §2.1 + §2.2.

### 2.9 — SIMM / ISDA Initial Margin  **[needs risk buckets, no MC] · effort M–L**

**What it is.** The ISDA **Standard Initial Margin Model**: a prescribed, sensitivity-based (delta/vega/curvature
across risk classes and buckets, with prescribed correlations) formula that turns a portfolio's SIMM
sensitivities into an IM number.

**Why it's impressive / who does it best.** A concrete, industry-mandated deliverable (bilateral IM / UMR);
ORE has a SIMM module. It is attractive because it is a **rules engine over sensitivities we already produce
analytically** — no MC required for SIMM *itself*.

**Concrete mapping.** SIMM consumes CRIF-style sensitivities. We already generate curve delta ladders
analytically (`generate_risk`, `price_portfolio_risk`, `transform_matrix` — all AAD, no bumping) and can
produce vega from the `vol_cube`. SIMM is then a **deterministic aggregation** (bucket → risk-class →
portfolio with ISDA-published correlations). Implement it as a pure post-processor over our existing
sensitivity output. **Reuses:** `generate_risk`/`price_portfolio_risk` ladders, `vol_cube` vega. **Depends
on:** nothing new for SIMM itself — but it becomes *powerful* combined with §2.3 (MVA below).

### 2.10 — MVA / KVA / ColVA  **[needs 2.9 + 2.3/2.1] · effort L each**

**What it is.** **MVA** = funding cost of posting **initial margin** over the trade's life (integrate expected
future IM × funding spread). **KVA** = cost of holding regulatory capital over the life. **ColVA** = the
collateral rate vs. the discount rate basis adjustment.

**Why it's impressive / who does it best.** The "full XVA suite" completeness items; ORE covers them. **MVA is
the standout** and the natural capstone of the AAD story: expected future IM requires **future SIMM
sensitivities along each path**, which is a *pathwise-AAD-over-the-simulation* problem — precisely where §2.3's
tape and §2.9's SIMM combine into something few open engines do well.

**Concrete mapping.** MVA = simulate forward, compute SIMM (§2.9) at each future node from **pathwise
sensitivities produced by the AAD tape (§2.3)**, integrate expected IM against a funding spread. This is the
most advanced item in the document and deliberately last: it stacks §2.1 + §2.3 + §2.9. ColVA is a cheaper
basis integral over the collateral profile (§2.4). KVA needs a regulatory-capital (SA-CCR / IMM) model on top
of exposure. **Reuses:** everything above. **Depends on:** §2.1, §2.3, §2.9. Capstone.

---

## Recommended sequence (tying it to the existing options-engine build order)

The options-engine plan's order is: **Bachelier → SABR → CMS → LGM+PDE → AAD tape → MC → LSM Bermudan →
CMS-spread/RFR/2F.** We are through SABR + the analytic `vol_cube`. This roadmap slots in as follows — the
vol-book front half needs **no new subsystem** and should be done first to build a real vol book while the
heavy machinery is stood up; the XVA back half unlocks strictly in dependency order.

**Phase A — extend the analytic vol-book (no MC, no tape; ship now):**
1. **SABR strip calibration + arbitrage gate** (§1.1, M) — reuse LM + `ad::Dual`.
2. **Vega/skew/vanna-volga risk SoA** (§1.5, S–M) — nearly free off the `Scalar` discipline.
3. **Cap/floor & swaption vol stripping** (§1.3, M) — a caplet is a 1-period swaption.
4. **Full vol-cube surface + no-arbitrage interpolation** (§1.2, M–L) — reuse `curve/` interpolators.
5. **Full Hagan CMS replication + CMS-spread** (§1.4, M–L) — the labelled follow-up in `cms.hpp`.
6. **Andreasen–Huge arbitrage-free surface** (§1.1 tail, M–L) and **Dupire local vol** (§1.6, L).

**Phase B — the model & MC spine (the planned milestones; everything downstream gates on these):**
7. **LGM/HW1F + PDE oracle + Jamshidian term-structure-of-vol calibration** (§1.7, L) — the plan's LGM+PDE step.
8. **Reverse-mode AAD tape** (§2.3 enabler, XL) — the plan's AAD-tape step; the XVA differentiator's foundation.
9. **MC path engine** (§2.1 enabler, XL) — `Model→PathGenerator→Product→Aggregator`, reusing ThreadPool/SoA/W-cache.
10. **LSM Bermudan** (§2.5 enabler, L) — the plan's LSM step, with the analytic-European control variate.

**Phase C — the XVA layer (net-new; strict dependency order):**
11. **Netting-set-native MC exposure: EPE/ENE/PFE** (§2.1 + §2.7, XL first) — reprice the book off simulated `x`.
12. **CSA/collateral (VM/IM, threshold, MTA, MPoR)** (§2.4, M–L) — a transform on the exposure paths.
13. **CVA/DVA** (§2.2, M) then **FVA** (§2.6, M) — quadratures over profiles against survival/funding curves.
14. **AMC exposure for callables** (§2.5, L) — LSM continuation reused for exposure.
15. **AAD XVA sensitivities** (§2.3, L on top of the tape) — **the headline differentiator**; thousands of Greeks, one adjoint pass.
16. **SIMM/ISDA IM** (§2.9, M–L; can start in Phase A as a pure sensitivity aggregator) → **MVA/KVA/ColVA** (§2.10, L each) — the capstone.
17. **Wrong-way risk** (§2.8, M–L) — a correlation overlay, slot in after CVA.

**The three things to be honest about:** (1) nearly all of Section 2 is gated on the MC engine (item 9), which
does not exist — Phase A is where the shippable near-term value is; (2) the AAD-XVA differentiator (§2.3/item
15) additionally needs the reverse-mode tape (item 8), the single deepest build in this document; (3) SLV
(§1.6) and MVA (§2.10) are genuine multi-month capstones — worth naming as the destination, not the next
sprint. The consistent architectural bet throughout is that **our W-cache book reprice is the exposure kernel**
and **our `Scalar` templating is the AAD substrate** — the same two ideas that made calibration fast, reused
for the vol-book and XVA.
