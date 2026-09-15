# SwapsEngine — a history

How this engine was built and why, told in broad strokes, with detail on the features that make it unusual.

The engine was built between July and mid-September 2026 by one owner working with AI coding agents (Claude Code). The owner set the goals,
the market model and the rules; the agents wrote most of the code under those rules, and later audited it. This history is condensed from
the git history (397 commits), the project's design documents, the review and task ledgers, and the working-session notes.

**How to read the numbers.** Figures quoted from commit messages are "as measured at the time" on the owner's reference machine. They were not
re-measured for this document. From September 2026 the performance gate compares the engine only against its own baselines; QuantLib timings
are informational.

---

## 1. What it is

SwapsEngine is a header-only C++20 library for **multi-curve interest-rate curve calibration and rates analytics**, built for one job above all:
re-pricing a live multi-curve rates book on every market tick, fast and exactly. The owner's framing, from the first week, was real-time
market-making. There a stale price hands a client a free option and a wrong price is worse, so exactness and latency are one requirement,
never a trade-off.

Around the calibration core sit:
- a batched analytics layer: book reprice, analytic risk ladders, scenarios, P&L explain;
- a conventions database;
- options, bonds and exposure kernels;
- a JSON/C-ABI facade with generated Python and Excel bindings.

A sibling web application composes curves and streams them live (section 7).

QuantLib is the correctness oracle and the speed reference, not a runtime dependency. The shipped engine needs only Eigen, plus Boost.JSON for
the API layer.

## 2. The arc, in four phases

### Phase I — a faster, exact QuantLib extension (7–22 July)

The founding goal was to replace QuantLib's two slowest workflows with a global, differentiated, vectorised implementation:
- swap-curve calibration;
- bulk swap analytics.

Correctness and performance gates existed from the first day: a harness compared every number against QuantLib, and a benchmark gate compared
speed. Checkpoint tags marked each green milestone.

- **Foundations (7–10 July).** A templated pricing kernel and a global Levenberg–Marquardt calibration of all knots together, rather than
  QuantLib's sequential bootstrap. The decisive insight came early: a curve that is **linear in log-discount space** lets the engine cache a weight
  matrix W once and price every discount factor as `DF = exp(−W·x)`. The same structure gives an analytic Jacobian and analytic risk through
  the implicit-function theorem. Early measured results: curve build 6.9× faster than QuantLib's GlobalBootstrap; delta ladder 39× faster than
  bump-and-reprice; a 1,000-swap book reprice about 126× faster.
- **Microsecond streaming (11–12 July).** A warm re-calibration path took a full LM solve from 2.8 ms to 48 µs. Then came **exact frozen-Newton**
  streaming, the design the rest of the project refined:
  - every tick iterates to the exact curve using a cached Jacobian as a preconditioner;
  - the Jacobian is recomputed only when progress stalls.

  The owner rejected looser tolerances and synthetic-looking curves at this point. Curves had to show flat forwards between central-bank
  meetings, a smooth back end, and realistic 0.1–0.2 bp ticks. That "realistic bundle" became the benchmark model.
- **Multi-curve bundles (13–15 July).** N interdependent curves (e.g. SOFR, Fed Funds via averaging futures and basis, ESTR, EURIBOR) calibrated
  jointly over one stacked state:
  - a curve's outright-vs-spread status is part of its definition;
  - independent sub-bundles solve separately;
  - the W-cache extends to the whole bundle.

  A realistic 4-curve bundle built about 20× faster than QuantLib's iterative bootstrap.
- **One generic instrument pipeline (16–19 July).** Currency, OIS/IBOR, spread/outright and averaging/compounding all pass through one cashflow
  model; index knowledge lives only in data and tests. Additions:
  - B-spline and MonotoneCubic (Hyman) interpolation;
  - moment-integrated daily averaging;
  - RFR lookback, lockout and observation shift;
  - an allocation-free tick loop.

  The rule "hot path over cold path" came from the owner: a live engine almost always has a previous calibration, so optimise the tick.
- **Multi-currency and the public API (20–22 July).**
  - EUR curves and cross-currency: FX points to about one year, then mark-to-market cross-currency basis;
  - a smoothness regulariser;
  - the first JSON object graph with a session facade.

### Phase II — the web composer forces genericity (22 July – 2 August)

A proof-of-concept web app, where users compose curves and watch them calibrate to a simulated live market, exposed every special case in the
engine. Its demands became the engine's defining features:

- **Interpolation regions replace "front/back" curves.** A curve became an ordered list of regions in any order, and the W-cache was
  generalised to any region that is linear in its knots.
- **Portfolio instruments and bid/offer bands.** Butterflies quoted in basis points, and bands that let extra instruments *guide* a fit rather
  than pin it. Both came from the owner's desk practice.
- **A conventions database as the single source of truth.** Day counts, calendars, frequencies and lags; a new product is a data row.
- **Turns** (year-end jumps) as instruments, and tension-energy smoothing that must never erase an explicit discontinuity.
- **"The engine owns everything."** Risk logic moved from the web into C++, and one API descriptor began generating every language binding.
- **A C++ bundle compiler** mirroring the web's Python spec compiler to about 1e-9, with a parity gate.

### Phase III — breadth (2 August – 7 September)

- **Options (early August).** Bachelier, SABR with calibration, swaption and vol-cube repricing. A QuantLib-paired benchmark caught a 40× reprice
  regression hidden behind "faster than before". The fix led to a compiled vol surface at about 2.3 µs per reprice. This episode produced the
  rule that every performance claim is measured natively, never through JSON or a web page.
- **Bonds.** Batched yield-space and curve-space kernels, and asset-swap spreads.
- **Exposure.** A Monte-Carlo EPE/ENE/PFE kernel with pooled, heap-free dual numbers.
- **Object model and commercial review (late August – early September).** A rework into:
  - first-class regions with their own smoothing;
  - curve-unbound instruments that reference indices;
  - reference-data objects;
  - a forkable model;
  - Market and Trade domains.

  A commercial review against QuantLib, ORE and OpenGamma Strata concluded: a best-in-class calibration and risk core, but not yet a product.
  G20 conventions and a breadth wave followed: FX options, inflation, bond futures, CDS, NDF, scenarios.
- **The first fresh-context review and bug hunt (7 September).** The owner declined in-session self-grading and commissioned independent
  reviews by agents with no development history. The verdict was "a strong kernel inside a periphery that over-claims". A kernel bug hunt under
  a "no executable reproduction, no bug" rule found 44 reproduced bugs; 24 were fixed with regression tests the same day. The **Huber bid/offer
  band with an active-set streamer** replaced an earlier Gaussian band that had multiple minima.

### Phase IV — engine-first cleanup and honest gates (8–15 September)

A second fresh review asked a harder question: *do the gates actually do what they claim?* The answer was largely no:
- only about a third of "oracle" tests compared against QuantLib;
- the performance gate only failed at about 2×;
- QuantLib had been built with different compiler flags;
- mutation testing showed tests passing with their feature deleted.

The owner reset the programme: **engine first, then the APIs; no web work until the engine is done.** A staged plan (E0–E7) followed, each
stage exiting before the next began:

- **E0 Principles as law.** The written rules P0–P14 are in section 4.
- **E1 An honest gate:**
  - one source of compiler flags;
  - QuantLib rebuilt with the engine's own flags;
  - a hard 1.25× self-regression limit;
  - absolute targets that only ratchet down;
  - a structural oracle registry with locked assertion counts;
  - a missing QuantLib made fatal;
  - CI and nightly runs.
- **E2 Conventions into the engine.** A runtime registry where unknown identifiers throw (silent fallbacks had mispriced trades). Every calendar
  checked day by day against QuantLib; convention literals in code ratcheted from 130 to 0.
- **E3 A deep review, hot path first.** An allocation and execution census of every production path, then six fresh-context reviews with probes.
- **E4 Fixes, repro-first.** Every fix carries a test shown failing on the reverted bug. This included the object-model decision (section 3.12)
  and an 18-item ordered fix list.
- **E5 A test rewrite** into a six-class taxonomy, with a mutation gate.
- **E6 Refactor.** Dead code deleted: cap stripping, CMS and a parallel cluster with no consumers. Duplicates collapsed, include cycles gated.
- **E7 "100% solid".** Principle P14 made every JSON verb a pure codec (decode → one library call → emit), enforced by a gate that measures logic
  left in the verb layer from the compiler's typed AST. A QuantLib oracle sweep of every verb followed, and a queue of convention corrections,
  each checked against QuantLib or OpenGamma Strata reference data.

Along the way a **"one golden source" programme** (10 September) required one engine definition per financial fundamental, one QuantLib oracle
each, and a per-cashflow router deciding the fast or generic tier as data. It was triggered by a duplicated averaging formula that had been
365/360 too high while a parity test pinned the wrong value.

The last days went into **streaming correctness** (14–15 September):
- a two-threshold rank-safe operator for near-singular curves;
- a rule that a quote is always four numbers, with a target outside its band refused;
- guard tests on every streaming stage;
- benchmark "premises";
- a hot-path census;
- a run of bit-identical allocation reductions.

A research line on compiling the pricing graph into fixed kernels was explored and parked on a separate branch (section 8).

## 3. The unique features, and why they exist

### 3.1 One curve type: order-agnostic interpolation regions

**What.** `ModularCurve` is an ordered list of regions. Flat, Linear, NaturalCubic, Hermite, MonotoneCubic, BSpline and Tension compose in any
order and position. A curve "flavour" (say, flat forwards between meeting dates, then a C¹ Hermite back end) is a list chosen at runtime, not a
class.

**Why.**
- The first design was a two-region curve, and every new flavour duplicated region maths, linearity checks and caches.
- A compile-time multi-region type was retired because it measured *slower* than type erasure: allocation dominated dispatch cost.

**How.**
- Each region maps its knot slice to the forward rate and its integral, and joins its neighbour continuously.
- The first region extrapolates flat, not pinned to zero.
- A combinatorial test drives every single, pair and triple of schemes (7 + 49 + 343) through every join.
- Smoothing can be set per region, which the design review found no surveyed commercial product offers.

### 3.2 One instrument, one rate formula, conventions as data

Every floating rate is one formula over sub-periods: `(Σ w·[DF(s)/DF(e) − 1] + realized) / τ`.
- A compounded OIS coupon is one telescoped sub-period.
- An averaged coupon is one sub-period per business day.
- An IBOR fixing is one.

Curve roles live on legs, so tenor basis and cross-currency need no new instrument types. Portfolios, FX forwards, MtM basis and turn jumps are
quote kinds, not classes.

The conventions database (18 currencies, 23 calendars, 26 indices, 34 products, plus bonds, CDS, FX pairs and central-bank schedules) is
schema-validated JSON, compiled into constant arrays, with a runtime overlay. A missed lookup throws. Calendars are interpreted holiday rules,
checked day by day against QuantLib.

### 3.3 The compiled W-cache: `DF = exp(−W·x)`

For linear layouts the integral of the forward curve at any time is a fixed weight row times the knot vector. The engine builds the block matrix
`W` once, by one AAD pass whose gradient is the weight row. After that, every reprice is a matrix-vector product plus a vectorised `exp`, and
the Jacobian is analytic: `J = −(∂r/∂DF · diag DF) · W`.

It matches AAD to about 1e-15 at roughly 15× the speed. Linearity is decided per time point, so only cashflows that reach a value-dependent region
leave the cache. The design documents call the cache "the single biggest win". They also note honestly that against the already-fast scalar
residual it is only about 1.3×; its big wins are against per-object loops.

### 3.4 The hybrid router and the width-reduced AAD block

One exotic trade used to push a whole book onto AAD. Now each residual row is routed individually:
- cacheable rows stay on the W-cache;
- only genuinely non-linear rows ride an AAD block: non-par MtM legs, compounded RFR details, rows reaching a MonotoneCubic region.

The block seeds derivatives only over the knots its rows touch, about 20 rather than 100, on in-object pooled dual numbers with no heap
allocation. A pooling threshold that was too small once cost 19,182 allocations per tick on the full-coverage `desk_mixed` rung; raising it gave
180.

### 3.5 Frozen-Newton streaming calibration

**What.** Each tick iterates `x ← x − M·r(x, q)` to a 1e-9 step tolerance, reusing a cached operator `M` built from an earlier Jacobian. `M` is a
Newton preconditioner, not a linear extrapolation. For a square, full-rank bundle the fixed point is the exact curve whatever `M` is, so ticks
are exact.

**Why not a linear update?** An early single-step update was accurate only to about 0.35 bp and re-anchored on about 20% of realistic ticks.

The mechanisms, each added for a measured reason:
- **Commit or fail.** A tick that hits a refresh, step or rescale cap, or diverges, is reported with a status and never committed, and the anchor
  is restored. A probe had once shown a failed tick leaving an anchor that later reported a curve 1,289 bp stale as "converged".
- **Refresh on staleness.** The Jacobian is recomputed when the frozen iteration stalls. Over-determined, banded or regularised problems also
  refresh on drift, because their frozen fixed point is exact only to second order.
- **Adaptive stall and predictive convergence.** Refresh when the contraction rate predicts that the remaining steps cost more than a refresh;
  stop early when the contraction bound puts the next step far below tolerance.
- **Rank safety.** `M` comes from a thresholded complete orthogonal decomposition, not an LDLT. On a singular problem the LDLT had walked
  unconstrained curves to negative forwards.
- **A two-threshold operator.** Weak directions present at the anchor are structural and kept. New weak directions met mid-walk are dropped, and
  the tick re-converges at full rank before committing. This was found on a pathological test market (section 6).

A realistic single-currency tick takes about 1.3 µs and the 70-instrument five-curve desk bundle about 22 µs. A mixed bundle with a
MonotoneCubic long end, which used to force a 215 ms cold solve, streams in milliseconds.

### 3.6 Huber soft-quote bands and the active-set walk

**What.** Any quote can carry a band: target, lower, upper, decay. The residual has slope `decay` inside the band and slope 1 outside,
continuous at the edges (a Huber loss). A hard pin is a zero-width band. A target outside its band is refused.

**Why.** Overlapping desk instruments (futures and swaps covering the same dates) cannot all reprice exactly. A band lets them settle inside
their bid/offer instead of fighting. The earlier Gaussian ramp gave two fits 110 bp apart in one knot. The Huber form is convex, and its Jacobian
row is a scalar times the quote gradient, so bands stay on the W-cache.

**How.**
- The streamer walks the convex piecewise-quadratic minimiser, predicting from the frozen operator which band edge is crossed first.
- Each slope change is a rank-one Sherman–Morrison update of the pseudo-inverse, not a refactorisation. That took a desk refresh from 6.4 to
  1.8 ms, pinned to 3e-14 against refactorising.
- A row walked onto its edge twice in one tick sits at a kink optimum. It is **pinned** as a stiff equality row, and its KKT multiplier is read
  from the converged gap. It is released only if the implied slope leaves [decay, 1], at most once per tick.

### 3.7 Turns

Year-end and quarter-end jumps are an additive overlay on log-discount factors: a jump size times a closed-form overlap with the turn window.
The overlap does not depend on the curve, so each jump is just one more W column; the Jacobian, streaming and risk inherit it at no cost. A turn
instrument pins the jump to a banded target, and turn states sit outside the smoothing penalty, so smoothing never erases the jump.

### 3.8 Tension-energy smoothing

**Why.** Par and basis quotes pin integrals of the forward curve, so wiggles that preserve those integrals are invisible to the fit. Basis-only
curves are genuinely rank-deficient; a global solve exposes this where a bootstrap hides it.

**How.**
- A constant penalty block approximates `∫(f″)² + σ²∫(f′)²`, justified by the tension spline being the minimum-energy interpolant.
- The block is folded into the streaming operator.
- Flat pieces have zero energy, so meeting-date steps survive.

**Measured.** On a basis-only EUR trio the condition number fell from about 1e18 to about 1e10, with no per-tick cost. The documentation says
plainly that it does not pin every long-end direction.

### 3.9 B-splines and moment-integrated averaging

- **B-spline region.** A clamped cubic whose unknowns are control points: C², no overshoot, still linear, so it stays on the cache.
- **Averaged legs.** A daily-averaged Fed Funds leg has one sub-period per business day. The moment path approximates it from the integral of the
  forward curve plus precomputed higher moments of the calendar's day counts. That prices an averaged leg at roughly the cost of a SOFR leg:
  about 9× faster ticks and 65× faster Jacobians than the exact daily path. It is labelled an approximation (about 3e-9 of the rate on real
  calendars), and the exact path remains.

### 3.10 Risk without bumping

- **Ladders.** Bucketed risk is `dx/dq = J⁺·D` from the calibration Jacobian: one calibration, one AAD gradient and a small solve, with no bump
  noise. It matches bump-and-recalibrate to about 2e-8, and parallel PV01 is a single directional dual.
- **Cross-bundle transform.** A transform matrix re-expresses a ladder between bundles with different knot layouts.
- **Consistent risk.** `generate_risk` risks one book across several bundles off the first bundle's discount factors, completing unquoted knots
  with self-quoted pillars so real pillars are not biased.

### 3.11 One API descriptor, generated interfaces

`api/api_surface.py` lists the engine's operations once. Generators emit from it:
- the C++ JSON dispatch;
- the Python (pybind) wrappers and typed stubs;
- the Excel/C functions over a single C-ABI symbol.

A drift check regenerates and diffs. Principle P14 keeps each verb a codec, and a gate that counts logic in the verb layer only allows that count
to fall.

### 3.12 The object model

Blueprint (editable) → CompiledModel (built once; only quote, band and fixing arrays change) → CalibratedState. One engine per compiled model,
shared by the session, the streamer and bound books.

The owner chose **invalidation by construction** over a runtime structure hash: only the quote arrays may change, and anything else is a new
model. A hash costs time and had proven gaps. Rebinding a session fell from 13.7 ms to 261 µs and stopped allocating.

### 3.13 Options, bonds, exposure (scoped)

- **Options.** Bachelier and full-β SABR with calibration and an arbitrage gate; swaption and vol-cube repricing; Garman–Kohlhagen FX with a
  delta-quoted smile.
- **Bonds.** A bond is data priced in curve space and street yield space; a whole universe solves as one batched Newton. The bond speed-up is
  712× against QuantLib's yield function, but the documents insist on quoting 125× against a correctly scaled QuantLib Newton, because QuantLib's
  solver there receives a mis-scaled derivative.
- **Exposure.** Labelled illustrative.

These were deliberately scoped down in September to prove the swaps-and-bonds core first.

### 3.14 Gates that cannot silently lie

The later programme's most distinctive output is not a pricing feature but a set of gates, each added after a gate was caught not doing its job:

| Gate | What it enforces | Why it exists |
|---|---|---|
| QuantLib oracle registry | Oracle tests have banners and locked assertion counts; the set of headers an oracle reaches may only grow | "Oracle" tests had drifted into comparing the engine with itself |
| Performance gate | Per-machine self-baseline (≤ 1.25×), absolute targets that only ratchet down, refuses to run on a busy CPU | The old gate was advisory, and baselines had been rewritten upward inside perf commits |
| Benchmark premises | A metric also asserts what its ticks did (e.g. `refreshes == 0`) | A square rung's "25 bp refresh tick" never refreshed; a time can be right for the wrong reason |
| Mutation gate | Curated real bugs are re-introduced into headers and the session layer; ≥ 90% must be caught | Tests passed with their feature deleted |
| Test taxonomy | Every test file is labelled oracle / calibration / parity / hot-path invariant / property / regression | Parity tests had pinned wrong values as "truth" |
| Verb density | Logic in the JSON verb layer, counted from the typed AST, may only fall | The API layer sat outside every oracle's reach |
| Hot-path census | Every streaming stage is stamped per tick, reached by a pinned scenario, and new refresh/factorisation call sites are locked | Audits kept finding stages added later with nothing pinning their cost |

## 4. How it was built: method and rules

**Principles (ratified 8 September; one line each):**

| Principle | Rule | Why |
|---|---|---|
| P0 | the fastest correct per-tick streaming path | — |
| P1 | one curve type | flavours are data |
| P2 | specifics as data | misses throw |
| P3 | the linear-map thesis | exactly two fingerprinted tiers |
| P4 | one templated implementation of each concept | — |
| P5 | hot-path discipline | no allocation, shape branching, virtual calls or fast-math per tick |
| P6 | measured, not assumed | native benches only |
| P7 | accuracy-first streaming | exact ticks, no uncommitted answers |
| P8 | QuantLib is an oracle, never a dependency | — |
| P9 | two gates | correctness never regresses; performance gated on self |
| P10 | performance integrity | one flag source, same toolchain, machine fingerprints |
| P11 | protected, honest tests | — |
| P12 | no reproduction, no bug | fresh-context review |
| P13 | the architecture map stays current | — |
| P14 | the verb layer is a codec | — |

**Working rules the owner set, and why:**
- **Reproduce before fixing, and mutation-gate every fix.** Several "bugs" dissolved on reproduction, and several parity tests pinned wrong numbers.
- **Fresh-context reviews, not self-grading.** An author's critique inherits the author's blind spots. Every major review used agents with no
  development history.
- **Quiet-machine verification.** The performance gate refuses to run under load, and commits wait for a quiet window. Background load had skewed
  timings, and a failed build once let tests "pass" against stale binaries.
- **Targets only ratchet down.** Loosening needs a written reason and owner sign-off.
- **Engine first; the web waits.** Web consequences of engine changes are recorded, then done once against the final engine.
- **Market realism in fixtures.** Several "performance problems" were test markets no desk would see: a sine-pattern 25 bp move whose optimum put
  a 30-year rate negative, and FX forwards bumped by a hidden 5 bp rate move. Fixtures were fixed rather than tuned around.
- **Research stays on branches.** Parallel agents research and draft; only the main line builds and commits, after a quiet full verification.

**The owner's domain knowledge shaped the core model.** Bands as bid/offer freedom, portfolio instruments, turns as instruments, one outright
curve per sub-bundle, cross-currency curves keyed by FX pair, and separating instrument, quote and market all came from the owner, often
pushing back on a first draft. Many errors were also caught by the owner rather than by tests: missing meeting jumps, legacy day counts, a live
performance regression, and gates that measured but never blocked. That history is why the later work weighs honest gates so heavily.

## 5. Selected measured results (from commit messages)

| Result | As recorded |
|---|---|
| Curve build vs QuantLib GlobalBootstrap | 16.10 ms → 2.35 ms (6.85×) |
| Delta ladder vs bump-and-reprice | 27.6 ms → 0.70 ms (39×) |
| 1,000-swap book reprice vs QuantLib NPV loop | 23.9 ms → 0.19 ms (~126×) |
| 4-curve bundle vs QuantLib iterative bootstrap | ~12.7 ms vs 258 ms (~20×) |
| Analytic vs AAD Jacobian | 373 µs → 24.8 µs (~15×) |
| Cross-currency streaming tick after moving FX/MtM onto the cache | 129 µs → 13 µs |
| Compiled vol surface vs QuantLib lean loop | 2.3 µs vs 10.4 µs |
| Pooled-dual AAD Jacobian | 246 µs → 47.4 µs |
| Session rebind (object model) | 13.72 ms → 261 µs, 0 allocations |
| One-shot PV01 | 71,341 → 387 allocations |
| Like-for-like 4-curve ±0.3 bp tick vs QuantLib re-bootstrap | 4.3 µs (663×) |
| Mixed bundle: cold solve → streamed tick | 214.7 ms → 4.98 ms |
| Allocations per refresh (September hot-path work) | 13 → 3; `desk_mixed` 54 → 31 |

**Noted honestly in the history.**
- The AVX-512 path measured no better than AVX2, because the bottleneck is memory bandwidth, so AVX2 became the default.
- A reciprocal-DF kernel change made no measurable difference and was recorded as such.

## 6. Bugs that taught something

| Symptom | Root cause | What changed |
|---|---|---|
| Parallel shocks 2–3× too large on spread curves | the shift was added to every curve's own knots, and a spread's forward is base + spread | one `parallel_direction` definition |
| Averaged overnight rates +1.389% | per-day weight divided by curve time instead of the index day-count fraction; a parity test pinned the wrong value | one golden source per fundamental; parity is not truth |
| A futures month −25 bp | a weekend-opening averaging window dropped its first fixing | found by the first calibration-oracle run |
| A streamed tick reported "converged" on a 1,289 bp stale curve | a failed tick left its anchor at a divergent point | commit-or-fail tick contract |
| NaN quotes returned the seed curve as success | a solver status was never read | non-finite input throws; status at every seam |
| 27% NPV error after a rebind | three engines per session, blind to each other's updates | one shared engine (object model) |
| Resolve returned to the old market after streaming | a converged tick never stored its quotes in the session | quotes committed before solving |
| Divergence on 25 bp requotes | a pathological test market plus a near-singular MonotoneCubic direction | two-threshold operator; a realistic test move |
| A refresh on every noisy tick of the mixed bundle | a Newton 2-cycle across two MonotoneCubic slope-clamp kinks | diagnosed and parked (section 8) |

## 7. The web product, SDK and Excel add-in

A sibling repository wraps the engine as a product:
- a composer web app: compose curves from generic instrument rows and interpolation regions, calibrate, and stream them against a simulated
  market over a WebSocket, with streaming "tile health" per frame;
- risk, options, bonds, exposure and scenario pages;
- a Python SDK;
- an Excel add-in.

The engine is a git submodule, and the web's spec compiler is mirrored by the engine's C++ compiler. Every binding is generated from the
engine's single API descriptor.

Distinctive product decisions:
- one quote model (target/upper/lower/decay) everywhere;
- content-hashed static assets, after a browser once ran a stale bundle against a new server;
- a deploy preflight that imports the server from the container's exact file set.

Web work was frozen on 9 September by the engine-first rule. The changes it will need, including two known wrong numbers in its averaged
overnight observations, are recorded for one scoped update against the final engine.

## 8. Current state, open work and research

**State at 15 September 2026.** The working branch carries the E0–E7 programme and the streaming hot-path work, with every gate green.

**Open on the main line:**
- reusing the previous tick's final anchor (a refresh-schedule change awaiting sign-off);
- the remaining hybrid-evaluation allocations;
- a "free" first Newton step;
- the mixed bundle's clamp 2-cycle and a walk divergence at ±1 bp;
- an explicit pseudo-inverse;
- copy-on-write model forks and fixings as scalar updates.

**Parked research** (branch `research/aad-graph-kernels`, not for merging):
- **The idea.** Record the templated pricing once as an AAD graph and execute residuals and Jacobians through pre-compiled kernels driven by
  data, so the hot path is generic instead of hand-written per instrument.
- **Findings.**
  - No JIT is needed.
  - Grouped kernels cut routine dispatch from about 15,000 to under 200 per evaluation on the desk bundle, with zero allocation and
    bit-identical results under strict floating-point contraction.
  - A "cashflow table" instrument definition covers 97% of benchmark rows with no exceptions.
  - Value-dependent branches (the MonotoneCubic limiter) are best handled as compute-both-then-select, and their mask bits could feed the
    streamer's active set.
- **Not measured.** The timing comparison against the hand-written kernels. The branch's `research/PARKED.md` records what was and was not
  measured.

**Sharing.** The engine builds and tests without QuantLib and without the web repository (`-DSWAPS_ALLOW_NO_ORACLE=ON`, see the README). The
existing git history contains working-session material that should not be shared, so an outside release should come from a fresh, cleaned export.
