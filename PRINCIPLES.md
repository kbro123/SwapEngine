# PRINCIPLES.md — SwapEngine (the law)

Ratified 2026-09-08. `CLAUDE.md` is build facts and history; `ARCHITECTURE.md` is the map; **this file is the
law**. Every change is judged against it. Change this file only by explicit decision, in its own commit.

## Purpose

**P0.** The engine exists for one thing: **maximum speed in the hot path for streaming rates instruments** —
swaps first, then bonds, then options — without giving up correctness. A stale price is a free option
handed to the client; a wrong price is worse. Every design choice serves the per-tick path.

## Model

**P1. One curve type, generic building blocks.** `ModularCurve` is an ordered list of interpolation regions.
The schemes {Flat, Linear, NaturalCubic, Hermite, MonotoneCubic, BSpline, Tension} compose in any order and
position; there is no front/back concept; region 0 flat-extrapolates its first knot. A curve flavour is data
(a module list), never a subclass. Prefer a generic composable abstraction over a per-case branch.

**P2. Specifics as data.** Market conventions — day counts, calendars, frequencies, spot/pay/fixing lags,
index and product definitions, recovery rates, contract specs, central-bank schedules, default currencies
and indices — live in `conventions/conventions.json` and flow through the code. A new product, index or
currency is a data entry. **Zero index, currency or calendar identifiers appear in engine code**, and a
lookup that misses **throws**; it never falls back to a default silently. The baked DB is the *default*
set; a runtime registry lets any API add or override entries without rebuilding.

**P3. The linear-map thesis.** `∫f = w(t)·x` is linear in the knot forwards with structure-only weights, so
`DF = exp(−W·x)` with `W` built once. Everything rides this: cache `W` per instrument, reprice as `L = W·x`,
analytic Jacobian `−(∂r/∂DF·diag DF)·W`, implicit-function-theorem risk. There are exactly **two tiers**: the
compiled W-cache tier, and the pooled-AAD block for non-cacheable rows (MtM funding legs, FX-in-portfolio) and
as the generic fallback. A value-dependent scheme rides the compiled tier too, with `W` RE-TAKEN AT THE STATE --
cell-constant for a piecewise-linear scheme (MonotoneCubic), the scheme's closed-form TANGENT otherwise
(MonotoneConvex, 2026-10-31) -- and the log-discount kept exact; a scheme reaches the compiled tier only by
supplying one of the two, pinned against AAD, which stays the oracle for every analytic derivative. A row's tier
is explicit and fingerprinted.

**P4. One templated implementation, no duplication.** Scalar-templated kernels give value and Jacobian
from one code path. ONE rate formula, ONE float primitive (`BundleFloatBatch`), roles on legs, ONE residual
definition serving every problem type, ONE band/target hand-off, ONE risk operator. Residual order is THE
contract. If a concept is defined twice, one of them is a bug waiting to diverge.

## Hot path

**P5. Hot-path discipline.** Per tick: nothing allocated (any allocator, not just Eigen), nothing branched
on data shape, no virtual dispatch, no QuantLib objects, no per-instrument loops where a batched operation
exists. Never hard-code a SIMD width. SoA layout. Deterministic run-to-run (no `-ffast-math`). Fused fast
paths are measured and must not regress. **The scope of "allocation-free" is exactly what the T4 invariant
tests prove** — no claim beyond that.

**P6. Measured, not assumed.** Every performance statement in code, docs or commit messages comes from a
committed benchmark. Every engine change that touches a hot path includes or refreshes a native C++
google-benchmark; perf tests are **never** routed through JSON or a web layer. The engine's perf gate is
the engine's own benches on the engine's own build — a web deployment has its own latency budget and never
gates the engine.

**P7. Accuracy-first streaming.** The default streaming mode is exact every tick. The Jacobian is
recomputed only on genuine staleness. A tick that does not converge is **not committed** and is reported.
Structure changes are detected by a **complete** fingerprint (every field the compiled engine bakes).

## Truth

**P8. Oracles.** QuantLib is the correctness oracle and the informational reference — it is **not** a
dependency of the shipped engine. Pricing and interpolation match QuantLib to `rel ≤ 1e-10`, deterministically,
with no optimizer involved, through the `YieldTermStructure` adapter (every scheme, every quote kind).
Calibration is solver-dependent: assert first-order optimality, a committed golden objective and agreement
with an independent optimizer — never `1e-10`. AAD vs bump: `1e-6`. Batched analytics vs per-instrument
QuantLib: `1e-10`. A test earns the word *oracle* only by comparing an engine number to a QuantLib number;
everything else is labelled for what it is (calibration, cross-path parity, invariant, property, regression).

**P9. Two non-negotiable gates.**
- *Correctness*: never regresses. A failing test blocks the checkpoint; a skipped or gutted test is a failure.
  The gate cannot pass with QuantLib absent, with a shrunken test count, or with assertions removed.
- *Performance*: gated against **ourselves and against absolute targets**, not against QuantLib.
  (a) Self-baseline: each metric must be within **1.25×** of the committed number for this fingerprint,
  measured min-of-N on a quiesced machine (the gate refuses to run under load rather than warn).
  (b) Absolute desk-scale targets in `baselines/targets.json`; targets only ever **ratchet down**.
  (c) QuantLib (and QuantLib+XAD, rateslib) comparisons are an informational reference bench, run nightly,
  published with losses as well as wins — they inform the commercial story and never gate.
  Curve build, when compared, is compared to `GlobalBootstrap` (same algorithm class), never `IterativeBootstrap`.

**P10. Perf-gate integrity.** One source of truth for compiler flags (`cmake/DetectISA.cmake`); every
reference library is built with the engine's exact toolchain and that toolchain is part of the baseline
fingerprint. The gate refuses to compare across fingerprints. Medians of repetitions, `DoNotOptimize`,
load-average check before every run.

**P11. Tests are protected and honest.** Never delete or weaken a test to make a change pass — reproduce
the comparison instead. Tolerances have one source of truth (`tests/tolerances.hpp`) and change only with a
documented reason. A regression test is accepted only when shown to fail on the reverted bug. Goldens come
from independent sources (QuantLib, Rateslib, hand derivation, textbook literals), never from the engine's
own output unless explicitly labelled a regression freeze with a reason. A curated mutation set runs nightly.

**P12. No repro, no bug; no fresh eyes, no review.** A bug is accepted with an executable reproduction.
Reviews are done from a fresh context, not by in-context self-ranking.

**P13. The map stays current.** `ARCHITECTURE.md`, the test registries and this file's cross-references
update in the same commit as the change they describe. A stale map is worse than none.

**P14. The verb layer is a codec.** A `run_json` verb body is parse -> ONE call into `include/swaps/**` -> emit.
No convention lookups, no defaults for absent fields, no result-building loops, no arithmetic, no branching
beyond decoding. Behaviour lives in the library, where the unit tests and the oracles already reach it; a verb
that needs something the library cannot do gets a library function first. A default is declared ONCE, on the
library struct, never re-stated in a decoder. `tools/verb_density.py` (verify.sh) counts what each api file still
carries from clang's typed AST and locks it in `tests/verb_density.lock`: a count may only go down (E7).

**P15. Every curve has a front anchor, and the anchor is the curve's own fixing.** A curve's first knot is pinned by
an instrument that reads THAT curve's forward over the window of its index tenor starting at the front -- the 1D fixing
of an overnight curve (ESTR, SOFR), the 3M fixing of a 3M tenor curve (2026-11-02: never a 1W deposit, a 1M future or a
1Y OIS standing in for the 1D fixing); for a spread curve its 1D fixing at zero spread over the base, since the
overnight fixing is the same whoever clears it; an xccy collateral curve, which has no index, its FX forward from the
value date. Never the leading region's flat extrapolation of a term
pillar, and never a discount-only read (a par rate's discounting column vanishes at the front). The state at every
knot is a forward; the front one is a market quote like any other. `calibration::front_anchors` (`bundle_problem.hpp`) is the check and the `compile` verb warns on
an unanchored curve; `tests/front_anchor_rule_test.cpp` / `front_anchor_reference_test.cpp` walk every fixture in the repo (ratified
2026-11-01, after the G4 desk's CCP curves were found carrying a 2Y spread from day one; ASSUMPTIONS K53).

## What "done" means

A capability is done when: it is on the hot path it claims (P3/P5), benchmarked (P6), oracle-pinned or
labelled otherwise (P8), gated (P9), data-driven (P2), mapped (P13) — and every claim about it in `README.md`
or `CLAUDE.md` points at a named test or bench that proves it.
