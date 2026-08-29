# OBJECT_MODEL_PLAN.md — the region-first object-model rework

The competitive review + full target model live in the artifact **"Modular by Region"**
(https://claude.ai/code/artifact/9c7c3385-dc25-4151-be36-569c3cbb6168). This is the engineering
anchor: the settled model, the governing perf constraint, the phased plan, and the baseline to beat.

## Governing constraint (read first)

The clean object model lives **only in the definition/authoring layer**. The compiler collapses it into
the existing flat, alloc-free SoA structures the hot path already runs on. **The calc kernel
(`curve/`, `pricing/`, `calibration/` W-cache + AAD, `portfolio/`) stays byte-identical**; only the
`build/` (spec) + `api/compile` seam changes. OOP ends, and flat structures begin, at "spec → compiled."

**No unnecessary malloc in the hot path.** Every phase is proven regression-free against the perf gate.

## Settled model (from the design dialogue)

Three kinds of thing — keep them distinct:

- **Definitions (specs)** — immutable, serializable, hashable, *compiled once* → the W-cache:
  `Index` · `Convention` · `Instrument` (incl. `Turn`) · `Region` · `CurveSpec` · `ModelSpec` · `VolSurfaceSpec`.
- **Market quotes** — the only thing that changes tick to tick.
- **Calibrated outputs** — the solved, forkable state you price/risk: `Curve` · `Model` · `VolSurface`.

Naming convention: **`<Thing>Spec` for the blueprint, bare noun for the calibrated output** (you *price* a
`Model`, *sample* a `Curve`). Pure definitions with no solved twin keep bare names (`Index`, `Region`, …).

Key relationships:
- An **`Instrument` is curve-unbound**: it references its projection **`Index`(es)** (which also give it
  conventions, via a **`Convention`** preset) and carries **no curve**. Role *shape* is on the instrument;
  role *binding* (`Index→Curve`) is on the `Model`. Discounting is the `Model`'s CSA choice. One instrument
  is reusable across every build; given a target quote it calibrates, given a notional it prices.
- A **`Curve` realizes an `Index`** inside one `Model`; several models realize the same index differently.
- A **`Region` is first-class**: it owns its interpolation scheme, **its own smoothing/tension**, its
  boundary-continuity policy, its span, and a stable id. Per-region smoothing is the differentiator no
  surveyed product (QuantLib, ORE, rateslib, finmath, Deriscope, Bloomberg, FINCAD, Numerix, Murex) offers.
- A **`Model` is immutable and forkable**. A **state fork** perturbs the calibrated output (shock `x`, rebind
  discount) — nearly free over the W-cache (it *is* what the exposure kernel does). A **structure fork** forks
  the *spec* and recalibrates. The live streaming session is the one mutable head; forks are frozen snapshots.
- **`VolSurface`** is the vol-of-an-index twin of `Curve`; **inflation** is just an `Index` variant realized
  as a price-index curve. Neither adds a top-level object.

## Baseline to beat (engine ff916f8, 2026-08-29; re-baseline quiesced for exact figures)

| metric | ours_ns | role in the rework |
|---|---|---|
| `curve_build` | 615,300 | the spec→flat compile path — the OOP-collapse tripwire |
| `warm_recalibration` | 20,304 | streaming hot path — must stay µs |
| `risk_full_jacobian` | 105,451 | AAD risk sweep |
| `portfolio_analytics` | 91,395 | book reprice + risk |
| `vol_cube_warm` | 11,774 | vol path |
| `bond_sweep` / `bond_book` | 3,897,567 / 458,137 | bond analytics |

All correctness + oracle gates PASS at baseline. **Every phase must re-pass `tools/verify.sh` (both gates)
and hold these numbers** (a new `bench_spec_compile` may be added to guard the compile path explicitly).

## Phased plan (engine-first; kernel untouched; gate green each step)

### DONE
- **Phase 0 — One wire representation.** `bundle_to_json` normalises to `regions` via `modules()`;
  `compile.py._curve_layout` always emits regions; `bundle_from_json` still reads `meeting`/`back` (old data
  loads). Bit-identical (compile-parity + oracle; web A/B max|Δfwd|=0.0). Engine `b234056`, web `4e4e8fb`.
- **Phase 1 — Per-region CURVATURE smoothing (the differentiator, curvature path).** `CurveModule.reg_lambda`;
  `second_difference_operator` + `SmoothedProblem` weight each row by its centre knot's region λ, via
  `region_knot_lambdas` (uniform λ ⇒ byte-identical). Round-trips through bundle JSON (emitted only when set).
  `RegionSmoothing` tests prove per-region + backward-compat; 296/296, oracle green; warm/curve_build under
  baseline. Engine `a276d43`, web `4153154`. (The tension-energy default penalty is finished in Phase 2.)
- **Phase 2 — Per-region smoothing complete (tension-energy path).** `CurveModule.reg_sigma`;
  `curve_tension_stiffness` assembles K per INTERVAL using each piece's region (midpoint→region), a relative
  weight ρ=reg_lambda/default and the region's σ: `K += ρ²·(K2p + σ²·K1p)`. Inheriting ⇒ K unchanged ⇒ outer
  `weight` reproduces the old operator byte-for-byte (eigen-rank tolerance held). `RegionSmoothing.
  TensionEnergyIsPerRegionSigma` proves per-region σ (+weight); A1/A2/A3 + TensionRegularizerOracle green,
  301 tests; setup-only, warm/curve_build within noise. Engine `cda0923`, web `13563e6`. **Per-region
  smoothing now complete for BOTH penalty modes — the differentiator no surveyed product offers.**
- **Phase 3 — First-class reference-data objects (engine-objects-first).** `build/ref_data.hpp`: typed
  VALUE objects `Calendar`/`DayCount`/`Index`/`Convention` over the string-keyed free functions + the
  constexpr DB; the `Index` is the pure pivot the model references (its own conventions, no curve). Each
  method DELEGATES to the parity-tested free functions ⇒ additive, header-only, off every hot path; oracle/
  compile-parity untouched. `RefData` tests pin delegation; 301/301. Engine `9f9c039`, web `5e4f135`.
  (Chose engine objects only — the full C++/Python de-duplication, which would collapse the independent-
  implementation parity check, is deferred.)

### REMAINING (larger, product-facing — a checkpoint before the saved-spec migration is wise)

- **Phase 4 — Curve-unbound instruments + the spec/output object split.** An `Instrument` references its
  projection `Index` (not bundle curve indices); the compiler DERIVES knot/region placement (retire
  `adds_knot`/`knot_region` from instrument rows); introduce the `CurveSpec`/`ModelSpec` definition objects
  distinct from the calibrated `Curve`/`Model`; `Index→Curve` binding lives on the Model. Folds in the
  Phase 0b cleanup (retire the now-vestigial `CurveStructure.meeting/back` fields + the `CalibrationProblem`
  legacy). Reusable Instrument + build-time role binding → compiled to the fast index struct (OOP collapses
  to flat). *Gate:* compile-parity; oracle; a test proving ONE instrument reused across two builds; the
  `curve_build` compile path within baseline. *Deliverable:* reusable instruments; clean Spec objects — the
  "generic building blocks, reusable" North Star realised.
  *Scout findings (blast radius):* (a) curve-unbinding is a **web-layer remap** — engine legs stay int
  indices (`problem.hpp` `FloatLeg.forecast/discount`, `Instrument.fx_num/turn_curve`), the wire format is
  unchanged; the compiler translates Index→int. (b) Retiring `meeting/back` is mechanical but WIDE:
  `CurveStructure.meeting/back` (~12 construction/test sites) plus the separate `CalibrationProblem.
  meeting_times/back_times` twin (~15 more test files); every positional aggregate init `{meeting,back,base}`
  must be rewritten, not reordered. (c) **Clean-break migration** (chosen): saved specs are JSON blobs in
  `engines.spec`, recompiled on open — old blobs relying on `adds_knot`/`knot_region` will break, so existing
  saved engines need rebuilding. (d) `CurveStructure` is the shared pricing↔calibration fulcrum — a prior
  duplicate was deliberately folded in, so the `CurveSpec`/`ModelSpec` split must NOT re-fork it (highest
  regression risk of the whole plan; keep the definition/output split at the compile boundary, not by
  cloning the struct).

- **Phase 5 — Named handles + the forkable Model.** Make the definition/output split concrete —
  `ModelSpec.calibrate(quotes) → Model`; the immutable, forkable `Model` (state-fork over the W-cache =
  nearly free, structure-fork of the spec); a named-handle object surface (optional name → auto type-handle)
  across the web composer, then Excel/Python to parity. *Gate:* exposure/scenario expressed as fork-per-state
  (reuse the exposure kernel); state-fork nearly free; interface smoke. *Deliverable:* forkable models +
  named-handle UX — the commercial polish.

### AFTER
- **Phase 6 — Beyond rates.** `VolSurface` (the vol-of-an-index twin of `Curve`, forking the same way) beside
  the curves; inflation as an `Index` variant realised as a price-index curve. Slots into the Model; no new
  top-level object.

## Verification discipline

`tools/verify.sh` (both gates) after every phase. Perf-sensitive phases (0, 1) additionally re-run the
`curve_build`/`warm_recalibration`/`risk_full_jacobian` benches on a quiesced box and record before/after in
this file's changelog. Never route a perf check through JSON — native google-benchmark only.
