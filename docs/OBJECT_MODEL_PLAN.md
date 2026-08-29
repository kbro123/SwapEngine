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

### NEXT FOUR

- **Phase 2 — Complete per-region smoothing (tension-energy path).** Extend per-region weighting to the
  DEFAULT penalty (`calibration/regularize.hpp` `tension_energy_operator` / `curve_tension_stiffness`): vary
  the per-region weight/σ **per element** inside the stiffness assembly, respecting the C0 cross-join coupling
  (a region's energy touches its neighbour's boundary forward). Add a per-region reg-σ field (or reuse
  `reg_lambda` + a mode); demote `RegSpec`'s global σ to a default. *Gate:* `TensionRegularizer` + oracle
  bit-identical when uniform; a per-region-σ unit test; setup-only, warm unchanged. *Deliverable:* per-region
  smoothing complete for BOTH penalty modes.

- **Phase 3 — First-class, single-source reference data.** Promote `Calendar`/`DayCount`/`Convention`/`Index`
  from string-keyed free functions (`build/*.hpp`, transcribed again in `server/calendars.py`/`conventions_db.py`
  and kept in sync by parity tests) to real typed objects over the `conventions.json` DB (still the source of
  truth); the web binds the same objects, thinning the Python duplicates to wrappers. *Gate:* conventions-sync
  + the C++↔Python parity tests confirm equivalence (then simplify); oracle green. *Deliverable:* one
  reference-data source — a new instrument/index is a DB entry, not code in two places.

- **Phase 4 — Curve-unbound instruments + the spec/output object split.** An `Instrument` references its
  projection `Index` (not bundle curve indices); the compiler DERIVES knot/region placement (retire
  `adds_knot`/`knot_region` from instrument rows); introduce the `CurveSpec`/`ModelSpec` definition objects
  distinct from the calibrated `Curve`/`Model`; `Index→Curve` binding lives on the Model. Folds in the
  Phase 0b cleanup (retire the now-vestigial `CurveStructure.meeting/back` fields + the `CalibrationProblem`
  legacy). Reusable Instrument + build-time role binding → compiled to the fast index struct (OOP collapses
  to flat). *Gate:* compile-parity; oracle; a test proving ONE instrument reused across two builds; the
  `curve_build` compile path within baseline. *Deliverable:* reusable instruments; clean Spec objects — the
  "generic building blocks, reusable" North Star realised.

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
