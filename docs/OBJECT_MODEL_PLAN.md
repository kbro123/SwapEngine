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

- **Phase 0 — Retire the legacy front/back topology.** Collapse the parallel `meeting`/`back` representation
  (`pricing/curve_spec.hpp` `CurveStructure`, `curve/curve_module.hpp` `flat_hermite`) into the single region
  list; remove the classic/modular fork in `server/compile.py` (`_curve_layout`). One curve model, not two.
  *Gate:* oracle bit-identical; `curve_build`/`warm` unchanged. Pure simplification, no new capability.

- **Phase 1 — Promote `Region` to first-class + per-region smoothing.** Widen `CurveModule` to own smoothing
  (λ + mode + energy σ), boundary policy, span, id; make construction a uniform parameter-bag (retire the
  `TensionHolder` special-case); rewrite the three penalty-operator builders in `calibration/regularize.hpp`
  to loop **per-region knot blocks** with per-block λ/σ; demote `RegSpec` to a mode/override. End the σ-name
  collision. *This is where per-region smoothing is born.* *Gate:* oracle green; `warm`/`risk`/`curve_build`
  within baseline; add a bench proving a two-region-different-smoothing curve calibrates without a hot-path
  malloc regression.

- **Phase 2 — First-class, single-source reference data.** `Calendar`/`DayCount`/`Convention`/`Index` as real
  objects defined once, ending the C++↔Python double-maintenance (DB stays source of truth; objects are the
  typed surface). *Gate:* conventions-sync + oracle green.

- **Phase 3 — The spec contract + named handles.** Move knot/region derivation into the compiler so
  instruments stop carrying `adds_knot`/`knot_region`; introduce `CurveSpec`/`ModelSpec` + the curve-unbound
  `Instrument` (references `Index`, bound at build); named-handle object surface (optional name → auto
  type-handle). Web composer first, then Excel/Python to parity. *Gate:* compile-parity vs `compile.py`.

- **Later — Model tree + vols/inflation.** Make `Model` immutable/forkable (state-fork over the W-cache;
  structure-fork of the spec); `VolSurface` beside curves; inflation as an `Index` variant. Built *on* the
  core; does not touch Phase 0/1.

## Verification discipline

`tools/verify.sh` (both gates) after every phase. Perf-sensitive phases (0, 1) additionally re-run the
`curve_build`/`warm_recalibration`/`risk_full_jacobian` benches on a quiesced box and record before/after in
this file's changelog. Never route a perf check through JSON — native google-benchmark only.
