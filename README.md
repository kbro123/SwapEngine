# SwapsEngine

A high-performance interest-rate **swap-curve calibration engine** and **vectorized swap-analytics
library**, built to be provably faster and more accurate than stock QuantLib on the workflow it targets.

## Core ideas

- **Global calibration** of all knot forwards via **Levenberg–Marquardt** (not sequential bootstrapping).
- **Analytic Jacobian via forward-mode AAD** (Eigen `AutoDiffScalar`), reused for **analytic bucketed
  risk** through the implicit-function theorem — no bump-and-reprice.
- **Multi-region forward interpolation**: piecewise-flat forwards on **central-bank meeting dates** in
  the front end; a **local C¹ Hermite** spline beyond the last meeting date. The generic
  `MultiRegionCurve` composes any linear-in-values region policies (Flat/Linear/NaturalCubic/Hermite).
- **Multi-curve bundle**: N curves (e.g. SOFR + Fed-Funds + Prime …) calibrated **simultaneously** over
  one stacked parameter vector, with forecast ≠ discount pricing and basis-swap chains.
- **Real-time streaming re-calibration**: an exact frozen-Newton path reprices to the market every tick.
- **Vectorized portfolio analytics**: par rates, NPV, PV01/DV01, bucketed delta as batched Eigen
  algebra — no per-swap loops.

QuantLib **is a linked dependency**: the engine reuses its calendars, schedules, instruments and
conventions wholesale, and extends only the two hot workflows (curve calibration + bulk analytics).
QuantLib is **also** the **correctness oracle** and **speed baseline** we must beat.

## Build

Dependencies are vendored into `third_party/` (this machine is Homebrew "Tier 3" — no bottles).

```bash
./tools/bootstrap_deps.sh   # one-time: fetch + build deps (resumable). ~30-50 min.
./tools/verify.sh           # runs correctness + performance gates
```

See [CLAUDE.md](CLAUDE.md) for the full architecture, rules, and the two-gate verification policy.

## Status

**Phases 0–6 + Stage 2 (streaming) + Stage 3 (multi-curve bundle) complete.** Both gates green:
correctness against the QuantLib `YieldTermStructure` oracle, and a fingerprint-keyed performance gate.
Representative speedups vs QuantLib (same compiler/flags): curve build ~5×, bucketed risk ~32×,
portfolio reprice ~150×, multi-curve bundle build ~20× vs IterativeBootstrap.

See the roadmap in [CLAUDE.md](CLAUDE.md#8-phased-roadmap).

## License

The shipped engine is original work. QuantLib (BSD) and Eigen (MPL2) are used as build/dev
dependencies only.
