# SwapsEngine

A high-performance interest-rate **swap-curve calibration engine** and **vectorized swap-analytics
library**, built to be provably faster and more accurate than stock QuantLib on the workflow it targets.

## Core ideas

- **Global calibration** of all knot forwards via **Levenberg–Marquardt** (not sequential bootstrapping).
- **Analytic Jacobian via forward-mode AAD** (Eigen `AutoDiffScalar`), reused for **analytic bucketed
  risk** through the implicit-function theorem — no bump-and-reprice.
- **Two-region forward interpolation**: piecewise-flat forwards on **central-bank meeting dates** in
  the front end; **C²-smooth** forwards beyond the last meeting date.
- **Vectorized portfolio analytics**: par rates, NPV, PV01/DV01, bucketed delta as batched Eigen
  algebra — no per-swap loops.

QuantLib is used only as the **correctness oracle** and **speed baseline**; it is not linked into the
shipped engine.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./tools/verify.sh          # runs correctness + performance gates
```

See [CLAUDE.md](CLAUDE.md) for the full architecture, rules, and the two-gate verification policy.

## Status

Phase 0 (scaffolding). See the roadmap in [CLAUDE.md](CLAUDE.md#8-phased-roadmap).

## License

The shipped engine is original work. QuantLib (BSD) and Eigen (MPL2) are used as build/dev
dependencies only.
