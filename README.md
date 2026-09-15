# SwapsEngine

A high-performance **multi-curve interest-rate calibration engine** and **vectorized rates analytics library**, in header-only
C++20. It is built to be measurably faster than stock QuantLib on the workflows it targets, while matching it to tight tolerances.
QuantLib is used as the correctness oracle and the speed baseline, not as a runtime dependency.

How and why it was built, and what is unusual about it, is told in [docs/HISTORY.md](docs/HISTORY.md). The module map is
[ARCHITECTURE.md](ARCHITECTURE.md), and the engineering rules are in [PRINCIPLES.md](PRINCIPLES.md).

## Core ideas

- **Global calibration** of every curve's knots together, by Levenberg–Marquardt, rather than sequential bootstrapping. N curves
  (e.g. SOFR + Fed Funds + ESTR + EURIBOR + a cross-currency curve) share one stacked state, with forecast ≠ discount pricing,
  basis chains, FX forwards and MtM cross-currency swaps.
- **Analytic Jacobians and risk**:
  - A compiled "W-cache" (DF = exp(−W·x)) handles linear curve schemes.
  - Width-reduced forward AAD handles value-dependent ones.
  - Bucketed risk comes through the implicit-function theorem, with no bump-and-reprice.
- **Order-agnostic interpolation regions**:
  - One curve type, `ModularCurve`, is an ordered list of regions.
  - Flat, Linear, NaturalCubic, Hermite, BSpline, MonotoneCubic and Tension compose in any order.
  - A curve "flavour" is data chosen at runtime.
- **Conventions are data, not code.** Day counts, calendars, frequencies and lags live in `conventions/conventions.json`, which is
  compiled into a generated header. A new product or index is a data entry.
- **Real-time streaming re-calibration.** A frozen-Newton calibrator re-solves to the exact curve every tick. It reuses the
  Jacobian until it goes stale, walks soft-quote (Huber) bands by rank-one operator updates, and allocates nothing on the tick
  path of the compiled shapes.
- **Honest gates**:
  - a QuantLib oracle;
  - a fingerprint-keyed performance gate with targets and workload premises;
  - a mutation gate, a test taxonomy and a hot-path census.
- **One API descriptor** (`api/api_surface.py`) generates the JSON dispatch, and the web product's Python, `.pyi` and Excel
  bindings. The C ABI is a single `swaps_run_json` symbol.

## Building

Two supported builds:

| | QuantLib-free build | Full gated build |
|---|---|---|
| Builds | the engine headers, the API library (`libswaps_api`), the C-ABI shared library, the CLI, `swaps_tests`, `swaps_allocfree_tests`, `swaps_api_tests`, and the non-QuantLib benches | everything on the left, plus the QuantLib oracle and consistency test binaries, and the QuantLib reference benches |
| Verification | `ctest` | `tools/verify.sh`: guards, correctness gate and performance gate |
| Needs | a C++20 compiler, CMake ≥ 3.20, Eigen 3.4, Boost headers, GoogleTest, Google Benchmark | the same, plus QuantLib 1.35 built with the engine's own flags |

### Dependencies

| Dependency | Needed for | Version the project pins |
|---|---|---|
| C++20 compiler | everything | Apple clang 15+ (the reference machine); GCC 13 (the Linux CI, see below) |
| CMake (+ Ninja, optional) | everything | 3.29.6 / 1.12.1 |
| Eigen | the engine (required) | 3.4.0 |
| Boost headers (Boost.JSON, header-only) | the API layer: JSON codec, `BundleSession`, CLI, C ABI | 1.84.0 |
| GoogleTest | tests | 1.14.0 |
| Google Benchmark | benches | 1.8.4 |
| QuantLib | ONLY the oracle tests and reference benches | 1.35 |

CMake looks for dependencies in `third_party/` (git-ignored) first:
- `third_party/eigen`;
- `third_party/boost` (a directory holding `boost/json.hpp`);
- `third_party/gtest/install`;
- `third_party/benchmark/install`;
- `third_party/quantlib/install`.

Eigen, GoogleTest and Benchmark can also come from the system. **The API layer is built only if `third_party/boost` exists.** With
system Boost, symlink it: `mkdir -p third_party && ln -s /usr/include third_party/boost`.

### Build without QuantLib

This builds the engine, API and tests from a fresh clone, without QuantLib. The recipe was verified on 2026-09-15 on a clean clone
(macOS, Apple clang): 687 tests passed, and the CLI and a bench ran.

```bash
# 1. Dependencies: system packages (Ubuntu shown) or copies under third_party/ (see the table above).
sudo apt-get install -y ninja-build cmake g++ libeigen3-dev libgtest-dev libbenchmark-dev libboost-dev
mkdir -p third_party && ln -sfn /usr/include third_party/boost      # the API layer looks for third_party/boost

# 2. Configure. Without QuantLib, configuration stops with a fatal error unless you opt out of the oracle explicitly:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSWAPS_ALLOW_NO_ORACLE=ON
#   (dependencies under third_party/: add -DCMAKE_PREFIX_PATH="third_party/gtest/install;third_party/benchmark/install")
#   (Linux: add -DSWAPS_BUILD_BENCH=OFF -- bench/hotpath_census.cpp uses a macOS-only malloc hook)

# 3. Build and test
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Notes:
- Allocation-count tests use a macOS allocator hook. On other platforms they `GTEST_SKIP` and still pass.
- `tools/verify.sh` is NOT usable in this build: it requires the QuantLib-linked binaries. Use `ctest`.
- Performance numbers are only gated against a committed baseline for the same machine fingerprint (`baselines/baselines.json`).
  On another machine the benches run, but the gate has no baseline to compare against.
- Linux/GCC is exercised only by the CI workflow (`.github/workflows/ci.yml`), which runs this recipe with `-DSWAPS_BUILD_BENCH=OFF`.

A first calibration through the CLI, using the committed SOFR fixture:

```bash
python3 -c "import json; b=json.load(open('bench/fixtures/sofr_bundle.json')); print(json.dumps({'bundle': b, 'sample_times': [1, 5, 10]}))" \
  | build/api/swaps_api_cli
# -> {"calibration":{"converged":true,"rms_residual":3.2e-17,...},"x":[...],...}
```

### Full gated build (with the QuantLib oracle and performance gate)

The reference platform is macOS. `tools/bootstrap_deps.sh` fetches and builds CMake, Ninja, Eigen, Boost, GoogleTest, Google
Benchmark and QuantLib 1.35 into `third_party/`. QuantLib is compiled with the engine's own optimisation flags, so the speed
comparison measures algorithms, not compiler settings. The script is resumable and takes about 30–50 minutes (QuantLib dominates).

```bash
./tools/bootstrap_deps.sh     # one-time
./tools/verify.sh             # configure + build + guards + correctness gate (all tests, incl. QuantLib oracle) + performance gate
```

`verify.sh` has two partial modes:
- `--test-only` runs the guards and correctness only.
- `--bench-only` runs the guards and performance only.

The performance gate refuses to run on a busy machine (more than 15 % CPU), and fails a metric that regresses more than 1.25× against
its baseline or exceeds its target.

To build the same tree by hand:

```bash
TP=third_party
$TP/toolchain/cmake/CMake.app/Contents/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=$TP/toolchain/bin/ninja -DCMAKE_PREFIX_PATH="$TP/gtest/install;$TP/benchmark/install"
$TP/toolchain/cmake/CMake.app/Contents/bin/cmake --build build -j
(cd build && ../$TP/toolchain/cmake/CMake.app/Contents/bin/ctest -j 8 --output-on-failure)
```

The QuantLib-linked binaries are `swaps_oracle_tests`, which compares the engine against QuantLib numbers, and
`swaps_consistency_tests`, which uses QuantLib only to build fixtures. The QuantLib reference benches are the `*_ql_bench`,
`curve_build_bench`, `risk_bench`, `portfolio_bench`, `warm_bench`, `bundle_build_bench`, `bond_sweep_bench` and the rest listed in
`bench/CMakeLists.txt`.

## Status

The engine is past its build-out phases, and through a principles-first review and cleanup (E0–E7) and a streaming hot-path
programme. It now has:
- guard tests;
- a hot-path census;
- bit-identical allocation reductions;
- correctness fixes, each reproduced by a failing test before it was fixed.

[docs/HISTORY.md](docs/HISTORY.md) has the story and the measured results.

## Licensing

No licence file has been added yet. Third-party components:
- QuantLib (modified BSD), used only by the oracle tests and reference benches;
- Eigen (MPL2);
- Boost (Boost Software License);
- GoogleTest (BSD-3-Clause);
- Google Benchmark (Apache-2.0).

Part of the holiday data in `conventions/conventions.json` was derived from QuantLib's calendars.
