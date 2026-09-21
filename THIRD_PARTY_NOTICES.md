# Third-party notices

SwapsEngine itself is proprietary — see [`LICENSE`](LICENSE). This file records every third-party
component the repository uses, its licence, and **whether it reaches a shipped binary**. That second
column is the one that matters for redistribution: the engine's own rule (CLAUDE.md §1) is that
QuantLib is an *oracle*, not a dependency, and this table is where that claim is made checkable.

Nothing here is vendored into git: `third_party/` is gitignored and populated by
`tools/bootstrap_deps.sh`, which fetches each component from its own upstream at the pinned version.

## In the shipped engine (`api/`, the C ABI, the web binding)

| Component | Version | Licence | Notes |
|---|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | 3.4.0 | MPL-2.0 | Headers only, used unmodified. `Core`, `Dense`, `SparseCore`. See the MPL-2.0 note below. |
| [Boost](https://www.boost.org) (Boost.JSON) | 1.84.0 | BSL-1.0 | Headers + `boost/json/src.hpp`. BSL-1.0 requires no attribution in binary form. |

Both are permissive with respect to proprietary distribution: MPL-2.0 is *file-level* copyleft, so
using unmodified Eigen headers inside a larger proprietary work is expressly allowed (MPL-2.0 §3.3);
only modifications *to Eigen's own files* would have to be published. Eigen is not modified here.

**`EIGEN_MPL2_ONLY` is set on the `swaps` target** (`CMakeLists.txt`). Eigen is MPL-2.0 apart from a
few modules carrying third-party LGPL-2.1 code (`SimplicialCholesky`, AMD ordering, `constrained_cg`);
that define makes including any of them a **compile error**, so "the shipped engine links nothing
copyleft" is enforced by the build rather than asserted by an audit.

## Test and benchmark only — never linked into a shipped binary

| Component | Version | Licence | Used by |
|---|---|---|---|
| [QuantLib](https://www.quantlib.org) | 1.35 | QuantLib licence (modified BSD, 3-clause style) | `tests/` correctness oracle; `bench/` informational reference. Built from source with the engine's exact flags (CLAUDE.md §3 perf-gate integrity). |
| [GoogleTest](https://github.com/google/googletest) | 1.14.0 | BSD-3-Clause | `tests/` |
| [Google Benchmark](https://github.com/google/benchmark) | 1.8.4 | Apache-2.0 | `bench/` |

## Build tooling only

| Component | Version | Licence | Notes |
|---|---|---|---|
| [CMake](https://cmake.org) | 3.29.6 | BSD-3-Clause | Vendored under `third_party/toolchain/`, not on `PATH`. |
| [Ninja](https://ninja-build.org) | 1.12.1 | Apache-2.0 | Same. |
| Apple Clang / libc++ | CLT 26.x | Apple / Apache-2.0 with LLVM exception | System toolchain; not redistributed. |

## Generated reference data

| Artefact | Generator | Licence of the generator | Status |
|---|---|---|---|
| `tests/golden/bond_reference.csv` | [OpenGamma Strata](https://strata.opengamma.io) 2.12.56 via `tools/bond_reference/gen_golden.py` (JPype; jars pinned by SHA-256) | Apache-2.0 (Strata, Guava, Joda-Beans/Convert, commons-math3); MIT (SLF4J); CERN/LGPL-style (Colt) | The CSV is committed and **required** by the gate. No jar is committed or linked; the engine's build never needs Java. |

**Historical note (LIC1).** Until 2026-09-14 that golden was generated with
[Rateslib](https://rateslib.com), which is *source-available, not open-source*: without a registered
licence its terms permit non-commercial use only. It was replaced by Strata (commit `1c77771`) and the
`ust_31bii` rows went with it, so no Rateslib-derived data remains in the tree. Rateslib is still cited
in comments as a *convention reference* — citing a source is not a use of the software.

## Market conventions

`conventions/conventions.json` holds day counts, calendars, frequencies and lags transcribed from
published market conventions and public documents (e.g. 31 CFR Part 356 Appendix B for the US Treasury
yield method). Facts and formulae from a regulation or a market convention are not copyrightable
expression; no third-party data file is redistributed here.
