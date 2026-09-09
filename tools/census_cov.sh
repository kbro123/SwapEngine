#!/usr/bin/env bash
# E3.1 instrumented census: build bench/hotpath_census with clang SanitizerCoverage so every basic block,
# indirect (virtual) call, comparison, load and store is counted exactly — what hardware counters would give,
# without kperf. Counts are deterministic (one call per path is enough); timings from this build are
# meaningless and are not reported. Output: build/perf/hotpath_census_cov.json + a table on stdout.
#   tools/census_cov.sh            # configure build-cov (once), build the census target, run it
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CM="${ROOT}/third_party/toolchain/cmake/CMake.app/Contents/bin/cmake"; [ -x "$CM" ] || CM=cmake
MK=$(grep CMAKE_MAKE_PROGRAM "${ROOT}/build/CMakeCache.txt" | cut -d= -f2)
CXX=$(grep "CMAKE_CXX_COMPILER:" "${ROOT}/build/CMakeCache.txt" | cut -d= -f2)
COV="-fsanitize-coverage=trace-pc-guard,indirect-calls,trace-cmp,trace-loads,trace-stores"
[ -f "${ROOT}/build-cov/CMakeCache.txt" ] || "$CM" -S "$ROOT" -B "${ROOT}/build-cov" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSWAPS_ALLOW_NO_ORACLE=ON -DCMAKE_MAKE_PROGRAM="$MK" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CXX_FLAGS="$COV" > /dev/null
"$CM" --build "${ROOT}/build-cov" --target hotpath_census -j4 | grep -E "error|Linking" || true
mkdir -p "${ROOT}/build/perf"
HOTPATH_CENSUS_REPS=1 "${ROOT}/build-cov/bench/hotpath_census" "${ROOT}/build/perf/hotpath_census_cov.json"
