#!/usr/bin/env bash
# verify.sh — runs the two gates (correctness + performance) and prints a pass/fail table.
# Exit code 0 iff every enabled gate passes. This is the gate for every checkpoint.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"

# The toolchain is vendored (this machine is Homebrew Tier 3), so cmake/ninja are not
# on PATH. Prefer third_party/, fall back to a system install.
TP="${ROOT}/third_party"
CMAKE="${TP}/toolchain/cmake/CMake.app/Contents/bin/cmake"
CTEST="${TP}/toolchain/cmake/CMake.app/Contents/bin/ctest"
NINJA="${TP}/toolchain/bin/ninja"
[ -x "${CMAKE}" ] || CMAKE="$(command -v cmake || true)"
[ -x "${CTEST}" ] || CTEST="$(command -v ctest || true)"
[ -x "${NINJA}" ] || NINJA="$(command -v ninja || true)"
if [ -z "${CMAKE}" ] || [ -z "${CTEST}" ]; then
  echo "cmake/ctest not found. Run ./tools/bootstrap_deps.sh first." >&2; exit 1
fi

# Never let the build use ninja's default parallelism: it kernel-panicked this machine
# (see CLAUDE.md §4). Cap at physical cores.
JOBS="${SWAPS_BUILD_JOBS:-$( (sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4) )}"

BENCH_ONLY=0
TEST_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --bench-only) BENCH_ONLY=1 ;;
    --test-only)  TEST_ONLY=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

pass_oracle="SKIP"
pass_correctness="SKIP"
pass_perf="SKIP"
rc=0

# ---- Build ------------------------------------------------------------------
if [ ! -d "${BUILD_DIR}" ]; then
  echo ">> configuring (Release)"
  "${CMAKE}" -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
    ${NINJA:+-DCMAKE_MAKE_PROGRAM="${NINJA}"} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${TP}/gtest/install;${TP}/benchmark/install" \
    || { echo "configure failed"; exit 1; }
fi
echo ">> building (-j${JOBS})"
"${CMAKE}" --build "${BUILD_DIR}" -j "${JOBS}" || { echo "build failed"; exit 1; }

# ---- Oracle-test guard ------------------------------------------------------
# The QuantLib cashflow-for-cashflow validators must not be silently deleted or gutted in a refactor.
# Cheap check, so it runs even in --bench-only. See tools/check_oracle_tests.sh + tests/ORACLE_TESTS.md.
echo ">> oracle-test guard"
if bash "${ROOT}/tools/check_oracle_tests.sh"; then
  pass_oracle="PASS"
else
  pass_oracle="FAIL"; rc=1
fi

# ---- Correctness gate -------------------------------------------------------
if [ "${BENCH_ONLY}" -eq 0 ]; then
  echo ">> correctness gate (ctest)"
  if "${CTEST}" --test-dir "${BUILD_DIR}" --output-on-failure; then
    pass_correctness="PASS"
  else
    pass_correctness="FAIL"; rc=1
  fi
fi

# ---- Performance gate -------------------------------------------------------
# check_perf.py runs the google-benchmark targets, parses their JSON, and compares each metric to
# baselines/baselines.json -- FAILing if we regress vs our own committed baseline or fall below the
# required speedup vs the QuantLib baseline. (If the checker is ever absent the gate SKIPs rather than
# blocking, but it is committed and normally present.)
if [ "${TEST_ONLY}" -eq 0 ]; then
  echo ">> performance gate"
  if [ -f "${ROOT}/tools/check_perf.py" ]; then
    if python3 "${ROOT}/tools/check_perf.py" --build "${BUILD_DIR}" --baselines "${ROOT}/baselines/baselines.json"; then
      pass_perf="PASS"
    else
      pass_perf="FAIL"; rc=1
    fi
  else
    echo "   (tools/check_perf.py missing — perf gate skipped)"
    pass_perf="SKIP"
  fi
fi

# ---- Summary ----------------------------------------------------------------
echo ""
echo "========== VERIFY SUMMARY =========="
printf "  %-20s %s\n" "oracle-test guard:" "${pass_oracle}"
printf "  %-20s %s\n" "correctness gate:" "${pass_correctness}"
printf "  %-20s %s\n" "performance gate:" "${pass_perf}"
echo "===================================="
[ "${rc}" -eq 0 ] && echo "RESULT: OK" || echo "RESULT: FAILED"
exit "${rc}"
