#!/usr/bin/env bash
# verify.sh — runs the two gates (correctness + performance) and prints a pass/fail table.
# Exit code 0 iff every enabled gate passes. This is the gate for every checkpoint.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
BENCH_ONLY=0
TEST_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --bench-only) BENCH_ONLY=1 ;;
    --test-only)  TEST_ONLY=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

pass_correctness="SKIP"
pass_perf="SKIP"
rc=0

# ---- Build ------------------------------------------------------------------
if [ ! -d "${BUILD_DIR}" ]; then
  echo ">> configuring (Release)"
  cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release || { echo "configure failed"; exit 1; }
fi
echo ">> building"
cmake --build "${BUILD_DIR}" || { echo "build failed"; exit 1; }

# ---- Correctness gate -------------------------------------------------------
if [ "${BENCH_ONLY}" -eq 0 ]; then
  echo ">> correctness gate (ctest)"
  if ctest --test-dir "${BUILD_DIR}" --output-on-failure; then
    pass_correctness="PASS"
  else
    pass_correctness="FAIL"; rc=1
  fi
fi

# ---- Performance gate -------------------------------------------------------
# TODO(Phase 6): run google-benchmark targets, parse JSON, compare each metric to
# baselines/baselines.json, and FAIL if we are not faster than the QuantLib baseline
# by the agreed threshold (or if we regress vs our own committed baseline).
if [ "${TEST_ONLY}" -eq 0 ]; then
  echo ">> performance gate"
  if [ -x "${ROOT}/tools/check_perf.py" ] || [ -f "${ROOT}/tools/check_perf.py" ]; then
    if python3 "${ROOT}/tools/check_perf.py" --build "${BUILD_DIR}" --baselines "${ROOT}/baselines/baselines.json"; then
      pass_perf="PASS"
    else
      pass_perf="FAIL"; rc=1
    fi
  else
    echo "   (perf checker not implemented yet — Phase 6)"
    pass_perf="SKIP"
  fi
fi

# ---- Summary ----------------------------------------------------------------
echo ""
echo "========== VERIFY SUMMARY =========="
printf "  %-20s %s\n" "correctness gate:" "${pass_correctness}"
printf "  %-20s %s\n" "performance gate:" "${pass_perf}"
echo "===================================="
[ "${rc}" -eq 0 ] && echo "RESULT: OK" || echo "RESULT: FAILED"
exit "${rc}"
