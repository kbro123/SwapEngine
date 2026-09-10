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
pass_schema="SKIP"
pass_taxonomy="SKIP"
pass_graph="SKIP"
pass_lit="SKIP"
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
# The oracle + consistency registries (tests/CMakeLists.txt) must not be deleted, gutted or shrunk, and the
# QuantLib-linked binaries must actually exist (QuantLib absent used to silently drop 135 tests).
# See tools/check_oracle_tests.sh + tests/ORACLE_TESTS.md + tests/oracle_assertions.lock.
echo ">> oracle-test guard (registries + assertion lock + built binaries)"
if bash "${ROOT}/tools/check_oracle_tests.sh" --build "${BUILD_DIR}"; then
  pass_oracle="PASS"
else
  pass_oracle="FAIL"; rc=1
fi

# ---- Include-graph guard (E6.4) ---------------------------------------------
# The layer DAG drawn in ARCHITECTURE.md is the contract: every real cross-layer #include must be a declared
# arrow, and the graph must stay acyclic. Cheap (a file walk), so it runs even in --bench-only.
echo ">> include-graph guard (ARCHITECTURE.md's layer DAG vs the real includes)"
if python3 "${ROOT}/tools/check_include_graph.py"; then
  pass_graph="PASS"
else
  pass_graph="FAIL"; rc=1
fi

# ---- Oracle-REACH guard (golden-source step 6) ------------------------------
# check_oracle_tests.sh guards the oracle FILES (present, bannered, not gutted). This guards the other
# direction: which shipped HEADERS a QuantLib comparison can still reach. The 2026-09-10 averaged-weight
# bug sat in build/observations.hpp while every oracle that could have caught it built its observations
# test-side -- coverage that no file-level guard could see was missing. A header may only GAIN reach.
echo ">> oracle-reach guard (a QuantLib number still reaches the locked headers)"
if python3 "${ROOT}/tools/oracle_coverage.py" --check; then
  pass_reach="PASS"
else
  pass_reach="FAIL"; rc=1
fi

# ---- Test-taxonomy guard (E5) -----------------------------------------------
# Every tests/*.cpp says what its assertions compare an engine number TO (tests/TAXONOMY.md). Cheap.
echo ">> test-taxonomy guard (every test file labelled T1-T6)"
if bash "${ROOT}/tools/check_test_taxonomy.sh"; then
  pass_taxonomy="PASS"
else
  pass_taxonomy="FAIL"; rc=1
fi

# ---- Conventions-DB sync guard ----------------------------------------------
# include/swaps/conventions_data.hpp is codegen'd from conventions/conventions.json. Regenerate into a
# temp file and diff: a JSON edit without re-running the generator (so the C++ builders would read stale
# conventions) FAILs here. Cheap, so it runs even in --bench-only.
echo ">> conventions-DB sync guard"
pass_conv="PASS"
_conv_tmp="$(mktemp)"
if python3 "${ROOT}/tools/gen_conventions_hpp.py" --stdout > "${_conv_tmp}" 2>/dev/null \
   && diff -q "${_conv_tmp}" "${ROOT}/include/swaps/conventions_data.hpp" >/dev/null; then
  pass_conv="PASS"
else
  echo "   conventions_data.hpp is STALE — run: python3 tools/gen_conventions_hpp.py"
  pass_conv="FAIL"; rc=1
fi
rm -f "${_conv_tmp}"

# ---- Conventions-DB schema + referential-integrity guard (PRINCIPLES.md P2) ----
echo ">> conventions schema guard"
if python3 "${ROOT}/tools/check_schema.py"; then pass_schema="PASS"; else pass_schema="FAIL"; rc=1; fi

# ---- No-literal-conventions guard (PRINCIPLES.md P2) ------------------------
# No market-convention literal (currency/index/calendar id, day count, frequency, lag, recovery, contract
# spec, silent fallback default) may appear in include/ or api/. tools/check_no_literals.allow holds the
# vocabulary-dispatch permissions plus the adoption-day RATCHET (existing hits, to be burned down in E2);
# any NEW hit fails here. Cheap, runs even in --bench-only.
echo ">> no-literal-conventions guard"
if python3 "${ROOT}/tools/check_no_literals.py" --allow "${ROOT}/tools/check_no_literals.allow" >/dev/null; then
  pass_lit="PASS"
else
  python3 "${ROOT}/tools/check_no_literals.py" --allow "${ROOT}/tools/check_no_literals.allow" | tail -20
  pass_lit="FAIL"; rc=1
fi

# ---- API-descriptor sync guard: run_json's generated dispatch must match api/api_surface.py ----------
echo ">> api-dispatch sync guard"
pass_disp="PASS"
_disp_tmp="$(mktemp)"
_disp_tmp2="$(mktemp)"
if python3 "${ROOT}/tools/gen_dispatch.py" --stdout > "${_disp_tmp}" 2>/dev/null \
   && diff -q "${_disp_tmp}" "${ROOT}/api/run_json_dispatch.gen.inc" >/dev/null \
   && python3 "${ROOT}/tools/gen_dispatch.py" --stdout-stateless > "${_disp_tmp2}" 2>/dev/null \
   && diff -q "${_disp_tmp2}" "${ROOT}/api/run_json_stateless.gen.inc" >/dev/null; then
  pass_disp="PASS"
else
  echo "   api/run_json_{dispatch,stateless}.gen.inc is STALE — run: python3 tools/gen_dispatch.py"
  pass_disp="FAIL"; rc=1
fi
rm -f "${_disp_tmp}" "${_disp_tmp2}"

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
  # PRINCIPLES.md P9: self-baseline + absolute targets; refuses to run under load (exit 2 = not run = FAIL).
  mkdir -p "${BUILD_DIR}/perf"
  if python3 "${ROOT}/tools/check_perf.py" --build "${BUILD_DIR}" \
       --baselines "${ROOT}/baselines/baselines.json" --targets "${ROOT}/baselines/targets.json" \
       --record "${BUILD_DIR}/perf/last_run.json" ${SWAPS_PERF_ARGS:-}; then
    pass_perf="PASS"
  else
    pass_perf="FAIL"; rc=1
  fi
fi

# ---- Summary ----------------------------------------------------------------
echo ""
echo "========== VERIFY SUMMARY =========="
printf "  %-20s %s\n" "oracle-test guard:" "${pass_oracle}"
printf "  %-20s %s\n" "test-taxonomy guard:" "${pass_taxonomy}"
printf "  %-20s %s\n" "include-graph guard:" "${pass_graph}"
printf "  %-20s %s\n" "oracle-reach guard:" "${pass_reach}"
printf "  %-20s %s\n" "conventions sync:" "${pass_conv}"
printf "  %-20s %s\n" "conventions schema:" "${pass_schema}"
printf "  %-20s %s\n" "no-literal guard:" "${pass_lit}"
printf "  %-20s %s\n" "api-dispatch sync:" "${pass_disp}"
printf "  %-20s %s\n" "correctness gate:" "${pass_correctness}"
printf "  %-20s %s\n" "performance gate:" "${pass_perf}"
echo "===================================="
[ "${rc}" -eq 0 ] && echo "RESULT: OK" || echo "RESULT: FAILED"
exit "${rc}"
