#!/usr/bin/env bash
# check_oracle_tests.sh — STRUCTURAL guard for the two QuantLib-linked test registries (PRINCIPLES.md P8/P9/P11).
#
# Registries = the source lists of `swaps_oracle_tests` and `swaps_consistency_tests` in tests/CMakeLists.txt.
# For every listed file this FAILS the gate if:
#   (1) the file is missing;
#   (2) the banner is wrong (`@oracle-test` in the oracle list, `@consistency-test` in the consistency list);
#   (3) an ORACLE file has no `#include <ql/` / `QuantLib::` reference (an oracle must compare to QuantLib);
#   (4) its TEST()/TEST_F() count or its EXPECT_/ASSERT_ count dropped below tests/oracle_assertions.lock
#       (a gutted body, a deleted test or an assertion turned into a print all trip this — the old guard
#       was a banner grep and could not see any of them);
#   (5) with --build DIR: the binary is missing, or `--gtest_list_tests` lists fewer tests than the lock.
# Counts may only GROW silently; a deliberate reduction is committed with `--update` (and a reason).
#
#   check_oracle_tests.sh [--build BUILD_DIR] [--update]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_LIST="${ROOT}/tests/CMakeLists.txt"
TDIR="${SWAPS_TESTS_DIR:-${ROOT}/tests}"        # override = self-test hook (tools/selftest_guards.sh)
LOCK="${ROOT}/tests/oracle_assertions.lock"
BUILD=""; UPDATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD="$2"; shift 2 ;;
    --update) UPDATE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

registry() {  # $1 = target name -> newline-separated .cpp list
  awk -v tgt="add_executable($1" '
    index($0, tgt) {grab=1; next}
    grab && /^[ \t]*\)/ {grab=0}
    grab {gsub(/[ \t]/,""); if ($0 ~ /\.cpp$/) print $0}
  ' "${CMAKE_LIST}"
}
# Comment text is stripped first so a commented-out assertion/test does not count (that is exactly the gutting
# a refactor produces). Block comments are not handled — the lock is a floor, not a parser.
strip_comments() { sed -E 's#//.*$##' "$1"; }
count_tests()   { strip_comments "$1" | grep -cE '^[[:space:]]*TEST(_F|_P)?\(' || true; }
count_asserts() { strip_comments "$1" | grep -cE '\b(EXPECT|ASSERT)_[A-Z_]+\(' || true; }
listed_tests()  { "$1" --gtest_list_tests 2>/dev/null | grep -cE '^  ' || echo 0; }

fail=0; n=0
declare -a NEWLOCK
check_list() {  # $1 = target, $2 = required banner, $3 = require QL (1/0)
  local tgt="$1" banner="$2" needql="$3" f path t a lt la
  for f in $(registry "${tgt}"); do
    n=$((n+1)); path="${TDIR}/${f}"
    if [ ! -f "${path}" ]; then echo "  MISSING    ${tgt}/${f}" >&2; fail=1; continue; fi
    [ "${f}" = "ql_test_isolation.cpp" ] && continue   # gtest listener fixture (in both binaries), not a test
    grep -q "${banner}" "${path}" || { echo "  NO BANNER  ${f} — expected ${banner}" >&2; fail=1; }
    if [ "${needql}" = 1 ]; then
      grep -qE '#include <ql/|QuantLib::' "${path}" || { echo "  NO QUANTLIB ${f} — an oracle test must reference QuantLib" >&2; fail=1; }
    fi
    t="$(count_tests "${path}")"; a="$(count_asserts "${path}")"
    NEWLOCK+=("${tgt}|${f}|${t}|${a}")
    if [ -f "${LOCK}" ] && [ "${UPDATE}" = 0 ]; then
      lt="$(awk -F'|' -v t="${tgt}" -v f="${f}" '$1==t && $2==f {print $3}' "${LOCK}")"
      la="$(awk -F'|' -v t="${tgt}" -v f="${f}" '$1==t && $2==f {print $4}' "${LOCK}")"
      if [ -n "${lt}" ]; then
        [ "${t}" -lt "${lt}" ] && { echo "  SHRUNK     ${f}: TEST count ${t} < locked ${lt}" >&2; fail=1; }
        [ "${a}" -lt "${la}" ] && { echo "  GUTTED     ${f}: assertion count ${a} < locked ${la}" >&2; fail=1; }
      else
        echo "  UNLOCKED   ${f} — not in ${LOCK} (run with --update)" >&2; fail=1
      fi
    fi
  done
  return 0   # (set -e: the loop's last `[ ] && {}` must not become the function's status)
}
check_list swaps_oracle_tests      "@oracle-test"      1
check_list swaps_consistency_tests "@consistency-test" 0

# Binary-level check: the built oracle/consistency binaries must exist and list at least the locked count.
if [ -n "${BUILD}" ]; then
  for tgt in swaps_oracle_tests swaps_consistency_tests; do
    bin="${BUILD}/tests/${tgt}"
    if [ ! -x "${bin}" ]; then echo "  NO BINARY  ${bin} — QuantLib-linked gate binary not built" >&2; fail=1; continue; fi
    lc="$(listed_tests "${bin}")"
    NEWLOCK+=("${tgt}|__listed__|${lc}|0")
    if [ -f "${LOCK}" ] && [ "${UPDATE}" = 0 ]; then
      ll="$(awk -F'|' -v t="${tgt}" '$1==t && $2=="__listed__" {print $3}' "${LOCK}")"
      if [ -n "${ll}" ] && [ "${lc}" -lt "${ll}" ]; then echo "  SHRUNK     ${tgt}: ${lc} listed tests < locked ${ll}" >&2; fail=1; fi
    fi
  done
fi

if [ "${UPDATE}" = 1 ]; then
  { echo "# target|file|TEST count|EXPECT/ASSERT count — written by tools/check_oracle_tests.sh --update. Counts may only"
    echo "# grow silently; a reduction must be re-locked deliberately, in a commit that says why (PRINCIPLES.md P11)."
    printf '%s\n' "${NEWLOCK[@]}" | sort; } > "${LOCK}"
  echo "check_oracle_tests: lock updated (${#NEWLOCK[@]} entries) -> ${LOCK}"
fi
[ "${n}" -gt 0 ] || { echo "check_oracle_tests: FAIL — empty registry (could not parse ${CMAKE_LIST})" >&2; exit 1; }
if [ "${fail}" -ne 0 ]; then
  echo "check_oracle_tests: FAIL — see tests/ORACLE_TESTS.md" >&2; exit 1
fi
echo "check_oracle_tests: OK — ${n} registered files (oracle + consistency) present, bannered, locked"
