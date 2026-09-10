#!/usr/bin/env bash
# check_test_taxonomy.sh — every tests/*.cpp carries its E5 taxonomy label (T1 oracle / T2 calibration /
# T3 cross-path parity / T4 hot-path invariant / T5 properties + value pins / T6 regression), as a
# `// E5 taxonomy: T? ...` line within the first three lines (after the @oracle-test / @consistency-test
# banner where one exists). A test file without a label is a review finding: nobody has said what it pins.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TDIR="${SWAPS_TESTS_DIR:-${ROOT}/tests}"
fail=0; n=0
for f in "${TDIR}"/*.cpp; do
  n=$((n+1))
  if ! head -3 "$f" | grep -qE '^// E5 taxonomy: T[1-6] '; then
    echo "  NO LABEL   $(basename "$f") — add '// E5 taxonomy: T? …' (see tests/TAXONOMY.md)" >&2; fail=1
  fi
done
if [ "$fail" = 1 ]; then echo "check_test_taxonomy: FAIL" >&2; exit 1; fi
echo "check_test_taxonomy: OK — ${n} test files labelled"
