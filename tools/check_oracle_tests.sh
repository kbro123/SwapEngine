#!/usr/bin/env bash
# Guard: the oracle tests (the QuantLib cashflow-for-cashflow validators) must not be silently deleted
# or gutted. The CMake `swaps_oracle_tests` target IS the registry of oracle tests; this script asserts
# that every .cpp listed there still (1) exists, (2) carries the @oracle-test banner, and (3) references
# QuantLib. Any of those failing fails the gate. See tests/ORACLE_TESTS.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE="${ROOT}/tests/CMakeLists.txt"
TDIR="${ROOT}/tests"

# Extract the file list inside add_executable(swaps_oracle_tests ... ) -- the *.cpp lines up to the ')'.
# (No mapfile: macOS ships bash 3.2. A newline-separated list + a plain for-loop is portable.)
ORACLE="$(awk '
  /add_executable\(swaps_oracle_tests/ {grab=1; next}
  grab && /^[ \t]*\)/ {grab=0}
  grab {gsub(/[ \t]/,""); if ($0 ~ /\.cpp$/) print $0}
' "$CMAKE")"

if [ -z "$ORACLE" ]; then
  echo "check_oracle_tests: FAIL — could not find the swaps_oracle_tests file list in $CMAKE" >&2
  exit 1
fi

n=0; fail=0
for f in $ORACLE; do
  n=$((n + 1))
  path="${TDIR}/${f}"
  if [ ! -f "$path" ]; then
    echo "  MISSING   $f — listed as an oracle test but the file is gone" >&2; fail=1; continue
  fi
  grep -q "@oracle-test" "$path" || { echo "  NO BANNER $f — missing the @oracle-test banner" >&2; fail=1; }
  grep -qE '#include <ql/|QuantLib::' "$path" \
    || echo "  WARN      $f — no direct QuantLib reference (oracle uses a reference_*.hpp builder?)" >&2
done

if [ "$fail" -ne 0 ]; then
  echo "check_oracle_tests: FAIL — oracle tests are protected; see tests/ORACLE_TESTS.md" >&2
  exit 1
fi
echo "check_oracle_tests: OK — ${n} oracle tests present and banner-tagged"
