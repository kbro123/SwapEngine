#!/usr/bin/env bash
# selftest_guards.sh — prove the gate guards can FAIL (a guard that cannot fail is not a guard; PRINCIPLES.md P9).
# Copies tests/ to a scratch dir, guts one oracle file (comments out its EXPECTs), and asserts the guard trips.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S="$(mktemp -d)"; trap 'rm -rf "${S}"' EXIT
cp -r "${ROOT}/tests" "${S}/tests"
sed -i '' -E 's/^([[:space:]]*)((EXPECT|ASSERT)_)/\1\/\/\2/' "${S}/tests/extract_test.cpp"   # gut every assertion
if SWAPS_TESTS_DIR="${S}/tests" bash "${ROOT}/tools/check_oracle_tests.sh" >/dev/null 2>&1; then
  echo "selftest_guards: FAIL — the oracle guard did NOT catch a gutted extract_test.cpp" >&2; exit 1
fi
cp "${ROOT}/tests/extract_test.cpp" "${S}/tests/extract_test.cpp"
sed -i '' '/^TEST/,$d' "${S}/tests/extract_test.cpp"   # delete every TEST from the first one on
if SWAPS_TESTS_DIR="${S}/tests" bash "${ROOT}/tools/check_oracle_tests.sh" >/dev/null 2>&1; then
  echo "selftest_guards: FAIL — the oracle guard did NOT catch deleted TESTs in extract_test.cpp" >&2; exit 1
fi
echo "selftest_guards: OK — oracle guard trips on gutted assertions and on deleted tests"
