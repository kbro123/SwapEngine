#!/usr/bin/env bash
# Compile & run the QuantLib baseline smoke test against our vendored, self-built QuantLib.
# Proves the oracle/baseline library is usable before we build the correctness gate on it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
QL="${TP}/quantlib/install"

[ -d "${QL}/include/ql" ] || { echo "QuantLib not built. Run ./tools/bootstrap_deps.sh first." >&2; exit 1; }

OUT="${ROOT}/.cache/ql_smoke"
mkdir -p "${ROOT}/.cache"

echo ">> compiling ql_smoke against vendored QuantLib (${QL})"
"${CXX:-c++}" -std=c++20 -O2 \
  -I"${QL}/include" -I"${TP}/boost" \
  "${ROOT}/tools/ql_smoke.cpp" \
  -L"${QL}/lib" -lQuantLib \
  -o "${OUT}"

echo ">> running"
"${OUT}"
