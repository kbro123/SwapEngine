#!/usr/bin/env bash
# fingerprint.sh — emit a stable identity for "the machine + toolchain this was benchmarked on".
#
# Performance baselines are only comparable within one fingerprint. The perf gate uses this
# to refuse cross-machine comparisons (a 2017 Kaby Lake number vs an M4 number is meaningless).
# Prints JSON to stdout.
set -euo pipefail

CXX_BIN="${CXX:-c++}"

case "$(uname -m)" in
  arm64|aarch64) ARCH_FLAG="-mcpu=native" ;;
  x86_64)        ARCH_FLAG="-march=native" ;;
  *)             ARCH_FLAG="" ;;
esac

# CPU brand
if [ "$(uname -s)" = "Darwin" ]; then
  CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)"
else
  CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo unknown)"
  CORES="$(nproc 2>/dev/null || echo 0)"
fi

COMPILER="$("${CXX_BIN}" --version 2>/dev/null | head -1)"

# Which SIMD macros does this compiler+flag combination actually define?
MACROS="$("${CXX_BIN}" ${ARCH_FLAG} -dM -E -x c++ /dev/null 2>/dev/null || true)"
has() { echo "${MACROS}" | grep -q "define $1 " && echo true || echo false; }

AVX512="$(has __AVX512F__)"
AVX2="$(has __AVX2__)"
AVX="$(has __AVX__)"
FMA="$(has __FMA__)"
NEON="$(has __ARM_NEON)"
SSE2="$(has __SSE2__)"

if   [ "${AVX512}" = true ]; then ISA="AVX-512"; W=8
elif [ "${AVX2}"   = true ]; then ISA="AVX2";    W=4
elif [ "${AVX}"    = true ]; then ISA="AVX";     W=4
elif [ "${NEON}"   = true ]; then ISA="NEON";    W=2
elif [ "${SSE2}"   = true ]; then ISA="SSE2";    W=2
else                              ISA="scalar";  W=1
fi

# Short, stable key. Deliberately includes the compiler: changing compilers
# invalidates baselines just as surely as changing CPUs.
KEY_RAW="$(uname -m)|${CPU}|${ISA}|${COMPILER}|${ARCH_FLAG}"
KEY="$(printf '%s' "${KEY_RAW}" | shasum -a 256 2>/dev/null | cut -c1-12)"

cat <<EOF
{
  "key": "${KEY}",
  "arch": "$(uname -m)",
  "os": "$(uname -s) $(uname -r)",
  "cpu": "${CPU}",
  "physical_cores": ${CORES},
  "isa": "${ISA}",
  "doubles_per_register": ${W},
  "fma": ${FMA},
  "compiler": "${COMPILER}",
  "arch_flag": "${ARCH_FLAG}"
}
EOF
