#!/usr/bin/env bash
# bootstrap_deps.sh — fetch + build every project dependency into third_party/.
#
# Why not Homebrew? This machine is macOS 13 / Intel, which is Homebrew "Tier 3":
# no prebuilt bottles, so `brew install` compiles everything from source and pulls in
# go/rust/llvm build deps. We vendor instead. See CLAUDE.md §4.
#
# CRITICAL (perf-gate integrity, CLAUDE.md §3): QuantLib is compiled here with the SAME
# compiler and SAME flags as our engine. Benchmarking a tuned engine against a generically
# compiled QuantLib would measure compiler flags, not our algorithm.
#
# Idempotent: re-running skips anything already present. All downloads are resumable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
DL="${TP}/.downloads"
TOOLS="${TP}/toolchain"
mkdir -p "${TP}" "${DL}" "${TOOLS}"

# ---- Pinned versions (bump deliberately; changing these invalidates perf baselines) ----
CMAKE_VER=3.29.6
NINJA_VER=1.12.1
EIGEN_VER=3.4.0
GTEST_VER=1.14.0
BENCH_VER=1.8.4
BOOST_VER=1.84.0
QL_VER=1.35

# Optimization flags — MUST match what CMakeLists.txt/cmake/DetectISA.cmake give the engine, or the
# perf reference against QuantLib is invalid (PRINCIPLES.md, perf-gate integrity). There is ONE source of
# truth for the arch flag: cmake/DetectISA.cmake. It is probed below (after cmake/ninja are available) by
# configuring tools/archprobe with this compiler — never re-derived here by hand.
# (History: this script hard-coded -march=native while the engine moved to -march=x86-64-v3 on 2026-07-29;
#  the mismatch went unnoticed until 2026-09-08. Hence the probe.)
RELEASE_FLAGS="-O3 -DNDEBUG -fno-math-errno"   # == CMAKE_CXX_FLAGS_RELEASE in CMakeLists.txt
OPT_FLAGS=""                                    # set by probe_arch_flags after step 2
probe_arch_flags() {
  local dir="${TP}/.archprobe"
  rm -rf "${dir}"
  "${CMAKE}" -S "${ROOT}/tools/archprobe" -B "${dir}" -G Ninja -DCMAKE_MAKE_PROGRAM="${NINJA}" \
    -DCMAKE_BUILD_TYPE=Release ${SWAPS_ENABLE_AVX512:+-DSWAPS_ENABLE_AVX512=${SWAPS_ENABLE_AVX512}} \
    >"${dir}.log" 2>&1 || die "archprobe configure failed (see ${dir}.log)"
  ARCH_FLAG="$(tr -d '[:space:]' < "${dir}/arch_flags.txt")"
  OPT_FLAGS="${RELEASE_FLAGS} ${ARCH_FLAG}"
  log "engine flags (from cmake/DetectISA.cmake): ${OPT_FLAGS}  [$(tr -d '[:space:]' < "${dir}/isa_name.txt")]"
}

BOOST_USCORE="${BOOST_VER//./_}"

# Build parallelism. DO NOT let ninja use its default (logical cores + 2 = 10 here).
# On this 4-core/16GB Ventura machine that many concurrent clang processes exhausted
# memory and triggered an APFS kernel panic (OSMetaClassBase::_RESERVEDOSMetaClassBase6,
# panicking task = clang) plus non-deterministic clang segfaults. Cap at physical cores.
JOBS="${SWAPS_BUILD_JOBS:-$( (sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4) )}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# Resumable, retrying fetch. The network on this machine drops frequently.
fetch() {
  local url="$1" out="${DL}/$2"
  if [ -s "${out}" ]; then log "cached: $2"; return 0; fi
  log "fetching $2"
  # -sS: quiet progress bars (they render as thousands of lines in a log) but keep errors.
  curl -fL -sS --retry 8 --retry-delay 3 --retry-connrefused -C - -o "${out}.part" "${url}" \
    || die "download failed: ${url}  (re-run this script; it resumes)"
  mv "${out}.part" "${out}"
}

# ---- 0. Sanity: we need a C++20-capable compiler -----------------------------
CXX_BIN="${CXX:-c++}"
if ! echo 'int main(){}' | "${CXX_BIN}" -std=c++20 -x c++ - -o /dev/null 2>/dev/null; then
  die "${CXX_BIN} does not support -std=c++20.
     Install the Command Line Tools (Apple clang 15+ required for C++20 — no exact version pinned), then:
       sudo softwareupdate --install \"Command Line Tools for Xcode-16.2\"   # or: sudo xcode-select --install
       sudo xcode-select -s /Library/Developer/CommandLineTools
     Verify with: clang++ --version"
fi
log "compiler OK: $("${CXX_BIN}" --version | head -1)"

# ---- 1. CMake (prebuilt universal binary — no compile) -----------------------
if [ ! -x "${TOOLS}/cmake/CMake.app/Contents/bin/cmake" ]; then
  fetch "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-macos-universal.tar.gz" "cmake.tar.gz"
  mkdir -p "${TOOLS}/cmake"
  tar xzf "${DL}/cmake.tar.gz" -C "${TOOLS}/cmake" --strip-components=1
fi
CMAKE="${TOOLS}/cmake/CMake.app/Contents/bin/cmake"
log "cmake: $("${CMAKE}" --version | head -1)"

# ---- 2. Ninja (prebuilt binary) ---------------------------------------------
if [ ! -x "${TOOLS}/bin/ninja" ]; then
  fetch "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VER}/ninja-mac.zip" "ninja-mac.zip"
  mkdir -p "${TOOLS}/bin"
  unzip -oq "${DL}/ninja-mac.zip" -d "${TOOLS}/bin"
  chmod +x "${TOOLS}/bin/ninja"
fi
NINJA="${TOOLS}/bin/ninja"
log "ninja: $("${NINJA}" --version)"

# ---- 2b. The engine's exact optimisation flags (single source: cmake/DetectISA.cmake) ----
probe_arch_flags

# ---- 3. Eigen (header-only) --------------------------------------------------
if [ ! -d "${TP}/eigen/Eigen" ]; then
  fetch "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz" "eigen.tar.gz"
  mkdir -p "${TP}/eigen"
  tar xzf "${DL}/eigen.tar.gz" -C "${TP}/eigen" --strip-components=1
fi
log "eigen: header-only at third_party/eigen"

# ---- 4. Boost headers (QuantLib needs headers only) --------------------------
if [ ! -d "${TP}/boost/boost" ]; then
  fetch "https://archives.boost.io/release/${BOOST_VER}/source/boost_${BOOST_USCORE}.tar.gz" "boost.tar.gz"
  mkdir -p "${TP}/boost"
  # Extract only the header tree; the compiled libs are not needed for the QuantLib library.
  tar xzf "${DL}/boost.tar.gz" -C "${TP}/boost" --strip-components=1 "boost_${BOOST_USCORE}/boost"
fi
log "boost: headers at third_party/boost"

# ---- 5. GoogleTest -----------------------------------------------------------
if [ ! -f "${TP}/gtest/install/lib/libgtest.a" ]; then
  fetch "https://github.com/google/googletest/archive/refs/tags/v${GTEST_VER}.tar.gz" "gtest.tar.gz"
  rm -rf "${TP}/gtest/src"; mkdir -p "${TP}/gtest/src"
  tar xzf "${DL}/gtest.tar.gz" -C "${TP}/gtest/src" --strip-components=1
  "${CMAKE}" -S "${TP}/gtest/src" -B "${TP}/gtest/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${NINJA}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS="${OPT_FLAGS}" \
    -DCMAKE_INSTALL_PREFIX="${TP}/gtest/install" -DBUILD_GMOCK=OFF
  "${CMAKE}" --build "${TP}/gtest/build" --target install -j "${JOBS}"
fi
log "googletest: third_party/gtest/install"

# ---- 6. Google Benchmark -----------------------------------------------------
if [ ! -f "${TP}/benchmark/install/lib/libbenchmark.a" ]; then
  fetch "https://github.com/google/benchmark/archive/refs/tags/v${BENCH_VER}.tar.gz" "benchmark.tar.gz"
  rm -rf "${TP}/benchmark/src"; mkdir -p "${TP}/benchmark/src"
  tar xzf "${DL}/benchmark.tar.gz" -C "${TP}/benchmark/src" --strip-components=1
  "${CMAKE}" -S "${TP}/benchmark/src" -B "${TP}/benchmark/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${NINJA}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS="${OPT_FLAGS}" \
    -DCMAKE_INSTALL_PREFIX="${TP}/benchmark/install" \
    -DBENCHMARK_ENABLE_TESTING=OFF -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
  "${CMAKE}" --build "${TP}/benchmark/build" --target install -j "${JOBS}"
fi
log "google-benchmark: third_party/benchmark/install"

# ---- 7. QuantLib — the oracle + speed baseline -------------------------------
# Built with OUR compiler and OUR flags. This is a correctness requirement of the
# perf gate, not an optimization. See CLAUDE.md §3.
# Built STATIC on purpose: a shared QuantLib pays cross-DSO call overhead and blocks
# inlining, which would handicap the baseline and flatter our speedup. Give the
# baseline its best shot (CLAUDE.md §3).
QL_STATIC_LIB="${TP}/quantlib/install/lib/libQuantLib.a"
if [ ! -f "${QL_STATIC_LIB}" ]; then
  fetch "https://github.com/lballabio/QuantLib/releases/download/v${QL_VER}/QuantLib-${QL_VER}.tar.gz" "quantlib.tar.gz"
  # Do NOT re-extract if the source is already there: re-extraction resets every mtime,
  # which makes ninja rebuild all 937 objects and silently defeats the resume below.
  if [ ! -f "${TP}/quantlib/src/CMakeLists.txt" ]; then
    rm -rf "${TP}/quantlib/src"; mkdir -p "${TP}/quantlib/src"
    tar xzf "${DL}/quantlib.tar.gz" -C "${TP}/quantlib/src" --strip-components=1
  fi
  # QuantLib 1.35 + libc++ 26 (clang 21): in ql/errors.cpp the unqualified call format(file, line, ...)
  # finds std::format by ADL on std::string and fails (consteval format-string). Qualify the call to the
  # file's own anonymous-namespace helper (::format disables ADL). Fixed upstream after 1.35; idempotent sed.
  sed -i '' -e 's/std::runtime_error(format(/std::runtime_error(::format(/g' \
            -e 's/make_shared<std::string>(format(/make_shared<std::string>(::format(/g' \
            "${TP}/quantlib/src/ql/errors.cpp"
  log "building QuantLib ${QL_VER} with: ${OPT_FLAGS} -j${JOBS}  (~20-40 min on 4 cores)"
  # Resume-friendly: only discard the build tree if it was configured differently
  # (e.g. an older shared-library config). A partial QuantLib build is ~30 min of
  # work and this machine drops its network / crashes; never throw that away blindly.
  QL_CACHE="${TP}/quantlib/build/CMakeCache.txt"
  if [ -f "${QL_CACHE}" ] && ! grep -q "^BUILD_SHARED_LIBS.*=OFF" "${QL_CACHE}"; then
    warn "quantlib build tree has a stale (shared) config — reconfiguring from scratch"
    rm -rf "${TP}/quantlib/build"
  elif [ -f "${QL_CACHE}" ]; then
    log "resuming existing quantlib build ($(find "${TP}/quantlib/build" -name '*.o' | wc -l | tr -d ' ') objects already compiled)"
  fi
  "${CMAKE}" -S "${TP}/quantlib/src" -B "${TP}/quantlib/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${NINJA}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS="${OPT_FLAGS}" \
    -DCMAKE_INSTALL_PREFIX="${TP}/quantlib/install" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBOOST_ROOT="${TP}/boost" -DBoost_INCLUDE_DIR="${TP}/boost" \
    -DQL_BUILD_EXAMPLES=OFF -DQL_BUILD_TEST_SUITE=OFF -DQL_BUILD_BENCHMARK=OFF
  "${CMAKE}" --build "${TP}/quantlib/build" --target install -j "${JOBS}"
  # Record the toolchain QuantLib was built with. tools/fingerprint.sh folds this into the perf-baseline
  # key, so a QuantLib built with a different compiler/flags than the engine can never be silently compared.
  cat > "${TP}/quantlib/install/TOOLCHAIN.json" <<EOT
{
  "quantlib": "${QL_VER}",
  "compiler": "$("${CXX_BIN}" --version | head -1)",
  "flags": "${OPT_FLAGS}",
  "built_utc": "$(date -u +%FT%TZ)"
}
EOT
fi
log "quantlib: third_party/quantlib/install (static, built with ${OPT_FLAGS})"

# ---- Summary -----------------------------------------------------------------
cat <<EOF

$(printf '\033[1;32m==> all dependencies ready\033[0m')

  compiler          $("${CXX_BIN}" --version | head -1)
  flags             ${OPT_FLAGS}
  cmake             ${CMAKE}
  ninja             ${NINJA}
  eigen             ${TP}/eigen
  boost (headers)   ${TP}/boost
  googletest        ${TP}/gtest/install
  google-benchmark  ${TP}/benchmark/install
  quantlib          ${TP}/quantlib/install

Next:
  ${CMAKE} -S "${ROOT}" -B "${ROOT}/build" -G Ninja -DCMAKE_MAKE_PROGRAM="${NINJA}" \\
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${TP}/gtest/install;${TP}/benchmark/install"
EOF
