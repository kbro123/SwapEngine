cmake_minimum_required(VERSION 3.20)
project(SwapsEngine LANGUAGES CXX)

# ---- Standards & defaults --------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Aggressive, reproducible optimization for the engine + benchmarks.
# -fno-math-errno: math functions (exp/pow/…) never set errno, so the compiler can inline/vectorize them
# freely and drop the errno-store barrier. This is VALUE-PRESERVING — it changes error reporting, not
# results — so the ~1e-10 QuantLib oracle and the internal bit-identical gates are unaffected. (Deliberately
# NOT -ffast-math / -funsafe-math / reassociation, which WOULD break those gates — see
# docs/calibration-overhead-analysis.md.)
# These are GCC/Clang flags; on MSVC keep its own default Release flags (/O2 /Ob2 /DNDEBUG) — MSVC has no
# -O3/-fno-math-errno and its default float model is already value-preserving. (Overwriting the string on
# MSVC would silently drop /O2 and ship an unoptimised DLL.)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")
endif()

# ---- Automatic ISA/SIMD detection (portable across hosts) -------------------
# Detects AVX2 / AVX-512 / NEON / SSE2 on whatever machine this is built on and
# derives the doubles-per-register width. Nothing in the engine hard-codes a width.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(DetectISA)

if(SWAPS_ARCH_FLAGS)
  add_compile_options($<$<CONFIG:Release>:${SWAPS_ARCH_FLAGS}>)
endif()

# ccache if available.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# ---- The engine: header-only, templated on the scalar type -----------------
add_library(swaps INTERFACE)
target_include_directories(swaps INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(swaps INTERFACE cxx_std_20)

# MSVC needs /bigobj: bundle_api.cpp (header-only Boost.JSON + the templated instrument model in one TU)
# exceeds the COFF object-section limit (C1128). Generator-expression-guarded, so it is a no-op on
# GCC/Clang. Propagates to every target that links `swaps` (the api facade, tests, benches).
target_compile_options(swaps INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

# Generated header recording what DetectISA found (swaps/simd_config.hpp).
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/simd_config.hpp.in
  ${CMAKE_BINARY_DIR}/generated/swaps/simd_config.hpp
  @ONLY)
target_include_directories(swaps INTERFACE ${CMAKE_BINARY_DIR}/generated)

# Eigen opt-ins implied by the detected ISA (e.g. EIGEN_ENABLE_AVX512).
if(SWAPS_EIGEN_DEFINES)
  target_compile_definitions(swaps INTERFACE ${SWAPS_EIGEN_DEFINES})
endif()

# LICENCE, compiler-enforced (see THIRD_PARTY_NOTICES.md): Eigen is MPL-2.0 except for a handful of modules
# carrying LGPL-2.1 third-party code (SimplicialCholesky, AMD ordering, constrained_cg). EIGEN_MPL2_ONLY makes
# including any of them a COMPILE ERROR, so "the shipped engine links nothing copyleft" is a build invariant
# rather than an audit finding. The engine uses Core/Dense/SparseCore only, so this costs nothing today.
target_compile_definitions(swaps INTERFACE EIGEN_MPL2_ONLY)

# ---- Vendored dependencies (third_party/, populated by tools/bootstrap_deps.sh) ----
# Homebrew is NOT used: macOS 13 / Intel is Homebrew Tier 3 (no bottles). See CLAUDE.md §4.
set(SWAPS_THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Eigen (linear algebra + LevenbergMarquardt + AutoDiffScalar) — header-only.
if(EXISTS ${SWAPS_THIRD_PARTY}/eigen/Eigen)
  target_include_directories(swaps INTERFACE ${SWAPS_THIRD_PARTY}/eigen)
  message(STATUS "Eigen: vendored (${SWAPS_THIRD_PARTY}/eigen)")
else()
  find_package(Eigen3 3.4 QUIET)
  if(Eigen3_FOUND)
    target_link_libraries(swaps INTERFACE Eigen3::Eigen)
    message(STATUS "Eigen: system")
  else()
    message(WARNING "Eigen not found — run ./tools/bootstrap_deps.sh")
  endif()
endif()

# ---- Dev/baseline dependencies ---------------------------------------------
list(PREPEND CMAKE_PREFIX_PATH
  ${SWAPS_THIRD_PARTY}/gtest/install
  ${SWAPS_THIRD_PARTY}/benchmark/install)

find_package(GTest QUIET)
find_package(benchmark QUIET)

# QuantLib: correctness oracle + speed baseline. Built from source with OUR flags
# (perf-gate integrity, CLAUDE.md §3) — never a generically-compiled package.
set(QL_ROOT ${SWAPS_THIRD_PARTY}/quantlib/install)
find_library(QUANTLIB_LIB NAMES QuantLib HINTS ${QL_ROOT}/lib NO_DEFAULT_PATH)
find_path(QUANTLIB_INCLUDE ql/quantlib.hpp HINTS ${QL_ROOT}/include NO_DEFAULT_PATH)
if(QUANTLIB_LIB AND QUANTLIB_INCLUDE)
  set(QuantLib_FOUND TRUE)
  add_library(quantlib_oracle INTERFACE)
  target_include_directories(quantlib_oracle INTERFACE ${QUANTLIB_INCLUDE} ${SWAPS_THIRD_PARTY}/boost)
  target_link_libraries(quantlib_oracle INTERFACE ${QUANTLIB_LIB})
  message(STATUS "QuantLib oracle: ${QUANTLIB_LIB}")
else()
  # PRINCIPLES.md P9: the correctness gate cannot pass with QuantLib absent. Its absence used to be a
  # WARNING that silently dropped the oracle binary (135 tests) and every reference bench.
  option(SWAPS_ALLOW_NO_ORACLE "Configure without QuantLib (NOT a gated build; oracle tests + reference benches disabled)" OFF)
  if(SWAPS_ALLOW_NO_ORACLE)
    message(WARNING "QuantLib not found — SWAPS_ALLOW_NO_ORACLE=ON: this build is NOT gate-eligible")
  else()
    message(FATAL_ERROR "QuantLib not found (expected ${QL_ROOT}). Run ./tools/bootstrap_deps.sh, or pass -DSWAPS_ALLOW_NO_ORACLE=ON for a non-gated build.")
  endif()
endif()

# ---- Test & benchmark trees (guarded so a fresh clone still configures) -----
option(SWAPS_BUILD_TESTS "Build correctness gate" ON)
option(SWAPS_BUILD_BENCH "Build performance gate" ON)

include(CTest)

# Public API facade (JSON contract + BundleSession). QuantLib-free; guarded on the vendored Boost.
if(GTest_FOUND)
  include(GoogleTest)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/api/CMakeLists.txt AND EXISTS ${SWAPS_THIRD_PARTY}/boost)
  add_subdirectory(api)
endif()

# QuantLib calendar dump (tools/calendars_from_ql.py regenerates the tabulated calendars from it).
if(TARGET quantlib_oracle)
  add_executable(ql_calendars_dump tools/ql_calendars_dump.cpp)
  target_link_libraries(ql_calendars_dump PRIVATE quantlib_oracle)
endif()

if(SWAPS_BUILD_TESTS AND GTest_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
if(SWAPS_BUILD_BENCH AND benchmark_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt)
  add_subdirectory(bench)
endif()
