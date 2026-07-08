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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
option(SWAPS_NATIVE_ARCH "Build with -march=native (required for fair SIMD benchmarks)" ON)
if(SWAPS_NATIVE_ARCH)
  add_compile_options($<$<CONFIG:Release>:-march=native>)
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
  message(WARNING "QuantLib not found — run ./tools/bootstrap_deps.sh (oracle/baseline targets disabled)")
endif()

# ---- Test & benchmark trees (guarded so a fresh clone still configures) -----
option(SWAPS_BUILD_TESTS "Build correctness gate" ON)
option(SWAPS_BUILD_BENCH "Build performance gate" ON)

include(CTest)
if(SWAPS_BUILD_TESTS AND GTest_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
if(SWAPS_BUILD_BENCH AND benchmark_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt)
  add_subdirectory(bench)
endif()
