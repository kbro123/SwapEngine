# DetectISA.cmake — portable, automatic SIMD/ISA detection.
#
# Goal: this project must build optimally on ANY host (2017 Kaby Lake, Apple Silicon,
# an AVX-512 Xeon) without hard-coding a vector width anywhere.
#
# Sets in the parent scope:
#   SWAPS_ARCH_FLAGS            arch flag(s) to compile with (e.g. -march=native / -mcpu=native)
#   SWAPS_ISA_NAME              human-readable ISA level ("AVX2+FMA", "AVX-512", "NEON", "SSE2")
#   SWAPS_DOUBLE_PACKET_SIZE    doubles per SIMD register (1/2/4/8)
#   SWAPS_EIGEN_DEFINES         extra Eigen defines (e.g. EIGEN_ENABLE_AVX512)
#
# Options:
#   SWAPS_ARCH            native (default) | portable | <explicit flag, e.g. "-march=x86-64-v3">
#   SWAPS_ENABLE_AVX512   ON by default *when detected*; see the downclocking note below.

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

set(SWAPS_ARCH "native" CACHE STRING
    "Target ISA: 'native' (tune to this host), 'portable' (runs on any x86-64-v2 / armv8), or an explicit flag")
option(SWAPS_ENABLE_AVX512
    "Use AVX-512 (8-wide Eigen packets) when the host supports it. DEFAULT OFF: for this engine's small,
     mostly-bandwidth-bound matrices, 256-bit AVX2 packets MEASURED FASTER than AVX-512 on the AVX-512
     dev/CI Xeon — curve_build -13%, risk -29%, portfolio -13%, warm -16% — because heavy AVX-512
     downclocks the core. Native micro-arch tuning is still applied (-march=native); only the packet width
     is capped at 4 doubles. Turn ON to re-test on a core where AVX-512 doesn't throttle (e.g. AMD Zen4)." OFF)

# --- 1. Choose the architecture flag ------------------------------------------
# NOTE: clang accepts -mcpu=native on x86_64 too (it just doesn't mean what you want),
# so we branch on the target architecture rather than on "does the flag compile".
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(_swaps_is_arm TRUE)
else()
  set(_swaps_is_arm FALSE)
endif()

if(SWAPS_ARCH STREQUAL "native")
  if(_swaps_is_arm)
    set(_candidates "-mcpu=native" "-mcpu=apple-m1")
  elseif(NOT SWAPS_ENABLE_AVX512)
    # AVX-512 disabled (default): -march=native would still define __AVX512F__ and make Eigen use 8-wide
    # (512-bit) packets, which downclock and MEASURED SLOWER here. Target the AVX2 ISA level directly so
    # __AVX512F__ is undefined and Eigen caps at 256-bit (4-wide) packets. Runs on any x86-64-v3 host.
    set(_candidates "-march=x86-64-v3")
  else()
    set(_candidates "-march=native")
  endif()
elseif(SWAPS_ARCH STREQUAL "portable")
  # A binary that runs on any reasonably modern machine (no -native tuning).
  if(_swaps_is_arm)
    set(_candidates "")            # armv8-a baseline already implies NEON
  else()
    set(_candidates "-march=x86-64-v2")
  endif()
else()
  set(_candidates "${SWAPS_ARCH}")  # explicit user-supplied flag
endif()

set(SWAPS_ARCH_FLAGS "")
foreach(_flag IN LISTS _candidates)
  if(_flag STREQUAL "")
    set(SWAPS_ARCH_FLAGS "")
    break()
  endif()
  string(MAKE_C_IDENTIFIER "SWAPS_FLAG${_flag}" _cache_var)
  check_cxx_compiler_flag("${_flag}" ${_cache_var})
  if(${_cache_var})
    set(SWAPS_ARCH_FLAGS "${_flag}")
    break()
  endif()
endforeach()

if(NOT SWAPS_ARCH_FLAGS AND NOT SWAPS_ARCH STREQUAL "portable")
  message(WARNING "DetectISA: no usable arch flag from '${_candidates}' — building without ISA tuning.")
endif()

# --- 2. Probe which SIMD features that flag actually turns on -----------------
# We read the compiler's predefined macros, which is exactly what Eigen keys off.
set(CMAKE_REQUIRED_FLAGS "${SWAPS_ARCH_FLAGS}")

macro(_swaps_probe macro_name out_var)
  check_cxx_source_compiles("
    #ifndef ${macro_name}
    #error not available
    #endif
    int main() { return 0; }" ${out_var})
endmacro()

_swaps_probe(__AVX512F__  SWAPS_HAS_AVX512F)
_swaps_probe(__AVX2__     SWAPS_HAS_AVX2)
_swaps_probe(__AVX__      SWAPS_HAS_AVX)
_swaps_probe(__FMA__      SWAPS_HAS_FMA)
_swaps_probe(__SSE2__     SWAPS_HAS_SSE2)
_swaps_probe(__ARM_NEON   SWAPS_HAS_NEON)

unset(CMAKE_REQUIRED_FLAGS)

# --- 3. Derive the double-precision packet width -------------------------------
# These MUST agree with Eigen's packet_traits<double>::size. include/swaps/simd.hpp
# static_asserts that they do, so any drift is a hard compile error, not silent slowness.
set(SWAPS_EIGEN_DEFINES "")

if(SWAPS_HAS_AVX512F AND SWAPS_ENABLE_AVX512)
  # Eigen does NOT enable AVX-512 off __AVX512F__ alone; it requires this opt-in,
  # precisely because AVX-512 can be a performance regression on some cores.
  list(APPEND SWAPS_EIGEN_DEFINES EIGEN_ENABLE_AVX512)
  set(SWAPS_DOUBLE_PACKET_SIZE 8)
  set(SWAPS_ISA_NAME "AVX-512")
elseif(SWAPS_HAS_AVX512F AND NOT SWAPS_ENABLE_AVX512)
  set(SWAPS_DOUBLE_PACKET_SIZE 4)
  set(SWAPS_ISA_NAME "AVX2+FMA (AVX-512 present but disabled)")
elseif(SWAPS_HAS_AVX2)
  set(SWAPS_DOUBLE_PACKET_SIZE 4)
  if(SWAPS_HAS_FMA)
    set(SWAPS_ISA_NAME "AVX2+FMA")
  else()
    set(SWAPS_ISA_NAME "AVX2")
  endif()
elseif(SWAPS_HAS_AVX)
  set(SWAPS_DOUBLE_PACKET_SIZE 4)
  set(SWAPS_ISA_NAME "AVX")
elseif(SWAPS_HAS_NEON)
  set(SWAPS_DOUBLE_PACKET_SIZE 2)
  set(SWAPS_ISA_NAME "NEON")
elseif(SWAPS_HAS_SSE2)
  set(SWAPS_DOUBLE_PACKET_SIZE 2)
  set(SWAPS_ISA_NAME "SSE2")
else()
  set(SWAPS_DOUBLE_PACKET_SIZE 1)
  set(SWAPS_ISA_NAME "scalar (no SIMD detected)")
endif()

message(STATUS "DetectISA: arch=${CMAKE_SYSTEM_PROCESSOR} flags='${SWAPS_ARCH_FLAGS}' "
               "isa='${SWAPS_ISA_NAME}' doubles/register=${SWAPS_DOUBLE_PACKET_SIZE}")
