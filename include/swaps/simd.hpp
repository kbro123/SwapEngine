#pragma once
// SIMD width, portably.
//
// RULE (CLAUDE.md §5): never hard-code a vector width anywhere in the engine.
// The batched portfolio analytics must be written in terms of `packet_size<Scalar>`,
// so the same source compiles optimally to 2 doubles/register (SSE2, NEON),
// 4 (AVX/AVX2), or 8 (AVX-512) with no code change.

#include <Eigen/Core>
#include <cstddef>

#include "swaps/simd_config.hpp"  // generated: what CMake detected for this build

namespace swaps::simd {

/// Number of `Scalar`s that fit in one SIMD register for this build.
/// Compile-time constant. 1 when vectorization is unavailable/disabled.
template <class Scalar>
inline constexpr int packet_size = Eigen::internal::packet_traits<Scalar>::size;

/// Eigen's required alignment (bytes) for vectorized loads/stores.
inline constexpr std::size_t alignment_bytes = EIGEN_MAX_ALIGN_BYTES;

/// Round `n` up to a whole number of SIMD lanes, so batched loops never need a
/// scalar remainder path. Use this to pad the swap dimension of portfolio matrices.
template <class Scalar>
constexpr int padded_count(int n) {
  constexpr int w = packet_size<Scalar>;
  return ((n + w - 1) / w) * w;
}

// --- Consistency guard --------------------------------------------------------
// CMake detected a width independently (by probing compiler macros); Eigen derives
// its own from those same macros. If they ever disagree, the build is misconfigured
// (e.g. EIGEN_ENABLE_AVX512 not propagated, or EIGEN_DONT_VECTORIZE set) and the
// engine would silently run at the wrong width. Fail loudly instead.
//
// Define SWAPS_SKIP_ISA_CROSSCHECK to bypass (e.g. deliberately unvectorized builds).
#ifndef SWAPS_SKIP_ISA_CROSSCHECK
static_assert(packet_size<double> == detected::double_packet_size,
              "ISA mismatch: CMake's detected doubles/register disagrees with Eigen's "
              "packet_traits<double>::size. The build is misconfigured — check "
              "cmake/DetectISA.cmake and that SWAPS_EIGEN_DEFINES reached the compiler.");
#endif

}  // namespace swaps::simd
