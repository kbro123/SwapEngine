#pragma once
// SwapsEngine — header-only, templated-on-scalar engine.
// This header exists so the build/test wiring has something real to include
// before the curve/AD/calibration/portfolio modules land.

namespace swaps {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 0;
inline constexpr int version_patch = 0;  // Phase 0: scaffold

inline constexpr const char* version_string() { return "0.0.0-phase0"; }

}  // namespace swaps
