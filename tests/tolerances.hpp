#pragma once
// Shared tolerances of the correctness gate (E5, 2026-09-10). A test that needs a LOOSER bound states it
// inline WITH the reason (a discretisation, a bump-noise floor, a documented approximation, a measured
// regression pin) -- a bare literal with no reason is a review finding. Changing a constant here needs a
// documented reason in the commit message (CLAUDE.md §3).

namespace swaps::tol {

// ---- T1 oracle: engine number vs QuantLib number ---------------------------------------------------
inline constexpr double curve_rel = 1e-10;      // discount factors / par rates
inline constexpr double jacobian_rel = 1e-6;    // AAD Jacobian vs QuantLib bump-and-reprice (bump noise)
inline constexpr double analytics_rel = 1e-10;  // vectorised portfolio analytics vs per-swap QuantLib

// ---- T3 cross-path parity: two engine paths on the same double inputs -------------------------------
inline constexpr double parity_value = 1e-12;     // compiled vs templated vs AAD values (rounding only)
inline constexpr double parity_jacobian = 1e-9;   // analytic vs AAD Jacobian entries (relative to max(1,|J|))

// ---- T2 calibration: a fitted curve vs its own market ------------------------------------------------
inline constexpr double reprice = 1e-9;        // model quote vs market on a consistent (zero-residual) fixture
inline constexpr double stationarity = 1e-7;   // ||Jᵀr||∞ at a least-squares optimum
inline constexpr double step_tol = 1e-9;       // the streamer's committed-x contract (StreamingCalibrator)

// ---- T5 literals: an independently derived closed-form / hand value -----------------------------------
inline constexpr double literal = 1e-12;       // double-precision agreement with an independent evaluation

}  // namespace swaps::tol
