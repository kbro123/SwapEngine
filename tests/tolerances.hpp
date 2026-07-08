#pragma once
// Single source of truth for numerical tolerances used by the correctness gate.
// Changing any of these requires a documented reason in the commit message (see CLAUDE.md §3).

namespace swaps::tol {

// Discount factors / par rates vs the QuantLib oracle.
inline constexpr double curve_rel = 1e-10;

// AAD Jacobian vs QuantLib bump-and-reprice (bump noise dominates the bound).
inline constexpr double jacobian_rel = 1e-6;

// Vectorized portfolio analytics vs per-swap QuantLib pricing.
inline constexpr double analytics_rel = 1e-10;

}  // namespace swaps::tol
