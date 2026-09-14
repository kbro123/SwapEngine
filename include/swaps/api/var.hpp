#pragma once
#include <boost/json.hpp>
// Historical / full-revaluation VaR seam (api/var.cpp -> derive/var.hpp): the P&L distribution of a book under a SET of
// market moves, plus its Value-at-Risk and Expected-Shortfall quantiles. Every move is a real fork of the calibrated
// market repriced through the cached compiled twin (portfolio::XccyFxScaledBooks), so a 250-day window is 250
// µs-scale repricings against ONE calibration rather than a Taylor expansion around today.
//
// EXACTLY ONE input mode (both, or neither, is refused):
//   REVAL    — {"bundle", "book", "scenarios":[<move>, ...], "x0"?, "regularize"?}: calibrate the anchor once, fork and
//              reprice under each move: pnl_k = npv_k − base_npv. A <move> is the `scenario` verb's shape and rule
//              (an explicit shift_curve ADDS to parallel_bp):
//                {"parallel_bp"?, "shift_curve"? {"<role>": bp}, "bump_fx"? [{base, quote, rel}], "name"?}
//   SUPPLIED — {"pnl":[...]} (non-empty): quantile a P&L series produced elsewhere.
//
//   at top level or under "var":
//     "quantiles": [0.95, 0.99]      (optional) confidence levels in (0, 1); absent = [0.95, 0.99]; [] is refused
//
//   -> {"var": {
//         mode: "reval" | "supplied",
//         base_npv?, n_positions?, reval_us?,   (reval) reval_us = wall time of the fork+reprice sweep (µs)
//         n, mean_pnl, stdev_pnl,
//         pnl_sorted: [...],              ascending — the full distribution (a loss is negative)
//         quantiles: [ {q, var, es, var_pnl, es_pnl}, ... ]
//       }}
//
// CONVENTION. `var` and `es` are LOSSES (positive numbers): a loss is −P&L. VaR at confidence q is −(the (1−q)
// lower-tail P&L quantile, numpy "type 7" interpolation at position (1−q)·(N−1)). ES at q is −(the mean of the m =
// clamp(round((1−q)·N), 1, N) smallest P&Ls). `var_pnl`/`es_pnl` are the same numbers as SIGNED P&L.

#include <string>

namespace swaps::api {

std::string var_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string var_json(const std::string& request);

}  // namespace swaps::api
