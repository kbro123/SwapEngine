#pragma once
// Historical / full-revaluation VaR seam (api/var.cpp): the P&L distribution of a book under a SET of
// market-move scenarios, plus its Value-at-Risk and Expected-Shortfall quantiles. This is the "full reval,
// not a delta-approximation, intraday" capability — every scenario is a real fork of the calibrated market
// (market::Scenario) repriced through the cached compiled reprice twin (portfolio::CompiledMultiCurveBook,
// the reprice_bound kernel), so a 250-day historical window is 250 µs-scale repricings against ONE
// calibration rather than a Taylor expansion around today.
//
// TWO input modes (pick one):
//   REVAL    — {"bundle", "book", "scenarios":[<move>, ...]}: calibrate the anchor once, fork+reprice the
//              book under each move, and build the P&L distribution pnl_k = npv_k − base_npv. A <move> is
//              the same declarative shock the `scenario` verb takes:
//                {"parallel_bp"?, "shift_curve"? {role: bp}, "bump_fx"? [{base, quote, rel}], "name"?}
//   SUPPLIED — {"pnl":[...]}: skip the engine and quantile a P&L series produced elsewhere (an external
//              full-reval or a realized daily-P&L history). "historical (or supplied) market moves."
//
//   scenario keys accepted at top level or under "var":
//     "quantiles": [0.95, 0.99]      (optional) confidence levels; default [0.95, 0.99]
//
//   -> {"var": {
//         mode: "reval" | "supplied",
//         n: <int>,                       number of scenarios / P&L points
//         base_npv?, n_positions?,        (reval, with a book)
//         mean_pnl, stdev_pnl,
//         pnl_sorted: [...],              ascending — the full distribution (a loss is negative)
//         quantiles: [ {q, var, es, var_pnl, es_pnl}, ... ]
//         reval_us?: <double>             (reval) engine wall time of the fork+reprice sweep (µs)
//       }}
//
// CONVENTION. `var` and `es` are LOSSES (positive numbers): a loss is −P&L. VaR at confidence q is the
// q-quantile of the loss distribution = −(the (1−q) lower-tail quantile of P&L), the lower tail found by
// the numpy-"type 7" linear interpolation of order statistics (position (1−q)·(N−1)). ES (Expected
// Shortfall / CVaR) at q is the mean of the worst (1−q) fraction of P&L: −mean of the m = max(1,
// round((1−q)·N)) smallest P&L order statistics. `var_pnl`/`es_pnl` echo the same numbers as SIGNED P&L
// (negative for a loss) so a caller need not re-negate. QuantLib-free; additive; a new verb off the gated
// hot path.

#include <string>

namespace swaps::api {

std::string var_json(const std::string& request);

}  // namespace swaps::api
