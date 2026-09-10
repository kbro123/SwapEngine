#pragma once
#include <boost/json.hpp>
// P&L EXPLAIN seam: the stateless "pnl" run_json verb. Decomposes a book's NPV change between two
// dates/markets (bundle0 @ t0, bundle1 @ t1, dt = t1 - t0) into CARRY, ROLL-DOWN, MARKET-MOVE and a
// closing RESIDUAL — see include/swaps/calibration/pnl_explain.hpp for the exact contract. QuantLib-free.
#include <string>

namespace swaps::api {

// Request (top level or under "pnl"):
//   { "bundle0": {...BundleProblem...},   (required) curve topology + q0 (targets) + calibrated x0
//     "bundle1": {...BundleProblem...},   (optional) SAME structure; q1 + calibrated x1 (absent => q1=q0)
//     "book":    {...MultiCurveBook...},  (required) the positions to attribute
//     "dt_years": <double>,               (optional, default 0) t1 - t0 in curve time
//     "x0": [...], "x1": [...],           (optional) override the calibrated decomposition states
//     "regularize": {lambda,curves,tension,sigma} }   (optional) smooths the risk operator / ladder
// Response: { "pnl": { total, carry, roll, market, residual, npv_t0, npv_t1,
//                      market_ladder:[...], dq:[...], dt_years, n } }.  On error: { "error": "..." }.
std::string pnl_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string pnl_json(const std::string& request);

}  // namespace swaps::api
