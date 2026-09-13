#pragma once
// GenerateRisk seam: one portfolio, N curve bundles, ALL risk implied off the FIRST bundle's discount factors.
//
// The computation is cal::consistent_risk<Session> in include/swaps/calibration/consistent_risk.hpp (E7 stage 3.7 moved
// it, with null_completed_ladder -> calibration/risk.hpp, out of the api layer): the book is priced on bundle[0]'s
// calibrated curve, every other bundle is RE-LEVELED onto that curve, and the book's risk is re-expressed in each
// bundle's instrument basis, with any null direction of a bundle's Jacobian self-quoted as a synthetic pillar.
//
// KNOWN BUG, fixed separately (TASKS-ENGINE E7 "RISK SCALE BUGS" (1)): the ladder omits the residual market scale D,
// so a banded row is overstated by 1/decay and an FX row by q·T.
#include <string>

#include <boost/json/fwd.hpp>

namespace swaps::api {

// request = {"generate_risk": {book, bundles:[{curves, instruments}, ...], regularize?}}
// -> {npv, pv01, n, bundles:[{ladder, synthetic, synthetic_knot, npv, pv01, ladder_dv01, n_residuals, n_synthetic}]}.
// `regularize` only makes the re-leveling calibration of an under-determined bundle well-posed (the light preset is
// used when it is absent); the risk operator itself is rank-completed, never smoothed.
std::string generate_risk_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string generate_risk_json(const std::string& request);

}  // namespace swaps::api
