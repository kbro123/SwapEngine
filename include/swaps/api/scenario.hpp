#pragma once
// Scenario / stress seam (api/scenario.cpp): the production wiring of market::Scenario — the Tier-2
// analytics "what-if" primitive. It takes ONE bundle, calibrates it ONCE to an anchor state, then forks
// that fitted state per scenario ("fork over the market") and reports the shocked curve samples and,
// when a book is supplied, the book NPV under the shock and its delta vs the base.
//
//   scenario  {"scenario": {
//                "bundle": {...BundleProblem...},              (required) the curve bundle to anchor
//                "x0": [...],                                  (optional) calibration start; else flat guess
//                "regularize": {lambda, curves, tension, sigma}, (optional) same as the `bundle` verb
//                "scenarios": [                                (optional; empty => base only)
//                  {"name": "+25bp SOFR",
//                   "shift_curve": {"0": 25, "1": -5},         per-curve parallel forward shift in bp,
//                                                              keyed by INTEGER curve role (a bundle has
//                                                              no string curve names); overrides parallel_bp
//                   "parallel_bp": 25,                         global shift applied to every un-keyed curve
//                   "bump_fx": [{"base":"EUR","quote":"USD","rel":0.01}]}  relative FX bump (+1% => 0.01)
//                  , ...],
//                "book": {...MultiCurveBook...},               (optional) a book to reprice under each shock
//                "sample_times": [...]                         (optional) grid to sample every curve on
//              }}
//   -> {"scenario": {n_curves, n_knots,
//                    "base": {x, curves?, npv?, n?},
//                    "scenarios": [{name, curves?, npv?, npv_delta?, shift_bp?, parallel_bp?, fx?}, ...]}}
//
// The base is calibrated/priced exactly once; each scenario is a pure fork of the anchor state (the base
// is never mutated between scenarios). QuantLib-free; reuses market::Scenario for the curve-shift
// arithmetic and portfolio::MultiCurveBook for the (unchanged) book pricing path.

#include <string>

namespace swaps::api {

std::string scenario_json(const std::string& request);

}  // namespace swaps::api
