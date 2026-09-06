#pragma once
// Calibration-diagnostics seam declaration. The stateless "calib_report" run_json verb takes the SAME
// input as a normal `bundle` calibration (a resolved BundleProblem object graph, an optional x0 seed, an
// optional smoothness regulariser), calibrates it through the standard BundleSession path, then reports the
// HEALTH of that calibration read off the calibration Jacobian J = dq/dx (n_residuals x n_knots) at the
// solution:
//   * condition_number = sigma_max / sigma_min of J (Eigen JacobiSVD), with the full singular spectrum;
//   * rank_deficiency  = the engine's own report (CalibrationResult::rank_deficiency, the count of
//     numerically unconstrained state directions at the shared kRankThreshold);
//   * per-quote identifiability = the diagonal of the model-resolution (hat) matrix H = J*M, where
//     M = risk_operator = (JᵀJ + RᵀR)⁻¹Jᵀ = dx/dq. h_ii in [0,1] is how well quote i pins its own pillar:
//     ~1 = self-identified, ->0 = redundant/collinear (a near-duplicate instrument shares its leverage with
//     the pillar it collides with, so both drop). trace(H) = rank(J) = effective number of constrained knots.
//   * the existing per-quote in-band diagnostics (target/model/residual/weight/in_band), so this one verb is
//     the "one-stop calibration health" report.
// Additive and QuantLib-free; it reuses only BundleSession's public jacobian()/risk_operator()/
// quote_diagnostics_json(), never the internal calibration types. Units are model decimals (0.025 = 2.5%).
#include <string>

namespace swaps::api {

// request = {"calib_report": {                     // (the "calib_report" wrapper is optional; may be top-level)
//     bundle:      {...},                          // (required) the BundleProblem object graph (as `bundle`)
//     x0?:         [doubles],                      // (optional) start; else a flat market-implied guess
//     regularize?: {lambda, curves:[...],          // (optional) smoothness penalty (same shape as `bundle`);
//                   tension?:bool, sigma?}          //   affects M/identifiability through RᵀR, never J itself
//   }}
// -> {"rms_residual":..., "rank_deficiency":..., "condition_number":..., "singular_values":[...],
//     "quotes":[{"model","target","residual","weight","in_band","soft","identifiability"}, ...], "n":...}
std::string calib_report_json(const std::string& request);

}  // namespace swaps::api
