#pragma once
#include <boost/json.hpp>
// Credit seam declaration. The stateless "credit" run_json verb calibrates a hazard-rate (survival) curve
// to a strip of par CDS spreads given a nominal discount curve, then returns the calibrated hazard knots,
// the survival probability Q(t), the forward hazard h(t) and default density, and the per-instrument fitted
// par spreads. Curve-free of any bundle (like the "inflation" and "bonds" verbs): a builder fills the CDS
// instruments and the templated kernel calibrates them through the standard LM+AAD path. QuantLib-free.
#include <string>

namespace swaps::api {

// request = {"credit": {
//     recovery?       = 0.40,            // R: recovery rate (loss given default = 1 − R)
//     discount_zero?  = 0.0,             // flat continuous nominal discount rate; DF(t)=exp(−z·t)
//     discount_times? = [years...],      // OR an explicit nominal DF curve (log-linear in DF between nodes)
//     dfs?            = [DF...],         //    paired with discount_times; overrides discount_zero when given
//     premium_freq?   = 4,               // CDS premium coupons per year (quarterly)
//     prot_steps?     = 4,               // protection-integration subintervals per premium period
//     knot_times?     = [years...],      // hazard-curve back knots; default = the CDS maturities
//     output_times?   = [years...],      // where to report Q(t)/h(t); default = knot_times
//     instruments: [ {"type":"cds", "maturity": <years>, "spread": <par spread decimal>, "recovery"?} ]
//   }}
//   All times are curve-time year fractions (ACT/365F from the value date). Spreads are decimals
//   (150 bp = 0.0150). Premium accrual is ACT/360; protection is paid at period end (see build/credit_*).
// -> {"knot_times":[...], "hazards":[...],              // calibrated forward-hazard knots
//     "times":[...], "survival":[...],                  // survival probability Q(t) at output times
//     "hazard_curve":[...], "default_density":[...],    // forward hazard h(t) and −dQ/dt at output times
//     "par_spreads":[...],                              // per-instrument fitted model par spread
//     "recovery":..., "rms_residual":..., "rank_deficiency":..., "stationarity":..., "iterations":..., "n":...}
std::string credit_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string credit_json(const std::string& request);

}  // namespace swaps::api
