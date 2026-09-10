#pragma once
#include <boost/json.hpp>
// Inflation seam declaration. The stateless "inflation" run_json verb calibrates a breakeven-inflation
// curve to a strip of Zero-Coupon (ZCIS) and Year-on-Year (YoY) inflation-swap quotes, then returns the
// calibrated curve (breakeven forward knots), the projected price-index level I(t), the annual zero-coupon
// breakevens, and the per-instrument fitted breakevens. Curve-free of any bundle (like the "bonds" verb):
// a builder fills the inflation instruments and the templated kernel calibrates them through the standard
// LM+AAD path. QuantLib-free.
#include <string>

namespace swaps::api {

// request = {"inflation": {
//     base?           = 100.0,            // I(0), the base CPI level
//     nominal_zero?   = 0.0,              // flat continuous nominal zero rate (YoY par-rate weighting only)
//     seasonality?    = [12 doubles],     // per-month log factors (recentred to sum 0); omitted => none
//     knot_times?     = [years...],       // breakeven-curve back knots; default = the instrument maturities
//     output_times?   = [years...],       // where to report I(t)/breakevens; default = knot_times
//     instruments: [
//        {"type":"zcis", "maturity": <years>, "rate": <annually-compounded breakeven>},
//        {"type":"yoy",  "years": <int> | "ends":[years...], "rate": <par fixed rate>, "nominal_zero"?}
//     ]}}
//   All times are curve-time year fractions (ACT/365F from the value date). Rates are decimals (0.025 = 2.5%).
// -> {"knot_times":[...], "forwards":[...],            // calibrated breakeven instantaneous-forward knots
//     "times":[...], "index":[...], "growth":[...],    // projected index level I(t) and I(t)/I(0)
//     "forward_breakeven":[...], "zc_breakeven":[...], // fwd breakeven f(t) and annual ZC breakeven at t
//     "breakevens":[...],                              // per-instrument fitted model breakeven/par rate
//     "stationarity":..., "rms_residual":..., "rank_deficiency":..., "iterations":..., "n":...}
std::string inflation_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string inflation_json(const std::string& request);

}  // namespace swaps::api
