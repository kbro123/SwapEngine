#pragma once
#include <boost/json.hpp>
// RV seam — the stateless bond-RV / swap-spread verbs (api/rv.cpp): the production entry points of the
// derive/ + market/ layers. Each takes/returns a JSON document string (the run_json convention).
//
//   bond_universe  {"bond_universe": {value_date, settle?, convention, bonds:[{id, issue, maturity,
//                   coupon, first_coupon?}], clean:[...] | yield:[...]}}
//                  -> batched yields/cleans + modified duration + convexity + accrued (vectorized
//                     portfolio::BondUniverse — one Horner/Newton sweep, not a per-bond loop).
//   govvie_fit     {"govvie_fit": {value_date, convention, bonds:[...], clean:[...], model:
//                   "spline"|"nelson_siegel"|"svensson", meeting/back (spline) | tau1/tau2 (parametric),
//                   weight?:[...], settle_calendar?, settle_lag?, x0?}}
//                  -> the minimum-pricing-error fit: fitted state x, rms, per-bond fair-value residuals
//                     (the RV ladder), and per-bond z-spreads off the fitted curve (spline model).
//   swap_spread    {"swap_spread": {value_date, convention, bond:{...}, clean, spread, index, tenor,
//                   swap_curve, factor_curve, anchor?, spread_type?, settle_calendar?, settle_lag?}}
//                  -> the headline derivation: benchmark street yield (WI-aware) + the {pin, asw}
//                     calibration rows of the asset-swap BASIS, as instrument JSON for a bundle spec.
//
// REQUIRED (since E7 stage 3.4, 2026-09-13): each bond row's id / issue / maturity / coupon; swap_spread's clean,
// spread, index, tenor, swap_curve and factor_curve (they used to default to 0 / "" / 0 / 1). An unknown `model` or
// `spread_type` throws (spread_type used to fall back to headline silently); govvie_fit rejects duplicate bond ids
// and a `weight` of the wrong length. Absent settle_calendar / settle_lag are the bond convention row's. KNOWN GAP:
// spread_type "matched_maturity" still matches the TENOR swap (TASKS-ENGINE E7 3.4), pending an owner decision.

#include <string>

namespace swaps::api {

std::string bond_universe_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string bond_universe_json(const std::string& request);
std::string govvie_fit_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string govvie_fit_json(const std::string& request);
std::string swap_spread_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string swap_spread_json(const std::string& request);

}  // namespace swaps::api
