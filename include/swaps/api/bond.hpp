#pragma once
#include <boost/json.hpp>
// Bond seam declaration. The stateless "bonds" run_json verb does street/yield-space bond math (curve-free,
// no bundle): clean/dirty/accrued, yield-to-maturity, and modified/Macaulay duration + convexity for a list
// of fixed-rate bonds. See api/bond.cpp for the schema. Curve-space z-spread/PV off a calibrated curve is a
// follow-on verb on BundleSession.
#include <string>

namespace swaps::api {

// request = {"bonds": {value_date?, bonds:[{
//     convention?="US-TREASURY",           // a conventions/conventions.json bond id (see below)
//     settle, maturity, coupon,
//     issue,                               // SEASONED bond: the dated date
//     dated, first_coupon,                 // or WHEN-ISSUED: replaces `issue`; both required together
//     freq?,                               // overrides the convention's frequency
//     clean?|yield? }]}}                   // exactly one; the other is solved
// -> flat SoA {clean, dirty, accrued, ytm, modified_duration, macaulay_duration, convexity, n}.
//
// `convention` selects the YIELD convention — specifically how the fractional first period is discounted,
// which is the one thing the per-flow exponent cannot encode (pricing/bond.hpp YieldConvention):
//   "US-TREASURY"      US street: compound stub, SIMPLE once only the final coupon remains (~0.7 bp).
//   "US-TREASURY-TSY"  31 CFR Part 356 App B / the Bloomberg "Treasury method": simple stub always.
// The ids and their fields come from the conventions DB, shared with the Python web layer; an unknown id
// is an error rather than a silent default.
//
// WHEN-ISSUED: pass `dated` + `first_coupon`. A new issue settles ON the dated date (zero accrued); a
// reopening passes a later `settle` inside the first period. The short first coupon is prorated to actual
// days per App B. Pair it with "US-TREASURY-TSY" to get the convention the regulation is written for.
//
// Units are the model's decimals: coupon/yield are absolute rates (0.045 = 4.5%), prices are per unit
// notional (1.0 = par). Seasoned bonds must have regular coupon periods (an odd LAST stub throws; an odd
// first period is the when-issued path above).
std::string bonds_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string bonds_json(const std::string& request);

// Stateless asset-swap verb (curve-space): par asset-swap spread(s) for bonds off a CALIBRATED bundle.
// request = {"asset_swap": {value_date, bundle, currency?, index?, curve?, bonds:[{issue, settle, maturity,
// coupon, freq?, clean?|dirty?}]}} -> SoA {asw_spread(decimal), clean_curve, dirty_curve, annuity, accrued,
// n}. Par-par when no price is given; proceeds spread when clean/dirty is supplied.
std::string asset_swap_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string asset_swap_json(const std::string& request);

}  // namespace swaps::api
