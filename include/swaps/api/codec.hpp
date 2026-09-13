#pragma once
// swaps::api codecs — THE JSON <-> engine object-graph mapping (E7 stage 2, PRINCIPLES.md P14).
//
// Every run_json verb, the C ABI and the pybind module decode their documents HERE and nowhere else. Until
// 2026-09-13 these were buried in api/bundle_api.cpp (reachable from a test only by sending a request through
// run_json), and six verbs re-implemented reg_from_json by hand -- disagreeing on malformed input.
//
// The contract (tests/codec_test.cpp pins each clause):
//   * an ABSENT field   -> the library struct's OWN default. A decoder never states a default value: there is
//                          exactly one, on the struct (cal::Instrument, px::FloatCoupon, api::RegSpec, ...).
//   * a PRESENT field   -> carried exactly; the wrong JSON type THROWS (never truncated, never ignored). An
//                          explicit null is the same as absent.
//   * an unknown name   -> throws std::invalid_argument (quote kind, interpolation scheme, position kind, pay).
//   * a field the library cannot default is REQUIRED and throws std::invalid_argument when absent: a booked
//     trade's notional / pay / fixed_rate / index / effective / maturity, and a position's fixed_curve when it
//     carries fixed coupons.
// Decoders carry no conventions lookups, no arithmetic and no invented defaults -- tools/verb_density.py locks
// that for api/codec.cpp. They may loop over arrays and branch on presence.
//
// Declared against a forward-declared CurveSample (defined in bundle_api.hpp, which includes this header), so
// including either header gives the whole API.

#include <vector>

#include <boost/json/fwd.hpp>

#include "swaps/build/bond.hpp"         // StreetBondRequest
#include "swaps/build/bond_future.hpp"  // DeliveryBasketRequest
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/regularize.hpp"  // RegSpec
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::api {

namespace cal = swaps::calibration;

using RegSpec = cal::RegSpec;  // calibration/regularize.hpp
struct CurveSample;  // bundle_api.hpp

// ---- the calibration problem -----------------------------------------------------------------------
cal::BundleProblem bundle_from_json(const boost::json::value& v);
boost::json::value bundle_to_json(const cal::BundleProblem& p);
cal::Instrument instrument_from_json(const boost::json::value& v);
boost::json::value instrument_to_json(const cal::Instrument& ins);

// ---- a book to reprice -----------------------------------------------------------------------------
// Reuses the SAME coupon JSON shapes the instrument codecs use (obs/pay/tau_pay/... for a FloatCoupon,
// pay/tau/scale for a FixedCoupon). Schema:
//   { "positions": [
//       { "kind":"swap", "notional":<double>, "fixed_rate":<double>,
//         "fwd_curve":<int>, "disc_curve":<int>, "float_coupons":[<FloatCoupon>...],
//         "fixed_curve":<int> (REQUIRED when fixed_coupons is non-empty), "fixed_coupons":[<FixedCoupon>...] },
//       { "kind":"xccy", "notional":<double>, "fx_spot":<double>,
//         "fwd_curve":<int>, "disc_curve":<int>, "float_coupons":[<FloatCoupon>...],   // domestic leg
//         "mtm_fwd_curve":<int>, "mtm_disc_curve":<int>,
//         "mtm_reset_num":<int>, "mtm_reset_den":<int>, "mtm_coupons":[<FloatCoupon>...] } ],
//     and/or BOOKED trades materialised through trade::Trade (each rolls under its own index's conventions):
//     "value_date": "YYYY-MM-DD", "curve_roles": {"<index id>": <curve>, ...},
//     "trades": [ {"id", "notional", "pay": "fixed"|"float", "fixed_rate", "currency", "index",
//                  "effective", "maturity", "csa": {"collateral_currency"} | "discount_index"} ] }
//   The discount index is trade::discount_index_for(csa, discount_index): the CSA decides when present.
swaps::portfolio::MultiCurveBook book_from_json(const boost::json::value& v);

// ---- street bond analytics (the `bonds` verb) ---------------------------------------------------------
// {value_date?, bonds:[{convention, settle, maturity, coupon, issue | dated + first_coupon, freq?, clean? | yield?}]}
// settle / maturity / coupon are required here; the rest of the rule (convention required, exactly one quote,
// dated needs first_coupon) is build::bond_from_terms's and pricing::street_analytics's.
swaps::build::StreetBondRequest street_bond_request_from_json(const boost::json::object& payload);
// -> SoA {clean, dirty, accrued, ytm, modified_duration, macaulay_duration, convexity, n}
boost::json::object street_analytics_to_json(const std::vector<swaps::pricing::StreetAnalytics>& rows);

// ---- bond-future delivery basket (the `bond_future` verb) ------------------------------------------------
// {contract, value_date, first_delivery, delivery?, futures_price, repo, notional_coupon?, round_months?,
//  basket:[{id?, convention?, settle?, issue | dated + first_coupon, maturity, coupon, freq?, clean}]}
swaps::build::DeliveryBasketRequest delivery_basket_request_from_json(const boost::json::object& payload);
// -> SoA {conversion_factor, gross_basis, net_basis, implied_repo, invoice_price, n, ctd_index, ctd_id}
boost::json::object delivery_basket_to_json(const swaps::build::DeliveryBasketResult& r);

// ---- request pieces shared by run_json, the verbs and the C ABI ------------------------------------
boost::json::array sample_to_json(const std::vector<CurveSample>& samples);
// request["regularize"] -> RegSpec. Absent or null => RegSpec{}; present and not an object => throws.
RegSpec reg_from_json(const boost::json::object& request);

}  // namespace swaps::api
