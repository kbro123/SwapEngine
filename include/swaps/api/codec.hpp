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
// Declared against a forward-declared CurveSample (defined in calibration/bundle_state.hpp, which bundle_api.hpp includes), so
// including either header gives the whole API.

#include <vector>

#include <boost/json/fwd.hpp>

#include "swaps/build/bond.hpp"         // StreetBondRequest
#include "swaps/build/bond_future.hpp"  // DeliveryBasketRequest
#include "swaps/build/par_asset_swap.hpp"  // AssetSwapBond
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/conventions_data.hpp"  // OverlayBatch, Registry::Listings
#include "swaps/calibration/regularize.hpp"  // RegSpec
#include "swaps/portfolio/portfolio.hpp"

// The RV request/result types (derive/bond_rv.hpp, derive/asset_swap.hpp) are forward-declared: including those
// headers here would pull portfolio/ and the LM into every bundle_api.hpp translation unit.
namespace swaps::derive {
struct BondUniverseRequest;
struct BondUniverseResult;
struct GovvieFitRequest;
struct GovvieFitResult;
struct SwapSpreadRequest;
struct SwapSpreadResult;
struct ScenarioRequest;  // derive/scenario.hpp
struct ScenarioResult;   // derive/scenario.hpp
struct ScenarioGridRequest;  // derive/scenario_grid.hpp
struct ScenarioGridResult;   // derive/scenario_grid.hpp
}  // namespace swaps::derive

namespace swaps::calibration {
struct QuoteDiagnostic;           // calibration/diagnostics.hpp
struct CalibrationReportRequest;  // calibration/diagnostics.hpp
struct CalibrationReport;         // calibration/diagnostics.hpp
struct ConsistentRiskRequest;     // calibration/consistent_risk.hpp
struct ConsistentRisk;            // calibration/consistent_risk.hpp
struct CurveSample;               // calibration/bundle_state.hpp
}  // namespace swaps::calibration

namespace swaps::api {

namespace cal = swaps::calibration;

using RegSpec = cal::RegSpec;  // calibration/regularize.hpp
using CurveSample = cal::CurveSample;  // calibration/bundle_state.hpp

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

// ---- bond relative value (the bond_universe / govvie_fit / swap_spread verbs) -----------------------------
// Bond rows are {id, issue, maturity, coupon, first_coupon?}, all required but first_coupon, stamped with the
// request's `convention`. Settlement overrides settle_calendar? / settle_lag? are the bond row's when absent.
// bond_universe: {value_date, convention, settle?, bonds, clean? | yield?}
swaps::derive::BondUniverseRequest bond_universe_request_from_json(const boost::json::object& payload);
boost::json::object bond_universe_to_json(const swaps::derive::BondUniverseResult& r);
// govvie_fit: {value_date, convention, bonds, clean, weight?, model? (spline | nelson_siegel | svensson),
//              meeting?, back?, tau1?, tau2?, x0?, settle_calendar?, settle_lag?}
swaps::derive::GovvieFitRequest govvie_fit_request_from_json(const boost::json::object& payload);
boost::json::object govvie_fit_to_json(const swaps::derive::GovvieFitResult& r);
// swap_spread: {value_date, convention, bond, clean, spread, index, tenor, swap_curve, factor_curve, anchor?,
//               spread_type? (headline | matched_maturity), settle_calendar?, settle_lag?}
swaps::derive::SwapSpreadRequest swap_spread_request_from_json(const boost::json::object& payload);
boost::json::object swap_spread_to_json(const swaps::derive::SwapSpreadResult& r);

// ---- asset swap (the `asset_swap` verb) ----------------------------------------------------------------------
// {value_date, bundle, curve?, bonds:[{convention, issue, settle, maturity, coupon, freq?, index?, clean? | dirty?}]}
struct AssetSwapRequest {
  swaps::build::Date value_date;
  cal::BundleProblem bundle;  // calibrated by the verb, then the bonds are asset-swapped against `curve`
  int curve = 0;              // the bundle curve to asset-swap against
  std::vector<swaps::build::AssetSwapBond> bonds;
};
AssetSwapRequest asset_swap_request_from_json(const boost::json::object& payload);
// -> SoA {asw_spread, clean_curve, dirty_curve, annuity, accrued, n}
boost::json::object asset_swap_to_json(const std::vector<swaps::build::AssetSwapAnalytics>& rows);

// ---- calibration diagnostics (the `calib_report` verb, and the session's quote_diagnostics document) ------------
// {bundle, x0?, regularize?}
cal::CalibrationReportRequest calib_report_request_from_json(const boost::json::object& payload);
// -> {rms_residual, converged, status, rank_deficiency, condition_number, singular_values,
//     quotes:[{model, target, residual, weight, in_band, soft, identifiability}], n}
boost::json::object calibration_report_to_json(const cal::CalibrationReport& r);
// -> [{model, target, residual, soft, [lower, upper, decay], in_band, weight}] (band keys on soft quotes only)
boost::json::array quote_diagnostics_to_json(const std::vector<cal::QuoteDiagnostic>& diagnostics);

// ---- consistent risk (the `generate_risk` verb) ------------------------------------------------------------------
// {book, bundles:[{curves, instruments}, ...], regularize?}
cal::ConsistentRiskRequest consistent_risk_request_from_json(const boost::json::object& payload);
// -> {npv, pv01, n, bundles:[{ladder, synthetic, synthetic_knot, npv, pv01, ladder_dv01, n_residuals, n_synthetic}]}
boost::json::object consistent_risk_to_json(const cal::ConsistentRisk& r);

// ---- the conventions registry (the `conventions` / `list_conventions` verbs) -------------------------------------
// {"conventions": {family: {id: row}}} (or the families object itself) -> ONE OverlayBatch, rows in the exact
// conventions.json shapes; the field mapping is tools/gen_conventions_hpp.py's for the baked arrays. What a row must
// CONTAIN is Registry::apply's rule (conventions_db.hpp). This checks only what the JSON alone can show: a value of the
// wrong JSON type, a negative or fractional integer, a date that is not a calendar date, one leg under two names
// (float_leg | spread_leg | usd_leg, flat_leg | eur_leg), a family the registry does not have, and the JSON-only fields
// the schema requires (a product's description; a calendar's weekend, observance with holidays, holidays or join).
// The batch's string_views point into `payload`, which must outlive Registry::apply.
swaps::conventions::OverlayBatch overlay_batch_from_json(const boost::json::object& payload);
// {"added": {family: [id, ...]} for each family with rows, "overlay_size": n}
boost::json::object overlay_added_to_json(const swaps::conventions::OverlayBatch& batch, int overlay_size);
// {family: {"baked": [...], "overlay": [...]}, ..., "overlay_size": n}
boost::json::object conventions_listing_to_json(const swaps::conventions::Registry::Listings& listing);

// ---- scenario (the `scenario` verb) ------------------------------------------------------------------------------
// {"scenario": {bundle, x0?, regularize?, sample_times?, book?, scenarios: [{name?, parallel_bp?, shift_curve?:
// {"<role>": bp}, bump_fx?: [{base, quote, rel}]}]}} (or the body itself). A shift_curve key must be an integer curve
// role.
swaps::derive::ScenarioRequest scenario_request_from_json(const boost::json::object& payload);
// {"scenario": {n_curves, n_knots, base: {x, curves?, npv?, n?}, scenarios: [{name, curves?, npv?, npv_delta?,
// parallel_bp?, shift_bp?, fx?}]}} -- curves when sample_times were given, npv when a book was.
boost::json::object scenario_result_to_json(const swaps::derive::ScenarioResult& r);

// ---- scenario grid (the `scenario_grid` verb) --------------------------------------------------------------------
// {"scenario_grid": {bundle, axes: [1 or 2 of {label?, kind?: parallel_bp | shift_curve | fx, role (shift_curve),
// base + quote (fx), values}], x0?, regularize?, sample_times?, book?}} (or the body itself). kind absent = the axis
// struct's own (parallel_bp); an unknown kind throws.
swaps::derive::ScenarioGridRequest scenario_grid_request_from_json(const boost::json::object& payload);
// {"scenario_grid": {n_curves, n_knots, axes, shape: [n0, n1], base: {x, curves?, npv?, n?}, npv?, pnl?, grid_us?,
// n_cells?}} -- the surface fields when a book was given.
boost::json::object scenario_grid_result_to_json(const swaps::derive::ScenarioGridResult& r);

// ---- request pieces shared by run_json, the verbs and the C ABI ------------------------------------
boost::json::array sample_to_json(const std::vector<CurveSample>& samples);
// request["regularize"] -> RegSpec. Absent or null => RegSpec{}; present and not an object => throws.
RegSpec reg_from_json(const boost::json::object& request);

}  // namespace swaps::api
