// Bond seam: the stateless "bonds" run_json verb — street/yield-space bond math (curve-free), the classic
// desk calcs for a list of fixed-rate bonds: clean/dirty/accrued, yield-to-maturity, and modified/Macaulay
// duration + convexity. Mirrors sabr_calibrate (no bundle needed): a builder fills the plain bond structs
// (build/bond.hpp) from dates + terms, and the templated kernel (pricing/bond.hpp) prices them. Curve-space
// z-spread / PV off a calibrated curve is a follow-on verb on BundleSession. QuantLib-free.
//
// Units are the model's decimals (like sabr_calibrate): coupon/yield are absolute rates (0.045 = 4.5%),
// prices per unit notional (1.0 = par). The thin Python/Excel client converts to %/100-basis.
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bond.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/build/par_asset_swap.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace cvd = swaps::conventions;
namespace px = swaps::pricing;
namespace cal = swaps::calibration;

std::string bonds_json(const json::object& request) {
  return json::serialize(
      street_analytics_to_json(b::street_bonds(street_bond_request_from_json(sub(request, "bonds")))));
}

namespace {
// The calibrated curve sampled at a fixed set of times, exposed as the `discount(t)` interface the bond +
// asset-swap kernels want. Times are produced by the SAME curve_time calls that built the cashflows, so the
// keys are bit-identical — an exact lookup is safe.
struct SampledCurve {
  const std::vector<double>& t;
  const std::vector<double>& df;
  double discount(double x) const {
    for (std::size_t i = 0; i < t.size(); ++i)
      if (t[i] == x) return df[i];
    throw std::runtime_error("asset_swap: internal — a required curve time was not sampled");
  }
};
}  // namespace

// Stateless ASSET_SWAP verb: par asset-swap spread(s) for a list of bonds off a CALIBRATED curve. Request:
// {"asset_swap": {value_date, bundle, currency?, index?, curve?, bonds:[{issue, settle, maturity, coupon,
// freq?, clean?|dirty?}]}} -> SoA {asw_spread(decimal), clean_curve, dirty_curve, annuity, accrued, n}.
// A bond with no clean/dirty is priced par-par (purchase at par); with clean/dirty it's the proceeds spread.
std::string asset_swap_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o =
      top.contains("asset_swap") && top.at("asset_swap").is_object() ? top.at("asset_swap").as_object() : top;

  const std::string vd_iso = js(o, "value_date");
  if (vd_iso.empty()) throw std::invalid_argument("asset_swap: missing 'value_date'");
  if (!o.contains("bundle")) throw std::invalid_argument("asset_swap: missing 'bundle'");
  if (!o.contains("bonds") || !o.at("bonds").is_array())
    throw std::invalid_argument("asset_swap: missing 'bonds' array");
  const b::Date vd = b::Date::from_iso(vd_iso);
  const int curve = static_cast<int>(jd(o, "curve", 0.0));

  // Calibrate the bundle -> the discount curve we asset-swap against (same recipe as swaption_json).
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  const RegSpec reg = cal::smoothing_preset(cal::Smoothing::Light, static_cast<int>(sess.problem().curves.size()));
  sess.calibrate(flat_x0(sess.problem()), reg);

  std::vector<double> asw, clean_c, dirty_c, annuity_v, accrued_v;
  for (const auto& bv : o.at("bonds").as_array()) {
    const json::object& bo = bv.as_object();
    const std::string settle_iso = js(bo, "settle"), maturity_iso = js(bo, "maturity"),
                      issue_iso = js(bo, "issue");
    if (settle_iso.empty() || maturity_iso.empty() || issue_iso.empty())
      throw std::invalid_argument("asset_swap: each bond needs 'issue', 'settle', 'maturity'");
    if (!bo.contains("coupon")) throw std::invalid_argument("asset_swap: each bond needs 'coupon'");

    b::FixedBondTerms t;
    t.value_date = vd;
    t.settle = b::Date::from_iso(settle_iso);
    t.issue = b::Date::from_iso(issue_iso);
    t.maturity = b::Date::from_iso(maturity_iso);
    t.coupon = jd(bo, "coupon", 0.0);
    const std::string aconv = js(bo, "convention");
    if (aconv.empty()) throw std::invalid_argument("asset_swap: each bond needs 'convention' (a bonds[] row id, e.g. US-TREASURY)");
    const cvd::BondConv abc = cvd::require_bond(aconv);
    t.freq = bo.contains("freq") ? static_cast<int>(jd(bo, "freq", 0.0))
                                 : static_cast<int>(b::yield_convention(aconv).freq + 0.5);
    const b::BuiltBond bond = b::fixed_rate_bond(t);

    // ASW float leg on the bond currency's default swap product (frequency / day count / calendar / bdc from
    // the DB), backward from maturity to settlement (short front stub). Used to be hard-wired quarterly ACT/360.
    const b::SwapConv swc = b::swap_conv(std::string(abc.currency), js(bo, "index"));
    const int step_m = b::tok_months(swc.float_freq_tok);
    std::vector<b::Date> fd;
    for (b::Date d = t.maturity; d > t.settle; d = d.plus_months(-step_m)) fd.push_back(b::adjust(swc.calendar, d, swc.bdc));
    std::reverse(fd.begin(), fd.end());
    std::vector<double> fpay, ftau;
    b::Date prev = t.settle;
    for (const b::Date& pay : fd) {
      ftau.push_back(b::year_frac(swc.float_dc, prev, pay));
      fpay.push_back(b::curve_time(vd, pay));
      prev = pay;
    }

    // Sample the calibrated curve at every time the spread needs: settle, each bond flow, each float pay.
    std::vector<double> times{bond.curve.settle};
    for (const auto& cf : bond.curve.flows) times.push_back(cf.pay);
    times.insert(times.end(), fpay.begin(), fpay.end());
    const std::vector<CurveSample> cs = sess.sample(times);
    const SampledCurve sc{times, cs.at(static_cast<std::size_t>(curve)).discount};

    const bool has_clean = bo.contains("clean") && !bo.at("clean").is_null();
    const bool has_dirty = bo.contains("dirty") && !bo.at("dirty").is_null();
    const double dirty_purchase = has_dirty ? jd(bo, "dirty", 0.0)
                                  : has_clean ? jd(bo, "clean", 0.0) + bond.accrued
                                              : 1.0 + bond.accrued;  // par-par default
    const double dirty_curve = px::bond_dirty_price<double>(bond.curve, sc);
    asw.push_back(b::par_asset_swap_spread(bond.curve, sc, dirty_purchase, fpay, ftau));
    dirty_c.push_back(dirty_curve);
    clean_c.push_back(dirty_curve - bond.accrued);
    annuity_v.push_back(b::float_annuity(sc, sc.discount(bond.curve.settle), fpay, ftau));
    accrued_v.push_back(bond.accrued);
  }

  json::object out;
  out["asw_spread"] = vecf(asw);
  out["clean_curve"] = vecf(clean_c);
  out["dirty_curve"] = vecf(dirty_c);
  out["annuity"] = vecf(annuity_v);
  out["accrued"] = vecf(accrued_v);
  out["n"] = static_cast<int>(asw.size());
  return json::serialize(json::value(std::move(out)));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string bonds_json(const std::string& request) { return bonds_json(json::parse(request).as_object()); }
std::string asset_swap_json(const std::string& request) { return asset_swap_json(json::parse(request).as_object()); }
}  // namespace swaps::api
