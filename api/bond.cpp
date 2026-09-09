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

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
std::string js(const json::object& o, const char* k, const std::string& d = "") {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return d;
  return std::string(o.at(k).as_string().c_str());
}
json::array vecf(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}
}  // namespace

std::string bonds_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("bonds") && top.at("bonds").is_object() ? top.at("bonds").as_object()
                                                                               : top;
  if (!o.contains("bonds") || !o.at("bonds").is_array())
    throw std::invalid_argument("bonds: missing 'bonds' array");
  const std::string vd_top = js(o, "value_date");  // optional curve reference; street calcs don't use it

  std::vector<double> clean, dirty, accrued, ytm, mdur, macdur, convx;
  const json::array& arr = o.at("bonds").as_array();
  for (auto& r : {&clean, &dirty, &accrued, &ytm, &mdur, &macdur, &convx}) r->reserve(arr.size());

  for (const auto& bv : arr) {
    const json::object& bo = bv.as_object();
    const std::string settle_iso = js(bo, "settle");
    const std::string maturity_iso = js(bo, "maturity");
    const std::string issue_iso = js(bo, "issue");
    if (settle_iso.empty() || maturity_iso.empty() ||
        (issue_iso.empty() && js(bo, "dated").empty()))
      throw std::invalid_argument(
          "bonds: each bond needs 'settle', 'maturity' and either 'issue' (seasoned) or "
          "'dated'+'first_coupon' (when-issued), all YYYY-MM-DD");
    if (!bo.contains("coupon")) throw std::invalid_argument("bonds: each bond needs 'coupon'");

    // Yield CONVENTION. Named conventions come from conventions/conventions.json (the same DB the web
    // layer reads), so a caller says "US-TREASURY" (street) or "US-TREASURY-TSY" (31 CFR App B / the
    // Bloomberg Treasury method) rather than encoding the discounting rule. Explicit `freq` still
    // overrides for a bond that is not one of the catalogued types.
    const std::string conv_id = js(bo, "convention");
    if (conv_id.empty()) throw std::invalid_argument("bonds: each bond needs 'convention' (a bonds[] row id, e.g. US-TREASURY)");
    const px::YieldConvention yc = b::yield_convention(conv_id);

    const b::Date value = b::Date::from_iso(vd_top.empty() ? settle_iso : vd_top);
    const b::Date settle_d = b::Date::from_iso(settle_iso);
    const b::Date maturity_d = b::Date::from_iso(maturity_iso);
    const double cpn = jd(bo, "coupon", 0.0);
    const int freq = bo.contains("freq") ? static_cast<int>(jd(bo, "freq", 0.0))
                                         : static_cast<int>(yc.freq + 0.5);

    // WHEN-ISSUED / odd-first-period bond: signalled by `dated` + `first_coupon`. A new issue settles ON
    // the dated date (zero accrued); a reopening passes a later `settle` within the first period. The
    // short first coupon is prorated per 31 CFR Part 356 App B.
    const std::string dated_iso = js(bo, "dated"), first_cpn_iso = js(bo, "first_coupon");
    if (dated_iso.empty() != first_cpn_iso.empty())
      throw std::invalid_argument("bonds: a when-issued bond needs BOTH 'dated' and 'first_coupon'");
    const bool is_wi = !dated_iso.empty();

    b::BuiltBond bb = [&] {
      if (is_wi) {
        const b::Date dated = b::Date::from_iso(dated_iso);
        return b::when_issued_bond(value, dated, b::Date::from_iso(first_cpn_iso), maturity_d, cpn, freq,
                                   settle_iso.empty() ? dated : settle_d, yc.stub, yc.final_period_simple);
      }
      b::FixedBondTerms t;
      t.value_date = value;  // unused in street space
      t.settle = settle_d;
      t.issue = b::Date::from_iso(issue_iso);
      t.maturity = maturity_d;
      t.coupon = cpn;
      t.freq = freq;
      t.stub = yc.stub;
      t.final_period_simple = yc.final_period_simple;
      return b::fixed_rate_bond(t);
    }();

    const bool has_clean = bo.contains("clean") && !bo.at("clean").is_null();
    const bool has_yield = bo.contains("yield") && !bo.at("yield").is_null();
    if (has_clean == has_yield)
      throw std::invalid_argument("bonds: each bond needs EXACTLY ONE of 'clean' or 'yield'");
    const double y = has_clean ? px::bond_yield_from_clean(bb.yield, jd(bo, "clean", 0.0)) : jd(bo, "yield", 0.0);

    const double d = px::bond_dirty_from_yield(bb.yield, y);
    const px::BondRisk risk = px::bond_risk(bb.yield, y);
    clean.push_back(d - bb.accrued);
    dirty.push_back(d);
    accrued.push_back(bb.accrued);
    ytm.push_back(y);
    mdur.push_back(risk.modified_duration);
    macdur.push_back(risk.macaulay_duration);
    convx.push_back(risk.convexity);
  }

  json::object out;
  out["clean"] = vecf(clean);
  out["dirty"] = vecf(dirty);
  out["accrued"] = vecf(accrued);
  out["ytm"] = vecf(ytm);
  out["modified_duration"] = vecf(mdur);
  out["macaulay_duration"] = vecf(macdur);
  out["convexity"] = vecf(convx);
  out["n"] = static_cast<int>(clean.size());
  return json::serialize(json::value(std::move(out)));
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
std::string asset_swap_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
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
  RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  for (int c = 0; c < static_cast<int>(sess.problem().curves.size()); ++c) reg.curves.push_back(c);
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

}  // namespace swaps::api
