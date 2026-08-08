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
#include "swaps/build/bond.hpp"
#include "swaps/build/date.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace px = swaps::pricing;

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
    if (settle_iso.empty() || maturity_iso.empty() || issue_iso.empty())
      throw std::invalid_argument("bonds: each bond needs 'issue', 'settle' and 'maturity' (YYYY-MM-DD)");
    if (!bo.contains("coupon")) throw std::invalid_argument("bonds: each bond needs 'coupon'");

    b::FixedBondTerms t;
    t.value_date = b::Date::from_iso(vd_top.empty() ? settle_iso : vd_top);  // unused in street space
    t.settle = b::Date::from_iso(settle_iso);
    t.issue = b::Date::from_iso(issue_iso);
    t.maturity = b::Date::from_iso(maturity_iso);
    t.coupon = jd(bo, "coupon", 0.0);
    t.freq = static_cast<int>(jd(bo, "freq", 2.0));
    const b::BuiltBond bb = b::fixed_rate_bond(t);

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

}  // namespace swaps::api
