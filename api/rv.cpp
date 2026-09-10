// RV seam: the stateless bond-RV / swap-spread verbs — the production wiring of the derive/ + market/
// preview layers (the audit's "unreachable capabilities" 1-5).
//
//   * bond_universe — BATCHED street/yield-space analytics over a whole universe on the vectorized
//     portfolio::BondUniverse (Horner/FMA sweep + batched Newton), not a per-bond loop: cleans<->yields,
//     modified duration, convexity, accrued, in one call.
//   * govvie_fit    — MINIMUM-PRICING-ERROR govvie curve fit over a universe of bond clean prices
//     (spline / Nelson-Siegel / Svensson), via derive::make_govvie_fit / make_parametric_fit over a real
//     market::Market built from the request. Returns the fitted state, per-bond fair-value residuals (the
//     RV richness/cheapness ladder), and — for the spline — per-bond z-spreads off the fitted curve
//     (portfolio::CompiledBondBook, the W-cache bond repricer).
//   * swap_spread   — the HEADLINE swap-spread derivation (derive::derive_asset_swap): benchmark bond
//     clean price -> street yield (WI-aware via build::BondId), then the {pin, asw} calibration rows of
//     the asset-swap BASIS (build/swap_spread.hpp) serialized as instrument JSON, ready to compose into a
//     compile/bundle spec. This is the basis-row mode the shipped par-par `asset_swap` verb lacked.
//
// Bonds are identified by build::BondId rows ({id, issue, maturity, coupon, first_coupon?}) with a shared
// conventions-DB `convention` id; quotes are decimals (prices per unit notional, rates absolute).
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/rv.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/parametric.hpp"
#include "swaps/derive/asset_swap.hpp"
#include "swaps/market/market.hpp"
#include "swaps/market/quote.hpp"
#include "swaps/portfolio/bond_universe.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace cvd = swaps::conventions;
namespace der = swaps::derive;
namespace mkt = swaps::market;
namespace pf = swaps::portfolio;
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
json::array vecf(const Eigen::VectorXd& v) {
  json::array a;
  a.reserve(v.size());
  for (int i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}
std::vector<double> darr(const json::object& o, const char* k) {
  std::vector<double> out;
  if (o.contains(k) && o.at(k).is_array())
    for (const auto& e : o.at(k).as_array()) out.push_back(e.to_number<double>());
  return out;
}
const json::object& sub(const json::object& top, const char* key) {
  return top.contains(key) && top.at(key).is_object() ? top.at(key).as_object() : top;
}

// One BondId row: {id, issue, maturity, coupon, first_coupon?}; `convention` is shared across the universe.
b::BondId bond_id_from(const json::object& o, const std::string& convention) {
  b::BondId bi;
  bi.id = js(o, "id");
  bi.yield_conv = convention;
  bi.issue = b::Date::from_iso(js(o, "issue"));
  bi.maturity = b::Date::from_iso(js(o, "maturity"));
  bi.coupon = jd(o, "coupon", 0.0);
  const std::string fc = js(o, "first_coupon");
  if (!fc.empty()) bi.first_coupon = b::Date::from_iso(fc);  // when-issued / odd first period
  return bi;
}
std::vector<b::BondId> universe_from(const json::object& o, const std::string& convention) {
  if (!o.contains("bonds") || !o.at("bonds").is_array())
    throw std::invalid_argument("missing 'bonds' array");
  std::vector<b::BondId> u;
  for (const auto& e : o.at("bonds").as_array()) u.push_back(bond_id_from(e.as_object(), convention));
  return u;
}
}  // namespace

// ---- bond_universe: batched yield-space sweep over the whole universe ---------------------------------
std::string bond_universe_json(const json::object& request) {
  const json::object& o = sub(request, "bond_universe");
  const std::string convention = js(o, "convention");
  if (convention.empty()) throw std::invalid_argument("bond_universe: missing 'convention' (a bonds[] row id, e.g. US-TREASURY)");
  const cvd::BondConv bc = cvd::require_bond(convention);
  const b::Date vd = b::Date::from_iso(js(o, "value_date"));
  const std::string settle_iso = js(o, "settle");
  const b::Date settle = settle_iso.empty() ? b::advance_bd(std::string(bc.calendar), vd, bc.settle_lag)
                                            : b::Date::from_iso(settle_iso);

  const std::vector<b::BondId> ids = universe_from(o, convention);
  std::vector<px::YieldBond> yb;
  std::vector<double> accrued;
  for (const auto& bi : ids) {
    b::BuiltBond built = b::build_bond(bi, vd, settle);
    accrued.push_back(built.accrued);
    yb.push_back(std::move(built.yield));
  }
  pf::BondUniverse uni;
  uni.set(yb);

  const std::vector<double> clean = darr(o, "clean");
  const std::vector<double> yield = darr(o, "yield");
  if (clean.empty() == yield.empty())
    throw std::invalid_argument("bond_universe: provide exactly one of 'clean' or 'yield' (per bond)");
  const int n = static_cast<int>(ids.size());
  if (static_cast<int>(clean.empty() ? yield.size() : clean.size()) != n)
    throw std::invalid_argument("bond_universe: quote array length must match the bond count");

  Eigen::VectorXd y(n);
  json::object out;
  if (!clean.empty()) {
    y = uni.yields_from_clean(Eigen::Map<const Eigen::VectorXd>(clean.data(), n));
    out["clean"] = vecf(clean);
  } else {
    y = Eigen::Map<const Eigen::VectorXd>(yield.data(), n);
    out["clean"] = vecf(uni.clean_prices(y));
  }
  out["yield"] = vecf(y);
  out["modified_duration"] = vecf(uni.modified_durations(y));
  out["convexity"] = vecf(uni.convexities(y));
  out["accrued"] = vecf(accrued);
  out["n"] = n;
  json::object resp;
  resp["bond_universe"] = out;
  return json::serialize(resp);
}

// ---- govvie_fit: minimum-pricing-error curve over the universe (the RV fair-value fit) ----------------
std::string govvie_fit_json(const json::object& request) {
  const json::object& o = sub(request, "govvie_fit");
  const std::string convention = js(o, "convention");
  if (convention.empty()) throw std::invalid_argument("govvie_fit: missing 'convention' (a bonds[] row id)");
  const cvd::BondConv bc = cvd::require_bond(convention);
  const b::Date vd = b::Date::from_iso(js(o, "value_date"));

  der::AssetSwapConvention conv;  // settlement rule for the universe build: the bond convention's (DB), overridable
  conv.currency = std::string(bc.currency);
  conv.settle_calendar = js(o, "settle_calendar").empty() ? std::string(bc.calendar) : js(o, "settle_calendar");
  conv.settle_lag = o.contains("settle_lag") ? static_cast<int>(jd(o, "settle_lag", 0.0)) : bc.settle_lag;

  const std::vector<b::BondId> ids = universe_from(o, convention);
  const std::vector<double> clean = darr(o, "clean");
  if (static_cast<int>(clean.size()) != static_cast<int>(ids.size()))
    throw std::invalid_argument("govvie_fit: 'clean' length must match the bond count");

  // The market snapshot the derive layer consumes: as-of + one CLEAN-price quote per BondId.
  mkt::Market m;
  m.as_of(vd);
  for (std::size_t i = 0; i < ids.size(); ++i) m.add_quote(ids[i].id, mkt::Quote::mid(clean[i]));

  const std::vector<double> weight = darr(o, "weight");
  const std::string model = js(o, "model", "spline");
  json::object out;
  if (model == "spline") {
    const std::vector<double> meeting = darr(o, "meeting"), back = darr(o, "back");
    if (back.empty()) throw std::invalid_argument("govvie_fit: spline model needs 'back' knot times");
    cal::GovvieBondFit fit = der::make_govvie_fit(conv, ids, m, meeting, back, weight);
    const cal::CalibrationResult res =
        cal::calibrate(fit, Eigen::VectorXd::Constant(fit.n_knots(), jd(o, "x0", 0.03)));
    out["x"] = vecf(res.x);
    out["rms_residual"] = res.rms_residual;
    out["iterations"] = res.iterations;
    out["residuals"] = vecf(fit.residuals<double>(res.x));  // model − market per bond (fair-value ladder)
    // Per-bond z-spreads off the FITTED curve — the RV richness/cheapness number (spline topology only:
    // CompiledBondBook shares the same (meeting, back) layout as the fit).
    const b::Date settle = b::advance_bd(conv.settle_calendar, vd, conv.settle_lag);
    std::vector<px::Bond> bonds;
    Eigen::VectorXd target(static_cast<int>(ids.size()));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      b::BuiltBond built = b::build_bond(ids[i], vd, settle);
      target[static_cast<int>(i)] = clean[i] + built.accrued;  // dirty target
      bonds.push_back(std::move(built.curve));
    }
    const pf::CompiledBondBook book(meeting, back, bonds);
    out["z_spread"] = vecf(book.z_spreads(res.x, target));
  } else if (model == "nelson_siegel" || model == "svensson") {
    const double tau1 = jd(o, "tau1", 2.0), tau2 = jd(o, "tau2", 5.0);
    Eigen::VectorXd x, resid;
    cal::CalibrationResult res;
    if (model == "nelson_siegel") {
      auto fit = der::make_parametric_fit<cv::NelsonSiegel>(conv, ids, m, tau1, tau2, weight);
      Eigen::VectorXd seed(fit.n_knots());
      seed << jd(o, "x0", 0.03), 0.0, 0.0;
      res = cal::calibrate(fit, seed);
      resid = fit.residuals<double>(res.x);
    } else {
      auto fit = der::make_parametric_fit<cv::Svensson>(conv, ids, m, tau1, tau2, weight);
      Eigen::VectorXd seed(fit.n_knots());
      seed << jd(o, "x0", 0.03), 0.0, 0.0, 0.0;
      res = cal::calibrate(fit, seed);
      resid = fit.residuals<double>(res.x);
    }
    out["x"] = vecf(res.x);
    out["rms_residual"] = res.rms_residual;
    out["iterations"] = res.iterations;
    out["residuals"] = vecf(resid);  // model − market per bond: the parametric fair-value RV ladder
  } else {
    throw std::invalid_argument("govvie_fit: unknown model '" + model +
                                "' (spline | nelson_siegel | svensson)");
  }
  out["model"] = model;
  out["n"] = static_cast<int>(ids.size());
  json::object resp;
  resp["govvie_fit"] = out;
  return json::serialize(resp);
}

// ---- swap_spread: the headline swap-spread derivation -> the {pin, asw} BASIS rows --------------------
std::string swap_spread_json(const json::object& request) {
  const json::object& o = sub(request, "swap_spread");
  const b::Date vd = b::Date::from_iso(js(o, "value_date"));

  const std::string convention = js(o, "convention");
  if (convention.empty()) throw std::invalid_argument("swap_spread: missing 'convention' (the benchmark's bonds[] row id)");
  const cvd::BondConv bc = cvd::require_bond(convention);
  der::AssetSwapConvention conv;
  conv.currency = std::string(bc.currency);
  conv.settle_calendar = js(o, "settle_calendar").empty() ? std::string(bc.calendar) : js(o, "settle_calendar");
  conv.settle_lag = o.contains("settle_lag") ? static_cast<int>(jd(o, "settle_lag", 0.0)) : bc.settle_lag;
  conv.type = js(o, "spread_type", "headline") == "matched_maturity"
                  ? der::SwapSpreadType::MatchedMaturity
                  : der::SwapSpreadType::HeadlineYield;

  if (!o.contains("bond") || !o.at("bond").is_object())
    throw std::invalid_argument("swap_spread: missing 'bond' object");
  const b::BondId bond =
      bond_id_from(o.at("bond").as_object(), convention);
  const double clean = jd(o, "clean", 0.0);
  if (clean <= 0.0) throw std::invalid_argument("swap_spread: missing benchmark 'clean' price");
  const double spread = jd(o, "spread", 0.0);

  // The matched spot-start par swap, built GROUND-UP from the index's conventions-DB conventions:
  // {"index": "USD-SOFR", "tenor": "5Y", "swap_curve": <bundle role>}. The tenor resolves under the
  // index's calendar from the value date (the spread's TENOR, not the bond's maturity — the headline
  // convention's defining quirk).
  const std::string index_id = js(o, "index");
  if (index_id.empty()) throw std::invalid_argument("swap_spread: missing swap 'index'");
  const int swap_curve = static_cast<int>(jd(o, "swap_curve", 0.0));
  const int factor_curve = static_cast<int>(jd(o, "factor_curve", 1.0));
  const std::string tenor = js(o, "tenor");
  if (tenor.empty()) throw std::invalid_argument("swap_spread: missing swap 'tenor' (e.g. 5Y)");
  const b::SwapConv sconv = b::Index(index_id).par_convention().resolve();
  const b::Date mat = b::resolve(tenor, vd, sconv.calendar, sconv.bdc, sconv.spot_lag);  // tenor from SPOT on the index calendar
  const cal::Instrument spot_swap = b::par_swap(vd, sconv, mat, swap_curve, swap_curve, 0.0);
  const double anchor = jd(o, "anchor", b::curve_time(vd, mat));

  // The market snapshot: the benchmark's clean price + the quoted spread, keyed for the derive layer.
  mkt::Market m;
  m.as_of(vd);
  m.add_quote(bond.id, mkt::Quote::mid(clean));
  m.add_quote("spread", mkt::Quote::mid(spread));

  const der::DerivedAssetSwap d =
      der::derive_asset_swap(conv, bond, spot_swap, factor_curve, anchor, m, "spread");

  json::object out;
  out["bond_yield"] = d.bond_yield;
  out["spread"] = d.spread;
  out["anchor"] = anchor;
  json::object rows;
  rows["pin"] = instrument_to_json(d.rows.pin);  // QuoteKind::Rate on the govvie factor (bond bucket)
  rows["asw"] = instrument_to_json(d.rows.asw);  // Portfolio{+swap, -Rate(factor)} (the ASW basis row)
  out["rows"] = rows;
  json::object resp;
  resp["swap_spread"] = out;
  return json::serialize(resp);
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string bond_universe_json(const std::string& request) { return bond_universe_json(json::parse(request).as_object()); }
std::string govvie_fit_json(const std::string& request) { return govvie_fit_json(json::parse(request).as_object()); }
std::string swap_spread_json(const std::string& request) { return swap_spread_json(json::parse(request).as_object()); }
}  // namespace swaps::api
