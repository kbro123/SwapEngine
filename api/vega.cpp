// Vega-ladder seam: a run_json "vega" verb computing a swaption book's aggregate sensitivity to a vol surface's
// PARAMETERS (per-cell normal vol, or per-cell SABR {alpha,rho,nu}) off a CALIBRATED SOFR curve — the vol
// analogue of the rates delta ladder (api/generate_risk + risk_operator on the rates side). Request:
//   {"vega": {value_date, bundle, currency?, index?, curve?,
//             cells:[{expiry, tenor, normal_vol? | sabr?{alpha,rho,nu}}],
//             swaptions:[{expiry, tenor, strike? | moneyness_bp?, payer?, notional?}]}}
// It calibrates the bundle (same recipe as swaption_json), resolves each CELL's underlying forward-starting
// swap schedule (annual fixed), samples the curve's discount factors ONCE over the union of all schedule times
// to form each cell's forward swap rate + annuity, maps each SWAPTION to its (expiry, tenor) cell, then buckets
// d(book value)/d(vol parameter) per cell via vol/vega_ladder.hpp. Reuses BundleSession + the build/ date layer
// + the vol/ analytics; reimplements no pricing. QuantLib-free.
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/vega.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/conventions_data.hpp"
#include "swaps/vol/swaption.hpp"
#include "swaps/vol/vega_ladder.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace v = swaps::vol;

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
std::string js(const json::object& o, const char* k, const char* d = "") {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return d;
  return std::string(o.at(k).as_string().c_str());
}
bool jb(const json::object& o, const char* k, bool d) {
  return o.contains(k) && o.at(k).is_bool() ? o.at(k).as_bool() : d;
}
json::array vecf(const std::vector<double>& x) {
  json::array a;
  a.reserve(x.size());
  for (double e : x) a.push_back(e);
  return a;
}

// One cell's CURVE-INDEPENDENT schedule in curve time (ACT/365F): expiry, swap start, annual fixed pay times +
// accruals — the same walk options.cpp uses for the vol cube.
struct CellSchedule {
  double t_start = 0.0, t_expiry = 0.0;
  std::vector<double> pay_time, tau;
};
CellSchedule cell_schedule(const std::string& expiry, const std::string& tenor, const b::Date& vd,
                           const b::SwapConv& conv) {
  if (expiry.empty() || tenor.empty())
    throw std::invalid_argument("vega: each cell/swaption needs a non-empty 'expiry' and 'tenor'");
  const b::Date expiry_date = b::resolve(expiry, vd, conv.calendar, conv.bdc, 0);
  const b::Date swap_start = b::spot_date(expiry_date, conv.calendar, conv.spot_lag);
  const double tenor_years = conventions::period_years(tenor);
  if (!(tenor_years > 0.0))
    throw std::invalid_argument("vega: unrecognized or non-positive tenor '" + tenor + "' (use e.g. 2Y, 5Y, 10Y)");
  const int months = std::max(1, static_cast<int>(std::lround(tenor_years * 12.0)));
  CellSchedule s;
  s.t_start = b::curve_time(vd, swap_start);
  s.t_expiry = b::curve_time(vd, expiry_date);
  // The underlying's FIXED leg on the product's own schedule (frequency / bdc / pay lag / day count) — it used
  // to be hard-wired annual with no pay lag, wrong for every non-annual-fixed market (SAR/AUD/CNY/ZAR/CAD...).
  const b::Date und_mat = swap_start.plus_months(months);
  for (const auto& [ps, pe] : b::swap_periods_between(swap_start, conv.calendar, und_mat, conv.fixed_freq_tok, conv.bdc)) {
    s.tau.push_back(b::year_frac(conv.fixed_dc, ps, pe));
    s.pay_time.push_back(b::curve_time(vd, b::advance_bd(conv.calendar, pe, conv.pay_lag)));
  }
  return s;
}

// The swap's DB index is REQUIRED (it carries the product conventions); `currency` is optional and, if given,
// must agree with the index row (PRINCIPLES.md P2: no "USD"/"USD-SOFR" defaults).
std::string require_index_arg(const json::object& o, const char* verb) {
  const std::string index = js(o, "index");
  if (index.empty()) throw std::invalid_argument(std::string(verb) + ": missing 'index' (the swap's DB index id, e.g. USD-SOFR)");
  return index;
}
std::string currency_for_index(const std::string& index, const std::string& given, const char* verb) {
  const std::string ccy = std::string(swaps::conventions::require_index(index).currency);
  if (!given.empty() && b::upper(given) != ccy)
    throw std::invalid_argument(std::string(verb) + ": 'currency' " + given + " does not match index " + index + " (" + ccy + ")");
  return ccy;
}
}  // namespace

std::string vega_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("vega") ? top.at("vega").as_object() : top;

  const std::string vd_iso = js(o, "value_date");
  if (vd_iso.empty()) throw std::invalid_argument("vega: missing 'value_date'");
  if (!o.contains("bundle")) throw std::invalid_argument("vega: missing 'bundle'");
  const b::Date vd = b::Date::from_iso(vd_iso);
  const std::string index = require_index_arg(o, "vega");
  const std::string currency = currency_for_index(index, js(o, "currency"), "vega");
  const int curve = static_cast<int>(jd(o, "curve", 0.0));
  const b::SwapConv conv = b::swap_conv(currency, index);

  // Calibrate the bundle -> the curve we price/vega off (same recipe as swaption_json).
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  for (int c = 0; c < static_cast<int>(sess.problem().curves.size()); ++c) reg.curves.push_back(c);
  sess.calibrate(flat_x0(sess.problem()), reg);
  if (curve < 0 || curve >= static_cast<int>(sess.problem().curves.size()))
    throw std::invalid_argument("vega: curve index out of range");

  // ---- (1) resolve every cell's schedule + vol model; collect the union of all schedule times. ----------
  if (!o.contains("cells") || !o.at("cells").is_array())
    throw std::invalid_argument("vega: missing 'cells' array");
  std::vector<v::VegaCell> cells;
  std::vector<CellSchedule> scheds;
  std::map<std::string, int> cell_index;  // "expiry|tenor" -> cell index (for swaption -> cell mapping)
  std::vector<double> all_times;
  for (const auto& ce : o.at("cells").as_array()) {
    const json::object& c = ce.as_object();
    const std::string expiry = js(c, "expiry"), tenor = js(c, "tenor");
    const CellSchedule s = cell_schedule(expiry, tenor, vd, conv);
    v::VegaCell cell;
    cell.expiry_years = s.t_expiry;
    if (c.contains("sabr") && c.at("sabr").is_object()) {
      const auto& sj = c.at("sabr").as_object();
      cell.has_sabr = true;
      cell.sabr_alpha = jd(sj, "alpha", 0.0);
      cell.sabr_rho = jd(sj, "rho", 0.0);
      cell.sabr_nu = jd(sj, "nu", 0.0);
    } else {
      cell.normal_vol = jd(c, "normal_vol", 0.0);
    }
    cell_index[expiry + "|" + tenor] = static_cast<int>(cells.size());
    all_times.push_back(s.t_start);
    all_times.insert(all_times.end(), s.pay_time.begin(), s.pay_time.end());
    cells.push_back(cell);
    scheds.push_back(s);
  }

  // ---- (2) sample the calibrated curve ONCE -> each cell's forward swap rate + annuity. ------------------
  std::sort(all_times.begin(), all_times.end());
  all_times.erase(std::unique(all_times.begin(), all_times.end()), all_times.end());
  const std::vector<CurveSample> cs = sess.sample(all_times);
  const std::vector<double>& disc = cs.at(static_cast<std::size_t>(curve)).discount;
  const auto df_at = [&](double t) -> double {
    const auto it = std::lower_bound(all_times.begin(), all_times.end(), t);
    if (it == all_times.end() || *it != t) throw std::runtime_error("vega: internal sample-time lookup failed");
    return disc[static_cast<std::size_t>(it - all_times.begin())];
  };
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const CellSchedule& s = scheds[ci];
    std::vector<double> df_pay;
    df_pay.reserve(s.pay_time.size());
    for (double t : s.pay_time) df_pay.push_back(df_at(t));
    const v::ForwardSwap fs = v::forward_swap(df_at(s.t_start), df_pay.back(), df_pay, s.tau);
    cells[ci].forward = fs.rate;
    cells[ci].annuity = fs.annuity;
  }

  // ---- (3) map each swaption to its cell; resolve its strike (absolute / moneyness / ATM). ---------------
  std::vector<v::VegaSwaption> book;
  if (o.contains("swaptions") && o.at("swaptions").is_array()) {
    for (const auto& se : o.at("swaptions").as_array()) {
      const json::object& t = se.as_object();
      const std::string expiry = js(t, "expiry"), tenor = js(t, "tenor");
      const auto it = cell_index.find(expiry + "|" + tenor);
      if (it == cell_index.end())
        throw std::invalid_argument("vega: swaption (" + expiry + ", " + tenor +
                                    ") has no matching cell in 'cells'");
      v::VegaSwaption sw;
      sw.cell = it->second;
      const double F = cells[static_cast<std::size_t>(sw.cell)].forward;
      if (t.contains("strike") && !t.at("strike").is_null())
        sw.strike = jd(t, "strike", 0.0);
      else if (t.contains("moneyness_bp") && !t.at("moneyness_bp").is_null())
        sw.strike = F + jd(t, "moneyness_bp", 0.0) / 1e4;
      else
        sw.strike = F;  // ATM default
      sw.payer = jb(t, "payer", true);
      sw.notional = jd(t, "notional", 1.0);
      book.push_back(sw);
    }
  }

  // ---- (4) the ladder + response ------------------------------------------------------------------------
  const v::VegaLadder L = v::vega_ladder(cells, book);

  std::vector<double> cell_forward, cell_annuity, cell_expiry, cell_has_sabr;
  for (const v::VegaCell& c : cells) {
    cell_forward.push_back(c.forward);
    cell_annuity.push_back(c.annuity);
    cell_expiry.push_back(c.expiry_years);
    cell_has_sabr.push_back(c.has_sabr ? 1.0 : 0.0);
  }
  json::object out;
  out["n_cells"] = L.n_cells;
  out["n_swaptions"] = static_cast<int>(book.size());
  out["book_value"] = L.book_value;
  out["cell_forward"] = vecf(cell_forward);
  out["cell_annuity"] = vecf(cell_annuity);
  out["cell_expiry_years"] = vecf(cell_expiry);
  out["cell_has_sabr"] = vecf(cell_has_sabr);
  out["d_normal_vol"] = vecf(L.d_normal_vol);
  out["d_alpha"] = vecf(L.d_alpha);
  out["d_rho"] = vecf(L.d_rho);
  out["d_nu"] = vecf(L.d_nu);
  json::object resp;
  resp["vega"] = std::move(out);
  return json::serialize(resp);
}

}  // namespace swaps::api
