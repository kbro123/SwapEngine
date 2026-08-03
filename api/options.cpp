// Options seam: a run_json "swaption" verb that prices European swaptions off a CALIBRATED SOFR curve.
// Request: {"swaption": {value_date, bundle, currency?, index?, curve?, trades:[{expiry, tenor, strike,
// payer?, normal_vol? , sabr?{alpha,rho,nu}}]}}. It calibrates the bundle, and for each trade builds the
// underlying forward-starting swap schedule (annual fixed), samples the curve's discount factors, forms the
// forward swap rate + annuity, and prices with the Bachelier normal model (SABR vol at the strike if given).
// Reuses BundleSession + the build/ date-schedule layer + the vol/ analytics. QuantLib-free.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/conventions_data.hpp"
#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/swaption.hpp"

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
}  // namespace

std::string swaption_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("swaption") ? top.at("swaption").as_object() : top;

  const std::string vd_iso = js(o, "value_date");
  if (vd_iso.empty()) throw std::invalid_argument("swaption: missing 'value_date'");
  if (!o.contains("bundle")) throw std::invalid_argument("swaption: missing 'bundle'");
  const b::Date vd = b::Date::from_iso(vd_iso);
  const std::string currency = js(o, "currency", "USD");
  const std::string index = js(o, "index", "USD-SOFR");
  const int curve = static_cast<int>(jd(o, "curve", 0.0));
  const b::SwapConv conv = b::swap_conv(currency, 1.0, index);

  // Calibrate the bundle to its markets -> the curve we price off.
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  for (int c = 0; c < static_cast<int>(sess.problem().curves.size()); ++c) reg.curves.push_back(c);
  sess.calibrate(flat_x0(sess.problem()), reg);

  json::array out_trades;
  if (o.contains("trades") && o.at("trades").is_array()) {
    for (const auto& te : o.at("trades").as_array()) {
      const json::object& t = te.as_object();
      const std::string expiry = js(t, "expiry"), tenor = js(t, "tenor");
      const double strike = jd(t, "strike", 0.0);
      const bool payer = jb(t, "payer", true);

      // Underlying forward-starting swap: option expires at `expiry`, swap starts spot after expiry, annual
      // fixed leg to `tenor`.
      if (expiry.empty() || tenor.empty())
        throw std::invalid_argument("swaption: each trade needs a non-empty 'expiry' and 'tenor'");
      const b::Date expiry_date = b::resolve(expiry, vd);  // throws on an unrecognized expiry token
      const b::Date swap_start = b::spot_date(expiry_date, conv.calendar, conv.spot_lag);
      const double tenor_years = conventions::period_years(tenor);  // 0 for an unrecognized token
      if (!(tenor_years > 0.0))
        throw std::invalid_argument("swaption: unrecognized or non-positive tenor '" + tenor +
                                    "' (use e.g. 2Y, 5Y, 10Y)");
      const int n = std::max(1, static_cast<int>(std::lround(tenor_years)));
      std::vector<double> pay_time, tau;
      b::Date prev = swap_start;
      for (int i = 1; i <= n; ++i) {
        const b::Date pay = b::adjust(conv.calendar, swap_start.plus_months(12 * i), conv.bdc);
        tau.push_back(b::year_frac(conv.fixed_dc, prev, pay));
        pay_time.push_back(b::curve_time(vd, pay));
        prev = pay;
      }
      const double t_start = b::curve_time(vd, swap_start);
      const double t_expiry = b::curve_time(vd, expiry_date);

      // Sample the calibrated curve's discount factors at the swap start + every pay date.
      std::vector<double> times{t_start};
      times.insert(times.end(), pay_time.begin(), pay_time.end());
      const std::vector<CurveSample> cs = sess.sample(times);
      const std::vector<double>& df = cs.at(static_cast<std::size_t>(curve)).discount;
      const double df_start = df.front();
      std::vector<double> df_pay(df.begin() + 1, df.end());
      const double df_end = df_pay.back();

      const v::ForwardSwap fs = v::forward_swap(df_start, df_end, df_pay, tau);
      const v::Payoff cp = payer ? v::Payoff::Payer : v::Payoff::Receiver;

      double vol, atm_vol;
      if (t.contains("sabr") && t.at("sabr").is_object()) {
        const auto& s = t.at("sabr").as_object();
        const v::SabrParams sp{jd(s, "alpha", 0.0), jd(s, "rho", 0.0), jd(s, "nu", 0.0)};
        vol = v::sabr_normal_vol(fs.rate, strike, t_expiry, sp);
        atm_vol = v::sabr_normal_vol(fs.rate, fs.rate, t_expiry, sp);
      } else {
        vol = atm_vol = jd(t, "normal_vol", 0.0);
      }
      const double price = v::swaption_price(fs.rate, fs.annuity, strike, vol, t_expiry, cp);

      json::object r;
      r["expiry"] = expiry;
      r["tenor"] = tenor;
      r["strike"] = strike;
      r["payer"] = payer;
      r["expiry_years"] = t_expiry;
      r["forward"] = fs.rate;
      r["annuity"] = fs.annuity;
      r["normal_vol"] = vol;
      r["atm_normal_vol"] = atm_vol;
      r["price"] = price;
      r["vega"] = v::bachelier_vega<double>(fs.rate, strike, vol, t_expiry, fs.annuity);
      r["delta"] = v::bachelier_delta<double>(fs.rate, strike, vol, t_expiry, fs.annuity, cp);
      out_trades.push_back(std::move(r));
    }
  }
  json::object out;
  out["trades"] = std::move(out_trades);
  return json::serialize(json::value(std::move(out)));
}

// Batched vol-cube reprice off the CURRENTLY CALIBRATED curve — the options hot path (see VolCube in
// bundle_api.hpp). Curve-dependent work (each cell's forward/annuity) is done ONCE from a single sample()
// over the union of all schedule times; every strike is then a pure Bachelier/SABR eval. Reuses the
// calibrated/streaming session, so a live vol surface reprices with no recalibration.
VolCube BundleSession::price_vol_cube_json(const std::string& spec_json) const {
  const json::value req = json::parse(spec_json);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("vol_cube") ? top.at("vol_cube").as_object() : top;

  const std::string vd_iso = js(o, "value_date");
  if (vd_iso.empty()) throw std::invalid_argument("vol_cube: missing 'value_date'");
  const b::Date vd = b::Date::from_iso(vd_iso);
  const std::string currency = js(o, "currency", "USD");
  const std::string index = js(o, "index", "USD-SOFR");
  const int curve = static_cast<int>(jd(o, "curve", 0.0));
  const b::SwapConv conv = b::swap_conv(currency, 1.0, index);

  if (!o.contains("cells") || !o.at("cells").is_array())
    throw std::invalid_argument("vol_cube: missing 'cells' array");
  const json::array& cells = o.at("cells").as_array();

  const auto clock0 = std::chrono::steady_clock::now();

  // ---- (1) resolve each cell's schedule, MEMOIZED per cell (curve-independent — the calendar walk is the
  // dominant per-call cost, and it is identical for every reprice). Key by the fields that determine it.
  std::vector<const SwaptionSchedule*> sched(cells.size(), nullptr);
  std::vector<std::string> keys(cells.size());
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const json::object& c = cells[ci].as_object();
    const std::string expiry = js(c, "expiry"), tenor = js(c, "tenor");
    if (expiry.empty() || tenor.empty())
      throw std::invalid_argument("vol_cube: each cell needs a non-empty 'expiry' and 'tenor'");
    std::string key = vd_iso;
    key += '|'; key += currency; key += '|'; key += index; key += '|';
    key += std::to_string(curve); key += '|'; key += expiry; key += '|'; key += tenor;
    auto it = vol_sched_cache_.find(key);
    if (it == vol_sched_cache_.end()) {
      const b::Date expiry_date = b::resolve(expiry, vd);
      const b::Date swap_start = b::spot_date(expiry_date, conv.calendar, conv.spot_lag);
      const double tenor_years = conventions::period_years(tenor);
      if (!(tenor_years > 0.0))
        throw std::invalid_argument("vol_cube: unrecognized or non-positive tenor '" + tenor +
                                    "' (use e.g. 2Y, 5Y, 10Y)");
      const int n = std::max(1, static_cast<int>(std::lround(tenor_years)));
      SwaptionSchedule ss;
      ss.t_start = b::curve_time(vd, swap_start);
      ss.t_expiry = b::curve_time(vd, expiry_date);
      b::Date prev = swap_start;
      for (int i = 1; i <= n; ++i) {
        const b::Date pay = b::adjust(conv.calendar, swap_start.plus_months(12 * i), conv.bdc);
        ss.tau.push_back(b::year_frac(conv.fixed_dc, prev, pay));
        ss.pay_time.push_back(b::curve_time(vd, pay));
        prev = pay;
      }
      it = vol_sched_cache_.emplace(key, std::move(ss)).first;
    }
    sched[ci] = &it->second;
    keys[ci] = std::move(key);
  }

  // ---- (2) forward/annuity per cell, cached against the curve state x. If x is unchanged since the cache was
  // built (a vol-only reprice — e.g. a SABR-slider tick), reuse it and DO NOT sample the curve at all. When x
  // moved (or a cell is new), sample once over just the missing cells' times.
  const Eigen::VectorXd& xnow = this->x();
  const bool fa_valid = vol_fa_x_.size() == xnow.size() && xnow.size() > 0 &&
                        (vol_fa_x_.array() == xnow.array()).all();
  if (!fa_valid) {
    vol_fa_cache_.clear();
    vol_fa_x_ = xnow;
  }
  std::vector<double> need_times;
  for (std::size_t ci = 0; ci < cells.size(); ++ci)
    if (vol_fa_cache_.find(keys[ci]) == vol_fa_cache_.end()) {
      need_times.push_back(sched[ci]->t_start);
      need_times.insert(need_times.end(), sched[ci]->pay_time.begin(), sched[ci]->pay_time.end());
    }
  if (!need_times.empty()) {
    std::sort(need_times.begin(), need_times.end());
    need_times.erase(std::unique(need_times.begin(), need_times.end()), need_times.end());
    const std::vector<CurveSample> cs_all = this->sample(need_times);
    if (curve < 0 || curve >= static_cast<int>(cs_all.size()))
      throw std::invalid_argument("vol_cube: curve index out of range");
    const CurveSample& csamp = cs_all[static_cast<std::size_t>(curve)];
    std::map<double, double> df_by_time;
    for (std::size_t i = 0; i < csamp.t.size(); ++i) df_by_time[csamp.t[i]] = csamp.discount[i];
    const auto df_at = [&](double t) -> double {
      const auto itf = df_by_time.find(t);
      if (itf == df_by_time.end()) throw std::runtime_error("vol_cube: internal sample-time lookup failed");
      return itf->second;
    };
    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
      if (vol_fa_cache_.find(keys[ci]) != vol_fa_cache_.end()) continue;
      const SwaptionSchedule& s = *sched[ci];
      std::vector<double> df_pay;
      df_pay.reserve(s.pay_time.size());
      for (double t : s.pay_time) df_pay.push_back(df_at(t));
      const v::ForwardSwap fs = v::forward_swap(df_at(s.t_start), df_pay.back(), df_pay, s.tau);
      vol_fa_cache_.emplace(keys[ci], std::make_pair(fs.rate, fs.annuity));
    }
  }

  // ---- (3) price every cell x strike into the flat SoA result (pure Bachelier/SABR off the cached fwd/annuity).
  VolCube out;
  out.n_cells = static_cast<int>(cells.size());
  out.cell_forward.reserve(cells.size());
  out.cell_annuity.reserve(cells.size());
  out.cell_expiry_years.reserve(cells.size());
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const json::object& c = cells[ci].as_object();
    const SwaptionSchedule& s = *sched[ci];
    const std::pair<double, double>& fa = vol_fa_cache_.at(keys[ci]);
    const v::ForwardSwap fs{fa.first, fa.second};
    out.cell_forward.push_back(fs.rate);
    out.cell_annuity.push_back(fs.annuity);
    out.cell_expiry_years.push_back(s.t_expiry);

    const bool has_sabr = c.contains("sabr") && c.at("sabr").is_object();
    v::SabrParams sp;
    double flat_vol = 0.0;
    if (has_sabr) {
      const auto& sj = c.at("sabr").as_object();
      sp = v::SabrParams{jd(sj, "alpha", 0.0), jd(sj, "rho", 0.0), jd(sj, "nu", 0.0)};
    } else {
      flat_vol = jd(c, "normal_vol", 0.0);
    }
    const bool payer_set = c.contains("payer") && c.at("payer").is_bool();
    const bool payer_val = payer_set ? c.at("payer").as_bool() : true;

    // Strike list: explicit absolute strikes, moneyness offsets (bp from the forward), and/or ATM; default ATM.
    std::vector<double> strikes;
    if (c.contains("strikes") && c.at("strikes").is_array())
      for (const auto& k : c.at("strikes").as_array()) strikes.push_back(k.to_number<double>());
    if (c.contains("moneyness_bp") && c.at("moneyness_bp").is_array())
      for (const auto& m : c.at("moneyness_bp").as_array())
        strikes.push_back(fs.rate + m.to_number<double>() / 1e4);
    if (jb(c, "atm", false) || strikes.empty()) strikes.push_back(fs.rate);

    for (double strike : strikes) {
      const bool payer = payer_set ? payer_val : (strike >= fs.rate);
      const v::Payoff cp = payer ? v::Payoff::Payer : v::Payoff::Receiver;
      const double vol = has_sabr ? v::sabr_normal_vol(fs.rate, strike, s.t_expiry, sp) : flat_vol;
      out.point_cell.push_back(static_cast<double>(ci));
      out.strike.push_back(strike);
      out.moneyness_bp.push_back((strike - fs.rate) * 1e4);
      out.normal_vol.push_back(vol);
      out.price.push_back(v::swaption_price(fs.rate, fs.annuity, strike, vol, s.t_expiry, cp));
      out.vega.push_back(v::bachelier_vega<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity));
      out.delta.push_back(v::bachelier_delta<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity, cp));
      out.gamma.push_back(v::bachelier_gamma<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity));
      out.payer.push_back(payer ? 1.0 : 0.0);
    }
  }
  out.n_points = static_cast<int>(out.price.size());
  out.price_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - clock0).count();
  return out;
}

}  // namespace swaps::api
