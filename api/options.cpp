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
#include "swaps/vol/sabr_calibration.hpp"
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

// Stateless SABR strip calibration verb (see options.hpp). No bundle: fit (alpha,rho,nu) to (strikes,
// market_vols) at (forward, expiry) via vol/sabr_calibration.hpp, and report the arbitrage-free status.
std::string sabr_calibrate_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("sabr_calibrate") ? top.at("sabr_calibrate").as_object() : top;

  const double forward = jd(o, "forward", 0.0);
  const double expiry = jd(o, "expiry", 0.0);
  if (!(expiry > 0.0)) throw std::invalid_argument("sabr_calibrate: 'expiry' must be positive");
  std::vector<double> strikes, mvols;
  if (o.contains("strikes") && o.at("strikes").is_array())
    for (const auto& k : o.at("strikes").as_array()) strikes.push_back(k.to_number<double>());
  if (o.contains("market_vols") && o.at("market_vols").is_array())
    for (const auto& v : o.at("market_vols").as_array()) mvols.push_back(v.to_number<double>());
  if (strikes.empty() || strikes.size() != mvols.size())
    throw std::invalid_argument("sabr_calibrate: 'strikes' and 'market_vols' must be non-empty, equal length");

  v::SabrParams guess;
  if (o.contains("guess") && o.at("guess").is_object()) {
    const auto& g = o.at("guess").as_object();
    guess = v::SabrParams{jd(g, "alpha", 0.0), jd(g, "rho", 0.0), jd(g, "nu", 0.0)};
  }
  const v::SabrCalibResult r = v::sabr_calibrate(forward, expiry, strikes, mvols, guess);
  const double lo = jd(o, "arb_lo", strikes.front());
  const double hi = jd(o, "arb_hi", strikes.back());

  json::object out;
  out["alpha"] = r.params.alpha;
  out["rho"] = r.params.rho;
  out["nu"] = r.params.nu;
  out["rms"] = r.rms;
  out["iterations"] = r.iterations;
  out["converged"] = r.converged;
  out["arbitrage_free"] = v::sabr_arbitrage_free(forward, expiry, r.params, lo, hi);
  return json::serialize(json::value(std::move(out)));
}

// Parse the `vol_cube` JSON document into the native VolCubeSpec (the ONLY place JSON touches the vol cube).
VolCubeSpec vol_cube_spec_from_json(const std::string& spec_json) {
  const json::value req = json::parse(spec_json);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("vol_cube") ? top.at("vol_cube").as_object() : top;

  VolCubeSpec spec;
  spec.value_date = js(o, "value_date");
  if (spec.value_date.empty()) throw std::invalid_argument("vol_cube: missing 'value_date'");
  spec.currency = js(o, "currency", "USD");
  spec.index = js(o, "index", "USD-SOFR");
  spec.curve = static_cast<int>(jd(o, "curve", 0.0));
  if (!o.contains("cells") || !o.at("cells").is_array())
    throw std::invalid_argument("vol_cube: missing 'cells' array");
  for (const auto& ce : o.at("cells").as_array()) {
    const json::object& c = ce.as_object();
    VolCubeCell cell;
    cell.expiry = js(c, "expiry");
    cell.tenor = js(c, "tenor");
    if (c.contains("sabr") && c.at("sabr").is_object()) {
      const auto& sj = c.at("sabr").as_object();
      cell.has_sabr = true;
      cell.sabr_alpha = jd(sj, "alpha", 0.0);
      cell.sabr_rho = jd(sj, "rho", 0.0);
      cell.sabr_nu = jd(sj, "nu", 0.0);
    } else {
      cell.normal_vol = jd(c, "normal_vol", 0.0);
    }
    if (c.contains("payer") && c.at("payer").is_bool()) {
      cell.payer_set = true;
      cell.payer = c.at("payer").as_bool();
    }
    if (c.contains("strikes") && c.at("strikes").is_array())
      for (const auto& k : c.at("strikes").as_array()) cell.strikes.push_back(k.to_number<double>());
    if (c.contains("moneyness_bp") && c.at("moneyness_bp").is_array())
      for (const auto& m : c.at("moneyness_bp").as_array()) cell.moneyness_bp.push_back(m.to_number<double>());
    cell.atm = jb(c, "atm", false);
    spec.cells.push_back(std::move(cell));
  }
  return spec;
}

// NATIVE batched vol-cube reprice off the CURRENTLY CALIBRATED curve — the options hot path (see VolCube in
// bundle_api.hpp), no JSON. Curve-independent schedules are memoized per cell; each cell's (forward, annuity)
// is cached against the curve state x, so a vol-only reprice (SABR-slider tick) samples the curve NOT AT ALL —
// just the Bachelier/SABR pass. Reuses the calibrated/streaming session, so a live vol surface reprices with
// no recalibration. This is what benchmarks and native clients call; the JSON verb is a thin wrapper below.
VolCube BundleSession::price_vol_cube(const VolCubeSpec& spec) const {
  const b::Date vd = b::Date::from_iso(spec.value_date);
  const b::SwapConv conv = b::swap_conv(spec.currency, 1.0, spec.index);
  const std::vector<VolCubeCell>& cells = spec.cells;

  const auto clock0 = std::chrono::steady_clock::now();

  // ---- (1) resolve each cell's schedule, MEMOIZED per cell (curve-independent — the calendar walk is the
  // dominant cost and is identical for every reprice). Key by the fields that determine it.
  std::vector<const SwaptionSchedule*> sched(cells.size(), nullptr);
  std::vector<std::string> keys(cells.size());
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const VolCubeCell& c = cells[ci];
    if (c.expiry.empty() || c.tenor.empty())
      throw std::invalid_argument("vol_cube: each cell needs a non-empty 'expiry' and 'tenor'");
    std::string key = spec.value_date;
    key += '|'; key += spec.currency; key += '|'; key += spec.index; key += '|';
    key += std::to_string(spec.curve); key += '|'; key += c.expiry; key += '|'; key += c.tenor;
    auto it = vol_sched_cache_.find(key);
    if (it == vol_sched_cache_.end()) {
      const b::Date expiry_date = b::resolve(c.expiry, vd);
      const b::Date swap_start = b::spot_date(expiry_date, conv.calendar, conv.spot_lag);
      const double tenor_years = conventions::period_years(c.tenor);
      if (!(tenor_years > 0.0))
        throw std::invalid_argument("vol_cube: unrecognized or non-positive tenor '" + c.tenor +
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

  // ---- (2) forward/annuity per cell, cached against the curve state x. If x is unchanged (a vol-only reprice),
  // reuse it and DO NOT sample the curve. When x moved (or a cell is new), sample once over the missing times.
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
    if (spec.curve < 0 || spec.curve >= static_cast<int>(cs_all.size()))
      throw std::invalid_argument("vol_cube: curve index out of range");
    const CurveSample& csamp = cs_all[static_cast<std::size_t>(spec.curve)];
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
  std::size_t npts = 0;  // reserve the per-point SoA arrays up front (no reallocation in the hot loop)
  for (const VolCubeCell& c : cells) {
    const std::size_t k = c.strikes.size() + c.moneyness_bp.size() + (c.atm ? 1 : 0);
    npts += (k == 0) ? 1 : k;
  }
  for (std::vector<double>* col : {&out.point_cell, &out.strike, &out.moneyness_bp, &out.normal_vol,
                                   &out.price, &out.vega, &out.delta, &out.gamma, &out.vanna, &out.volga,
                                   &out.payer})
    col->reserve(npts);
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const VolCubeCell& c = cells[ci];
    const SwaptionSchedule& s = *sched[ci];
    const std::pair<double, double>& fa = vol_fa_cache_.at(keys[ci]);
    const v::ForwardSwap fs{fa.first, fa.second};
    out.cell_forward.push_back(fs.rate);
    out.cell_annuity.push_back(fs.annuity);
    out.cell_expiry_years.push_back(s.t_expiry);

    const v::SabrParams sp{c.sabr_alpha, c.sabr_rho, c.sabr_nu};

    // Strike list: explicit absolute strikes, moneyness offsets (bp from the forward), and/or ATM; default ATM.
    std::vector<double> strikes;
    strikes.reserve(c.strikes.size() + c.moneyness_bp.size() + 1);
    for (double k : c.strikes) strikes.push_back(k);
    for (double m : c.moneyness_bp) strikes.push_back(fs.rate + m / 1e4);
    if (c.atm || strikes.empty()) strikes.push_back(fs.rate);

    for (double strike : strikes) {
      const bool payer = c.payer_set ? c.payer : (strike >= fs.rate);
      const v::Payoff cp = payer ? v::Payoff::Payer : v::Payoff::Receiver;
      const double vol = c.has_sabr ? v::sabr_normal_vol(fs.rate, strike, s.t_expiry, sp) : c.normal_vol;
      out.point_cell.push_back(static_cast<double>(ci));
      out.strike.push_back(strike);
      out.moneyness_bp.push_back((strike - fs.rate) * 1e4);
      out.normal_vol.push_back(vol);
      out.price.push_back(v::swaption_price(fs.rate, fs.annuity, strike, vol, s.t_expiry, cp));
      out.vega.push_back(v::bachelier_vega<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity));
      out.delta.push_back(v::bachelier_delta<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity, cp));
      out.gamma.push_back(v::bachelier_gamma<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity));
      out.vanna.push_back(v::bachelier_vanna<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity));
      out.volga.push_back(v::bachelier_volga<double>(fs.rate, strike, vol, s.t_expiry, fs.annuity));
      out.payer.push_back(payer ? 1.0 : 0.0);
    }
  }
  out.n_points = static_cast<int>(out.price.size());
  out.price_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - clock0).count();
  return out;
}

// Thin JSON convenience over the native reprice (the run_json / pybind seam): parse -> price_vol_cube.
VolCube BundleSession::price_vol_cube_json(const std::string& spec_json) const {
  return price_vol_cube(vol_cube_spec_from_json(spec_json));
}

// ---- VolSurface: resolve schedules once, pre-index into one sample grid, pre-size the SoA output ----------
VolSurface::VolSurface(const VolCubeSpec& spec) : curve_(spec.curve), defs_(spec.cells) {
  const b::Date vd = b::Date::from_iso(spec.value_date);
  const b::SwapConv conv = b::swap_conv(spec.currency, 1.0, spec.index);

  // (1) resolve each cell's schedule (the calendar walk — ONCE) and collect the union of all schedule times.
  std::vector<double> times;
  std::vector<std::pair<double, std::vector<double>>> raw;  // (t_start, pay_times) per cell, indexed below
  cells_.reserve(spec.cells.size());
  raw.reserve(spec.cells.size());
  for (const VolCubeCell& c : spec.cells) {
    if (c.expiry.empty() || c.tenor.empty())
      throw std::invalid_argument("vol_surface: each cell needs a non-empty 'expiry' and 'tenor'");
    const b::Date expiry_date = b::resolve(c.expiry, vd);
    const b::Date swap_start = b::spot_date(expiry_date, conv.calendar, conv.spot_lag);
    const double tenor_years = conventions::period_years(c.tenor);
    if (!(tenor_years > 0.0))
      throw std::invalid_argument("vol_surface: unrecognized or non-positive tenor '" + c.tenor + "'");
    const int n = std::max(1, static_cast<int>(std::lround(tenor_years)));
    Cell cell;
    cell.t_expiry = b::curve_time(vd, expiry_date);
    const double t_start = b::curve_time(vd, swap_start);
    std::vector<double> pay_time;
    b::Date prev = swap_start;
    for (int i = 1; i <= n; ++i) {
      const b::Date pay = b::adjust(conv.calendar, swap_start.plus_months(12 * i), conv.bdc);
      cell.tau.push_back(b::year_frac(conv.fixed_dc, prev, pay));
      pay_time.push_back(b::curve_time(vd, pay));
      prev = pay;
    }
    times.push_back(t_start);
    times.insert(times.end(), pay_time.begin(), pay_time.end());
    cells_.push_back(std::move(cell));
    raw.emplace_back(t_start, std::move(pay_time));
  }

  // (2) the one sample grid, and (3) pre-index each cell's start/pay into it + assign output offsets + pre-size.
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  union_times_ = std::move(times);
  const auto idx_of = [&](double t) {
    return static_cast<std::size_t>(std::lower_bound(union_times_.begin(), union_times_.end(), t) -
                                    union_times_.begin());
  };
  int pt = 0;
  for (std::size_t ci = 0; ci < cells_.size(); ++ci) {
    cells_[ci].start_idx = idx_of(raw[ci].first);
    cells_[ci].pay_idx.reserve(raw[ci].second.size());
    for (double t : raw[ci].second) cells_[ci].pay_idx.push_back(idx_of(t));
    cells_[ci].point_offset = pt;
    const VolCubeCell& d = defs_[ci];
    std::size_t k = d.strikes.size() + d.moneyness_bp.size() + (d.atm ? 1 : 0);
    if (k == 0) k = 1;  // default ATM
    pt += static_cast<int>(k);
  }
  n_points_ = pt;
  out_.n_cells = static_cast<int>(cells_.size());
  out_.n_points = n_points_;
  out_.cell_forward.resize(cells_.size());
  out_.cell_annuity.resize(cells_.size());
  out_.cell_expiry_years.resize(cells_.size());
  for (std::vector<double>* col : {&out_.point_cell, &out_.strike, &out_.moneyness_bp, &out_.normal_vol,
                                   &out_.price, &out_.vega, &out_.delta, &out_.gamma, &out_.vanna, &out_.volga,
                                   &out_.payer})
    col->resize(n_points_);
  fwd_.assign(cells_.size(), 0.0);
  annuity_.assign(cells_.size(), 0.0);
  for (std::size_t ci = 0; ci < cells_.size(); ++ci) out_.cell_expiry_years[ci] = cells_[ci].t_expiry;
}

void VolSurface::set_sabr(const std::vector<double>& alpha, double rho, double nu) {
  if (alpha.size() != defs_.size())
    throw std::invalid_argument("vol_surface: set_sabr alpha length must equal n_cells");
  for (std::size_t ci = 0; ci < defs_.size(); ++ci) {
    defs_[ci].has_sabr = true;
    defs_[ci].sabr_alpha = alpha[ci];
    defs_[ci].sabr_rho = rho;
    defs_[ci].sabr_nu = nu;
  }
}

const VolCube& VolSurface::reprice(const BundleSession& sess) const {
  const auto clock0 = std::chrono::steady_clock::now();

  // Curve-dependent part: refresh per-cell forward/annuity only when x moved (a vol-only tick samples nothing).
  const Eigen::VectorXd& xnow = sess.x();
  const bool fa_valid = fa_x_.size() == xnow.size() && xnow.size() > 0 && (fa_x_.array() == xnow.array()).all();
  if (!fa_valid) {
    const std::vector<CurveSample> cs = sess.sample(union_times_);
    if (curve_ < 0 || curve_ >= static_cast<int>(cs.size()))
      throw std::invalid_argument("vol_surface: curve index out of range");
    const std::vector<double>& disc = cs[static_cast<std::size_t>(curve_)].discount;
    std::vector<double> df_pay;
    for (std::size_t ci = 0; ci < cells_.size(); ++ci) {
      const Cell& c = cells_[ci];
      df_pay.clear();
      for (std::size_t j : c.pay_idx) df_pay.push_back(disc[j]);
      const v::ForwardSwap fs = v::forward_swap(disc[c.start_idx], df_pay.back(), df_pay, c.tau);
      fwd_[ci] = fs.rate;
      annuity_[ci] = fs.annuity;
    }
    fa_x_ = xnow;
  }

  // Vol-dependent part: pure Bachelier/SABR into the pre-sized, index-addressed buffers (no allocation).
  for (std::size_t ci = 0; ci < cells_.size(); ++ci) {
    const Cell& c = cells_[ci];
    const VolCubeCell& d = defs_[ci];
    const double F = fwd_[ci], A = annuity_[ci], T = c.t_expiry;
    out_.cell_forward[ci] = F;
    out_.cell_annuity[ci] = A;
    const v::SabrParams sp{d.sabr_alpha, d.sabr_rho, d.sabr_nu};
    int k = c.point_offset;
    const auto price_one = [&](double strike) {
      const bool payer = d.payer_set ? d.payer : (strike >= F);
      const v::Payoff cp = payer ? v::Payoff::Payer : v::Payoff::Receiver;
      const double vol = d.has_sabr ? v::sabr_normal_vol(F, strike, T, sp) : d.normal_vol;
      out_.point_cell[k] = static_cast<double>(ci);
      out_.strike[k] = strike;
      out_.moneyness_bp[k] = (strike - F) * 1e4;
      out_.normal_vol[k] = vol;
      out_.price[k] = v::swaption_price(F, A, strike, vol, T, cp);
      out_.vega[k] = v::bachelier_vega<double>(F, strike, vol, T, A);
      out_.delta[k] = v::bachelier_delta<double>(F, strike, vol, T, A, cp);
      out_.gamma[k] = v::bachelier_gamma<double>(F, strike, vol, T, A);
      out_.vanna[k] = v::bachelier_vanna<double>(F, strike, vol, T, A);
      out_.volga[k] = v::bachelier_volga<double>(F, strike, vol, T, A);
      out_.payer[k] = payer ? 1.0 : 0.0;
      ++k;
    };
    bool any = false;
    for (double s : d.strikes) { price_one(s); any = true; }
    for (double m : d.moneyness_bp) { price_one(F + m / 1e4); any = true; }
    if (d.atm || !any) price_one(F);
  }
  out_.price_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - clock0).count();
  return out_;
}

}  // namespace swaps::api
