// Options seam: the `swaption` verb, the `sabr_calibrate` verb, the vol-cube codec, BundleSession::price_vol_cube
// and the compiled VolSurface (declared in swaps/api/options.hpp). A swaption's underlying schedule is
// build::swaption_schedule (the one walk); the cube is a VolSurface the session holds per spec shape; the
// `swaption` verb is one cell per trade of that cube. Reuses BundleSession + the build/ date-schedule layer + the
// vol/ analytics. QuantLib-free.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/options.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/build/swaption_schedule.hpp"
#include "swaps/conventions_data.hpp"
#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/sabr_calibration.hpp"
#include "swaps/vol/swaption.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace v = swaps::vol;

namespace {
// SABR β from a {alpha,rho,nu[,beta]} object: the CEV backbone in [0,1], 0 (normal SABR) when absent. Every
// vol verb dropped it until 2026-09-10 (E3-F1: {"alpha":0.30,"beta":1.0}, a 30 % Black level, priced as a
// 3039 bp NORMAL vol); an out-of-range value is refused rather than silently clamped.
double sabr_beta_of(const json::object& s) {
  const double b = jd(s, "beta", 0.0);
  if (!(b >= 0.0 && b <= 1.0)) throw std::invalid_argument("sabr: beta must be in [0, 1] (0 = normal SABR, 1 = lognormal)");
  return b;
}

}  // namespace

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

// Stateless SABR strip calibration verb (see options.hpp). No bundle: fit (alpha,rho,nu) to (strikes,
// market_vols) at (forward, expiry) via vol/sabr_calibration.hpp, and report the arbitrage-free status.
std::string sabr_calibrate_json(const json::object& request) {
  const json::object& top = request;
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
    guess = v::SabrParams{jd(g, "alpha", 0.0), jd(g, "rho", 0.0), jd(g, "nu", 0.0), sabr_beta_of(g)};
  }
  const v::SabrCalibResult r = v::sabr_calibrate(forward, expiry, strikes, mvols, guess);
  const double lo = jd(o, "arb_lo", strikes.front());
  const double hi = jd(o, "arb_hi", strikes.back());

  json::object out;
  out["alpha"] = r.params.alpha;
  out["rho"] = r.params.rho;
  out["nu"] = r.params.nu;
  out["beta"] = r.params.beta;  // the backbone the fit ran at (fixed at guess.beta; 0 = normal SABR)
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
  spec.index = require_index_arg(o, "vol_cube");
  spec.currency = currency_for_index(spec.index, js(o, "currency"), "vol_cube");
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
      cell.sabr_beta = sabr_beta_of(sj);
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

// NATIVE batched vol-cube reprice off the CURRENTLY CALIBRATED curve — the options hot path, no JSON. The session
// holds ONE compiled VolSurface per spec SHAPE (value date, currency, index, curve, the cells' expiry / tenor and
// their strike counts): the surface resolves the schedules once and caches each cell's (forward, annuity) against
// the curve state x, so a vol-only reprice (a SABR-slider tick) samples the curve NOT AT ALL. The spec's vols and
// strikes are copied onto the surface's cells and it reprices; the result is a copy of its SoA output. Reuses the
// calibrated/streaming session, so a live vol surface reprices with no recalibration. This is what benchmarks and
// native clients call; the JSON verb is a thin wrapper below.
VolCube BundleSession::price_vol_cube(const VolCubeSpec& spec) const {
  if (spec.index.empty()) throw std::invalid_argument("vol_cube: VolCubeSpec.index is required (the swap's DB index)");
  std::string key = spec.value_date;
  key += '|'; key += spec.currency; key += '|'; key += spec.index; key += '|'; key += std::to_string(spec.curve);
  for (const VolCubeCell& c : spec.cells) {
    key += '|'; key += c.expiry; key += '/'; key += c.tenor; key += '/';
    key += std::to_string(c.strikes.size()); key += ','; key += std::to_string(c.moneyness_bp.size()); key += ',';
    key += (c.atm ? '1' : '0');
  }
  auto it = vol_surfaces_.find(key);
  if (it == vol_surfaces_.end()) it = vol_surfaces_.emplace(key, std::make_unique<VolSurface>(spec)).first;
  VolSurface& surf = *it->second;
  surf.cells() = spec.cells;  // the vols / strikes of THIS request (the same shape, keyed above)
  return surf.reprice(*this);
}

// Thin JSON convenience over the native reprice (the run_json / pybind seam): parse -> price_vol_cube.
VolCube BundleSession::price_vol_cube_json(const std::string& spec_json) const {
  return price_vol_cube(vol_cube_spec_from_json(spec_json));
}

// The `swaption` verb: ONE cell per trade of the vol cube -- its strike and the ATM point -- priced through the
// session's surface and reshaped per trade (what the verb has always returned).
std::string swaption_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o = top.contains("swaption") ? top.at("swaption").as_object() : top;

  const std::string vd_iso = js(o, "value_date");
  if (vd_iso.empty()) throw std::invalid_argument("swaption: missing 'value_date'");
  if (!o.contains("bundle")) throw std::invalid_argument("swaption: missing 'bundle'");
  VolCubeSpec spec;
  spec.value_date = vd_iso;
  spec.index = require_index_arg(o, "swaption");
  spec.currency = currency_for_index(spec.index, js(o, "currency"), "swaption");
  spec.curve = static_cast<int>(jd(o, "curve", 0.0));

  // Calibrate the bundle to its markets -> the curve we price off.
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  sess.calibrate(flat_x0(sess.problem()));  // smoothing is the bundle's region data (regions[].reg_lambda)

  if (o.contains("trades") && o.at("trades").is_array()) {
    for (const auto& te : o.at("trades").as_array()) {
      const json::object& t = te.as_object();
      VolCubeCell c;
      c.expiry = js(t, "expiry");
      c.tenor = js(t, "tenor");
      if (c.expiry.empty() || c.tenor.empty())
        throw std::invalid_argument("swaption: each trade needs a non-empty 'expiry' and 'tenor'");
      c.strikes.push_back(jd(t, "strike", 0.0));
      c.atm = true;  // the ATM point beside the strike: the trade's atm_normal_vol
      c.payer_set = true;
      c.payer = jb(t, "payer", true);
      if (t.contains("sabr") && t.at("sabr").is_object()) {
        const auto& sj = t.at("sabr").as_object();
        c.has_sabr = true;
        c.sabr_alpha = jd(sj, "alpha", 0.0);
        c.sabr_rho = jd(sj, "rho", 0.0);
        c.sabr_nu = jd(sj, "nu", 0.0);
        c.sabr_beta = sabr_beta_of(sj);
      } else {
        c.normal_vol = jd(t, "normal_vol", 0.0);
      }
      spec.cells.push_back(std::move(c));
    }
  }
  const VolCube cube = sess.price_vol_cube(spec);

  json::array out_trades;
  for (std::size_t ci = 0; ci < spec.cells.size(); ++ci) {
    const VolCubeCell& c = spec.cells[ci];
    const std::size_t k = 2 * ci;  // the cell's two points: [strike, atm]
    json::object r;
    r["expiry"] = c.expiry;
    r["tenor"] = c.tenor;
    r["strike"] = cube.strike[k];
    r["payer"] = c.payer;
    r["expiry_years"] = cube.cell_expiry_years[ci];
    r["forward"] = cube.cell_forward[ci];
    r["annuity"] = cube.cell_annuity[ci];
    r["normal_vol"] = cube.normal_vol[k];
    r["atm_normal_vol"] = cube.normal_vol[k + 1];
    r["price"] = cube.price[k];
    r["vega"] = cube.vega[k];
    r["delta"] = cube.delta[k];
    out_trades.push_back(std::move(r));
  }
  json::object out;
  out["trades"] = std::move(out_trades);
  return json::serialize(json::value(std::move(out)));
}

// ---- VolSurface: resolve schedules once, pre-index into one sample grid, pre-size the SoA output ----------
VolSurface::VolSurface(const VolCubeSpec& spec) : curve_(spec.curve), defs_(spec.cells) {
  if (spec.index.empty()) throw std::invalid_argument("VolSurface: VolCubeSpec.index is required (the swap's DB index)");
  const b::Date vd = b::Date::from_iso(spec.value_date);
  const b::SwapConv conv = b::swap_conv(spec.currency, spec.index);

  // (1) resolve each cell's schedule (the calendar walk — ONCE) and collect the union of all schedule times.
  std::vector<double> times;
  std::vector<std::pair<double, std::vector<double>>> raw;  // (t_start, pay_times) per cell, indexed below
  cells_.reserve(spec.cells.size());
  raw.reserve(spec.cells.size());
  for (const VolCubeCell& c : spec.cells) {
    b::SwaptionSchedule ss = b::swaption_schedule(vd, conv, c.expiry, c.tenor, "vol_cube");  // the one walk
    Cell cell;
    cell.t_expiry = ss.t_expiry;
    cell.tau = std::move(ss.tau);
    times.push_back(ss.t_start);
    times.insert(times.end(), ss.pay_time.begin(), ss.pay_time.end());
    cells_.push_back(std::move(cell));
    raw.emplace_back(ss.t_start, std::move(ss.pay_time));
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
    const v::SabrParams sp{d.sabr_alpha, d.sabr_rho, d.sabr_nu, d.sabr_beta};
    int k = c.point_offset;
    const auto price_one = [&](double strike) {
      const bool payer = d.payer_set ? d.payer : (strike >= F);
      const v::Payoff cp = payer ? v::Payoff::Payer : v::Payoff::Receiver;
      const double vol = d.has_sabr ? v::sabr_normal_vol(F, strike, T, sp) : d.normal_vol;
      out_.point_cell[k] = static_cast<double>(ci);
      out_.strike[k] = strike;
      out_.moneyness_bp[k] = (strike - F) * 1e4;
      out_.normal_vol[k] = vol;
      const v::BachelierGreeks<double> g = v::bachelier_greeks<double>(F, strike, vol, T, A, cp);  // R1: one pass
      out_.price[k] = g.price;
      out_.vega[k] = g.vega;
      out_.delta[k] = g.delta;
      out_.gamma[k] = g.gamma;
      out_.vanna[k] = g.vanna;
      out_.volga[k] = g.volga;
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


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string swaption_json(const std::string& request) { return swaption_json(json::parse(request).as_object()); }
std::string sabr_calibrate_json(const std::string& request) { return sabr_calibrate_json(json::parse(request).as_object()); }
}  // namespace swaps::api
