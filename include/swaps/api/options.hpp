#pragma once
// The swaption VOL object model and the options verbs (api/options.cpp): VolCube (the flat SoA result), VolCubeCell /
// VolCubeSpec (the native input), VolSurface (the compiled surface: schedules resolved once, forward / annuity cached
// against the curve state, a pure Bachelier / SABR pass per reprice), the `swaption` / `sabr_calibrate` verbs, and
// the two index-argument readers every vol verb shares. BundleSession::price_vol_cube holds one VolSurface per spec
// shape and reprices it; the `swaption` verb is one cell per trade of that cube (2026-11-01, review item 2.2: the
// schedule walk was written four times and the cube re-implemented the surface with three ad-hoc caches).
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/json/fwd.hpp>

namespace swaps::api {

class BundleSession;

// Result of a batched swaption VOL CUBE reprice off the calibrated curve (BundleSession::price_vol_cube_json).
// Flat SoA (parallel arrays), so the whole expiry x tenor x strike surface reprices into reused buffers on the
// engine hot path — the options analogue of price_portfolio. The curve-dependent part (per-cell forward /
// annuity / expiry) is computed once from a SINGLE curve sample over the union of all schedule times; every
// strike is then a pure Bachelier/SABR evaluation in C++. `price_us` is the ENGINE-measured pricing time (a
// steady_clock pair around the reprice only, no JSON/marshalling), the exact analogue of price_us/last_solve_us.
// Per-cell arrays have length n_cells; per-point arrays have length n_points, and point i belongs to the cell
// index point_cell[i]. All arrays are double (int-valued ones like point_cell/payer carry whole numbers) so the
// one STRUCT marshaller (VEC) covers every field.
struct VolCube {
  std::vector<double> cell_forward;        // per cell: forward swap rate off the calibrated curve
  std::vector<double> cell_annuity;        // per cell: Σ τ_i·DF_i (the numeraire)
  std::vector<double> cell_expiry_years;   // per cell: option expiry in curve time (ACT/365F)
  std::vector<double> point_cell;          // per point: which cell (index into the per-cell arrays)
  std::vector<double> strike;              // per point: absolute strike (rate)
  std::vector<double> moneyness_bp;        // per point: (strike − forward) in bp
  std::vector<double> normal_vol;          // per point: Bachelier normal vol used (SABR at the strike, or flat)
  std::vector<double> price;               // per point: swaption price (annuity units)
  std::vector<double> vega;                // per point: dV/dσ
  std::vector<double> delta;               // per point: dV/dF
  std::vector<double> gamma;               // per point: d²V/dF²
  std::vector<double> vanna;               // per point: d²V/(dF dσ)
  std::vector<double> volga;               // per point: d²V/dσ²
  std::vector<double> payer;               // per point: 1 payer, 0 receiver
  double price_us = 0.0;                   // pure engine pricing time of the cube, microseconds
  int n_cells = 0;
  int n_points = 0;
};

// Native input for the vol cube (the C++ analogue of the JSON `vol_cube` document) — so callers and benchmarks
// build a surface as typed structs, no JSON in the hot path. sabr wins over normal_vol when has_sabr; strikes
// combine absolute `strikes` + `moneyness_bp` offsets from the forward + `atm`; `payer` is honoured only when
// `payer_set`, else the OTM convention (payer above the forward) is used.
struct VolCubeCell {
  std::string expiry;
  std::string tenor;
  bool has_sabr = false;
  double sabr_alpha = 0.0, sabr_rho = 0.0, sabr_nu = 0.0;
  double sabr_beta = 0.0;  // CEV backbone in [0,1]; 0 = normal SABR (E3-F1: was dropped by every verb until 2026-09-10)
  double normal_vol = 0.0;
  bool payer_set = false;
  bool payer = true;
  std::vector<double> strikes;       // absolute strike rates
  std::vector<double> moneyness_bp;  // strikes at forward + bp/1e4
  bool atm = false;                  // include the ATM strike (forward)
};
struct VolCubeSpec {
  std::string value_date;
  std::string currency;  // optional; must agree with the index's DB currency when given
  std::string index;     // REQUIRED: the swap's DB index (carries the product conventions)
  int curve = 0;
  std::vector<VolCubeCell> cells;
};

// A COMPILED vol surface: resolve a FIXED set of swaption cells' schedules ONCE (the calendar walk), pre-index
// them into a single sample grid, and pre-size the flat SoA output — then `reprice()` off the current curve
// with zero per-call schedule/key/allocation work: one curve sample (only when x moved), then a pure
// Bachelier/SABR pass writing straight into reused, index-addressed buffers. The stateful streaming analog of
// BundleSession::price_vol_cube for a live surface / a fixed swaption book repriced every tick — this is what
// beats QuantLib's native loop (price_vol_cube trades some of that away for a fresh spec on every call). The
// referenced session must outlive the surface. Not thread-safe (reuses one output buffer). To move the vols
// (a SABR-slider tick) mutate `cells()` in place and call `reprice()` — no schedule rebuild, no resample.
class VolSurface {
 public:
  explicit VolSurface(const VolCubeSpec& spec);   // resolve schedules + pre-index + pre-size (curve-independent)
  const VolCube& reprice(const BundleSession& sess) const;  // sample iff x moved, then price into reused VolCube
  // The surface's per-cell forward / annuity after the last reprice (the vega ladder reads them).
  const std::vector<double>& forwards() const { return fwd_; }
  const std::vector<double>& annuities() const { return annuity_; }
  double expiry_years(std::size_t ci) const { return cells_[ci].t_expiry; }
  // Overwrite every cell's SABR (alpha per cell, rho/nu shared) — the streaming update for a live surface (a
  // level/skew slider) without touching the schedules. alpha.size() must equal n_cells.
  void set_sabr(const std::vector<double>& alpha, double rho, double nu);
  std::vector<VolCubeCell>& cells() { return defs_; }        // mutate vols/strikes in place, then reprice()
  const std::vector<VolCubeCell>& cells() const { return defs_; }
  int n_cells() const { return static_cast<int>(defs_.size()); }
  int n_points() const { return n_points_; }

 private:
  struct Cell {
    double t_expiry = 0.0;
    std::vector<double> tau;
    std::size_t start_idx = 0;         // index into union_times_ / the sampled discount vector
    std::vector<std::size_t> pay_idx;
    int point_offset = 0;              // first output-point index for this cell
  };
  int curve_ = 0;
  int n_points_ = 0;
  std::vector<Cell> cells_;                   // resolved schedules (built once)
  std::vector<VolCubeCell> defs_;             // per-cell vol model + strike specs (mutable between reprices)
  std::vector<double> union_times_;           // the one sample grid (sorted, unique)
  mutable std::vector<double> fwd_, annuity_; // per-cell curve-dependent, refreshed when x moves
  mutable Eigen::VectorXd fa_x_;              // the x fwd_/annuity_ were computed at
  mutable VolCube out_;                       // pre-sized, reused across reprices
};

// Parse a `vol_cube` JSON document into the native VolCubeSpec (used by the JSON verb and to build a VolSurface
// from a spec string in the binding). Definition in api/options.cpp.
VolCubeSpec vol_cube_spec_from_json(const std::string& spec_json);

// The swap's DB index is REQUIRED (it carries the product conventions); `currency` is optional and, if given,
// must agree with the index row (PRINCIPLES.md P2: no "USD"/"USD-SOFR" defaults). Shared by every vol verb.
std::string require_index_arg(const boost::json::object& o, const char* verb);
std::string currency_for_index(const std::string& index, const std::string& given, const char* verb);

// request = {"swaption": {value_date, bundle, currency?, index?, curve?, trades:[{expiry,tenor,strike,
// payer?, normal_vol?|sabr?}]}} -> {"trades":[{forward,annuity,price,normal_vol,atm_normal_vol,vega,delta,...}]}.
// One cell per trade of the vol cube (strike + ATM), reshaped per trade.
std::string swaption_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string swaption_json(const std::string& request);

// Stateless SABR strip calibration (no bundle). request = {"sabr_calibrate": {forward, expiry, strikes:[...],
// market_vols:[...], guess?:{alpha,rho,nu}, arb_lo?, arb_hi?}} -> {alpha, rho, nu, rms, iterations, converged,
// arbitrage_free}. Units are the model's decimals (forward/strikes absolute rates, vols normal/absolute).
std::string sabr_calibrate_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string sabr_calibrate_json(const std::string& request);

}  // namespace swaps::api
