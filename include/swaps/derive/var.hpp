#pragma once
// derive/var.hpp — full-revaluation VALUE-AT-RISK / EXPECTED SHORTFALL (E7, P14): the `var` verb's whole computation as
// library code over any calibration::CalibrationSession (the verb instantiates it with api::BundleSession; this header
// never includes api).
//
// Two modes, exactly one per request:
//   SUPPLIED -- a P&L series handed in (an external full reval or a realised history) is reduced as it stands.
//   REVAL    -- calibrate the base ONCE, then for each market move fork the fitted state and reprice the book through
//               its compiled twin (portfolio::XccyFxScaledBooks: one compiled book per distinct FX factor):
//               pnl_k = npv(move_k) - base_npv. N moves cost one calibration plus N compiled repricings.
// Both reduce the distribution the same way (pnl_distribution).
//
// A MOVE is the `scenario` verb's (derive::ScenarioMove) resolved by the SAME rule (resolve_scenario_move): after SC1
// every verb ADDS an explicit shift_curve to parallel_bp -- a parallel reaches outright curves only (a spread curve
// inherits it from its base), each key adds bp / 1e4 to the curve it names, FX bumps compound in request order into
// one factor on every xccy position (SC2 will make that per pair; var picks it up through ResolvedMove for free).
//
// CONVENTION. var / es are LOSSES (positive for a loss): loss = -P&L. VaR at confidence q is -(the numpy "type 7"
// linearly interpolated (1-q) lower-tail P&L quantile, position (1-q)(N-1)); ES at q is -(the mean of the
// m = clamp(round((1-q) N), 1, N) smallest P&Ls). var_pnl / es_pnl are the same numbers as signed P&L.
//
// Bitwise with the verb it replaces (tests/golden/scenario/var_*.json): the fork through
// calibration::shift_interp_forwards (== var.cpp's segment add), the compiled book keyed by factor to 1e-12 (the
// factor-1 book is xccy_fx_scaled(book, 1.0), an exact copy), the mean / variance accumulated in pnl order, one
// std::sort, m and the quantile position in the verb's own int -> double conversions.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_state.hpp"  // shift_interp_forwards
#include "swaps/calibration/diagnostics.hpp"   // CalibrationSession, seed_or_flat (and lm.hpp's CalibrationResult)
#include "swaps/calibration/regularize.hpp"
#include "swaps/derive/scenario.hpp"           // ScenarioMove, ResolvedMove, resolve_scenario_move
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/portfolio/xccy_fx_scaled.hpp"

namespace swaps::derive {

// ---- reducing a P&L distribution -------------------------------------------------------------------------------

// numpy "type 7": the linearly interpolated order statistic at position p (n - 1) of a SORTED-ascending, non-empty s.
// p = 1 lands on the last point (reached for q below 2^-53, where 1 - q rounds to 1).
inline double type7_quantile_sorted(const std::vector<double>& s, double p) {
  const int n = static_cast<int>(s.size());
  if (n == 1) return s[0];
  const double pos = p * (n - 1);
  const double lo = std::floor(pos);
  const int i = static_cast<int>(lo);
  if (i >= n - 1) return s[static_cast<std::size_t>(n - 1)];
  const double frac = pos - lo;
  return s[static_cast<std::size_t>(i)] * (1.0 - frac) + s[static_cast<std::size_t>(i + 1)] * frac;
}

struct VarQuantile {
  double q = 0.0;
  double var = 0.0, es = 0.0;          // losses: -var_pnl, -es_pnl
  double var_pnl = 0.0, es_pnl = 0.0;  // signed P&L (negative for a loss)
};

// VaR and ES at confidence q from a SORTED-ascending, non-empty P&L vector.
inline VarQuantile var_es_at(const std::vector<double>& sorted, double q) {
  const int n = static_cast<int>(sorted.size());
  const double p = 1.0 - q;
  VarQuantile out;
  out.q = q;
  out.var_pnl = type7_quantile_sorted(sorted, p);
  int m = static_cast<int>(std::lround(p * n));
  if (m < 1) m = 1;
  if (m > n) m = n;
  double sum = 0.0;
  for (int i = 0; i < m; ++i) sum += sorted[static_cast<std::size_t>(i)];  // the m smallest (worst) P&Ls
  out.es_pnl = sum / m;
  out.var = -out.var_pnl;
  out.es = -out.es_pnl;
  return out;
}

inline void check_var_quantiles(const std::vector<double>& quantiles) {
  if (quantiles.empty()) throw std::invalid_argument("var: 'quantiles' is empty");
  for (double q : quantiles)
    if (!(q > 0.0 && q < 1.0)) throw std::invalid_argument("var: every quantile must be in (0, 1)");
}

struct PnlDistribution {
  int n = 0;
  double mean_pnl = 0.0;
  double stdev_pnl = 0.0;              // sample (n - 1); 0 for a single point
  std::vector<double> pnl_sorted;      // ascending: the whole distribution
  std::vector<VarQuantile> quantiles;  // in request order
};

inline PnlDistribution pnl_distribution(std::vector<double> pnl, const std::vector<double>& quantiles) {
  if (pnl.empty()) throw std::invalid_argument("var: 'pnl' array is empty");
  check_var_quantiles(quantiles);
  PnlDistribution out;
  out.n = static_cast<int>(pnl.size());
  double mean = 0.0;
  for (double v : pnl) mean += v;
  mean /= out.n;
  double var_acc = 0.0;
  for (double v : pnl) var_acc += (v - mean) * (v - mean);
  out.mean_pnl = mean;
  out.stdev_pnl = out.n > 1 ? std::sqrt(var_acc / (out.n - 1)) : 0.0;
  std::sort(pnl.begin(), pnl.end());
  out.pnl_sorted = std::move(pnl);
  out.quantiles.reserve(quantiles.size());
  for (double q : quantiles) out.quantiles.push_back(var_es_at(out.pnl_sorted, q));
  return out;
}

// ---- REVAL: the P&L of a book under a set of market moves -------------------------------------------------------

struct VarRevalRequest {
  calibration::BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;  // absent => the flat seed
  calibration::RegSpec reg;
  portfolio::MultiCurveBook book;     // required in reval mode (the codec's `need`)
  std::vector<ScenarioMove> scenarios;
};

struct VarRevalResult {
  calibration::CalibrationResult calibration;  // the base's solve (emitted from SC3 on)
  double base_npv = 0.0;
  int n_positions = 0;
  double reval_us = 0.0;    // wall-clock of the fork + reprice sweep
  std::vector<double> pnl;  // one per move, in REQUEST order
};

template <calibration::CalibrationSession Session>
VarRevalResult var_reval(VarRevalRequest r) {
  if (r.bundle.n_curves() == 0) throw std::invalid_argument("var: bundle has no curves");
  if (r.scenarios.empty()) throw std::invalid_argument("var: 'scenarios' is empty");
  // Every move is resolved BEFORE calibrating, so a bad role costs nothing (the verb found it mid-sweep).
  std::vector<ResolvedMove> moves;
  moves.reserve(r.scenarios.size());
  for (const ScenarioMove& m : r.scenarios) moves.push_back(resolve_scenario_move(m, r.bundle.curves, "var"));

  Session sess(std::move(r.bundle));
  const calibration::BundleProblem& P = sess.problem();
  VarRevalResult out;
  out.calibration = sess.calibrate(calibration::seed_or_flat(P, r.x0, "var"), r.reg);
  const Eigen::VectorXd x_base = sess.x();  // the anchor every move forks from; never mutated

  portfolio::XccyFxScaledBooks books(P.curves, std::move(r.book));
  out.n_positions = static_cast<int>(books.book().positions.size());
  out.base_npv = books.at(1.0).npv(x_base);
  out.pnl.reserve(moves.size());
  const auto t0 = std::chrono::steady_clock::now();
  for (const ResolvedMove& rm : moves)
    out.pnl.push_back(books.at(rm.fx_factor).npv(calibration::shift_interp_forwards(P, x_base, rm.curve_delta)) -
                      out.base_npv);
  out.reval_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

// ---- the `var` verb's whole computation -------------------------------------------------------------------------

struct VarRequest {
  std::vector<double> quantiles{0.95, 0.99};  // THE default (absent); an explicit [] is refused
  std::optional<std::vector<double>> pnl;     // SUPPLIED
  std::optional<VarRevalRequest> reval;       // REVAL
};

struct VarResult {
  std::optional<VarRevalResult> reval;  // present <=> mode "reval"
  PnlDistribution distribution;
};

template <calibration::CalibrationSession Session>
VarResult var(VarRequest r) {
  check_var_quantiles(r.quantiles);  // before any calibration, as the verb did
  if (r.pnl && r.reval)
    throw std::invalid_argument("var: give either 'pnl' (supplied) or 'bundle' + 'book' + 'scenarios' (reval), not both");
  if (!r.pnl && !r.reval) throw std::invalid_argument("var: needs either 'pnl' or 'bundle'+'scenarios'");
  VarResult out;
  if (r.pnl) {
    out.distribution = pnl_distribution(std::move(*r.pnl), r.quantiles);
    return out;
  }
  out.reval = var_reval<Session>(std::move(*r.reval));
  out.distribution = pnl_distribution(out.reval->pnl, r.quantiles);
  return out;
}

}  // namespace swaps::derive
