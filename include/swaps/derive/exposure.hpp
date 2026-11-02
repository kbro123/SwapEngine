#pragma once
// derive/exposure.hpp — the EPE / ENE / PFE counterparty-exposure computation behind the `exposure` verb
// (api/exposure.cpp decodes the request and serialises this; until 2026-11-02 the verb carried the behaviour,
// review item 2.9). Off a CALIBRATED single classic (Flat + Hermite) self-discounting curve: simulate a
// Gaussian curve-state grid x(t, path) = x_cal + OU(sigma, kappa; t)·Z, age each netting set's book to every node
// (calibration::roll_book), reprice it on the MC-exposure kernel (portfolio::CompiledPortfolio::npv_grid, ~1.9M
// full-book reprices/sec) and aggregate PER SET (xva::exposure_profile: netting within a set, never across sets).
//
// DEMO SCOPE: the kernel is single-curve, so xccy / custom-region / turn'd curves and non-swap positions are
// refused or skipped; the Gaussian proxy is illustrative -- a calibrated LGM / HW1F is a later step; the kernel
// is model-agnostic (pure f(exp(-W·x))). QuantLib-free.
#include <chrono>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "swaps/calibration/pnl_explain.hpp"  // roll_book: age the book to each exposure node
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/curve_spec.hpp"
#include "swaps/xva/exposure.hpp"

namespace swaps::derive {

namespace pf = swaps::portfolio;

// The simulation controls (the verb's request fields, with the verb's documented defaults).
struct ExposureParams {
  int n_paths = 4000;
  int n_nodes = 50;
  double horizon = 10.0;  // years
  double sigma = 0.008;   // absolute short-rate vol (~80 bp/yr)
  double kappa = 0.08;    // mean reversion -- bounds the curve-state dispersion (HW1F / OU)
  double pfe_q = 0.975;
};

// One netting set: its id and its book (a MultiCurveBook whose positions all discount on the set's CSA role).
struct ExposureSet {
  std::string id;
  pf::MultiCurveBook book;
};

struct ExposureProfileResult {
  xva::ExposureProfile profile;
  double mtm = 0.0;  // the set's value today
  int n = 0;         // single-curve swap positions the kernel priced
};
struct ExposureResult {
  std::optional<ExposureProfileResult> book;             // the legacy whole-book set, when one was given
  std::vector<std::pair<std::string, ExposureProfileResult>> sets;  // per netting set, in request order
  double wall_us = 0.0;  // the simulation + reprice + aggregation, microseconds (no calibration, no JSON)
  int n_paths = 0, n_nodes = 0;
};

// The single-curve filter: keep only swap-kind positions priced entirely off curve 0 (CompiledPortfolio is one
// self-discounting curve). Shared by the whole-book and the per-set paths.
inline pf::Portfolio to_single_curve_portfolio(const pf::MultiCurveBook& book) {
  pf::Portfolio port;
  for (const auto& b : book.positions) {
    if (b.kind != pf::MultiCurveBook::Kind::Swap) continue;
    if (b.fwd_curve != 0 || b.disc_curve != 0 || b.fixed_curve != 0) continue;
    port.positions.push_back({b.float_coupons, b.fixed_coupons, b.fixed_rate, b.notional});
  }
  return port;
}

// The guard: the kernel takes ONE classic Flat + Hermite self-discounting curve with no turns. `curves` is the
// calibrated bundle's curve list; returns that curve's modules.
inline std::vector<curve::CurveModule> exposure_curve_modules(const std::vector<pricing::CurveStructure>& curves) {
  if (curves.size() != 1)
    throw std::invalid_argument("exposure: demo restricted to a single classic Flat+Hermite self-discounting "
                                "curve (no custom regions/turns/xccy)");
  const pricing::CurveStructure& C0 = curves.front();
  const std::vector<curve::CurveModule> mods = C0.modules();
  const bool classic = mods.size() == 2 && mods[0].scheme == curve::Scheme::Flat &&
                       mods[1].scheme == curve::Scheme::Hermite;
  if (!classic || !C0.turns.empty())
    throw std::invalid_argument("exposure: demo restricted to a single classic Flat+Hermite self-discounting "
                                "curve (no custom regions/turns/xccy)");
  return mods;
}

// The whole computation off the calibrated state x_cal of the (guarded) curve. `book`: the legacy whole-book
// netting set, if any; `sets`: the typed netting sets. Each must hold at least one single-curve swap position.
inline ExposureResult exposure(const std::vector<pricing::CurveStructure>& curves, const Eigen::VectorXd& x_cal,
                               const ExposureParams& prm, const std::optional<pf::MultiCurveBook>& book,
                               const std::vector<ExposureSet>& sets) {
  const std::vector<curve::CurveModule> mods = exposure_curve_modules(curves);
  const int n_paths = std::max(1, prm.n_paths), n_nodes = std::max(2, prm.n_nodes);
  if (book && to_single_curve_portfolio(*book).positions.empty())
    throw std::invalid_argument("exposure: no single-curve swap positions in the book");
  for (const ExposureSet& s : sets)
    if (to_single_curve_portfolio(s.book).positions.empty())
      throw std::invalid_argument("exposure: netting set '" + s.id + "' has no single-curve swap positions");

  // Node times 0..horizon (node 0 = today: sqrt(0)=0 -> deterministic, EPE(0)=max(MtM,0)).
  std::vector<double> node_time(static_cast<std::size_t>(n_nodes));
  for (int j = 0; j < n_nodes; ++j) node_time[static_cast<std::size_t>(j)] = prm.horizon * static_cast<double>(j) / (n_nodes - 1);

  // Single-factor Gaussian curve-state grid (HW1F-style parallel level move): one Brownian shock per
  // (path, node) shifts the whole knot vector -- the illustrative desk proxy (a real LGM adds mean reversion
  // + a term structure of vol; the kernel is indifferent). NODE-MAJOR layout: column (path p, node j) at
  // j*n_paths + p (the layout exposure.hpp/npv_grid assume). Node 0 (t=0) has sd=0 -> deterministic today.
  // ONE grid serves every netting set: the curve states are market scenarios, shared across counterparties.
  const int K = static_cast<int>(x_cal.size());
  Eigen::MatrixXd X(K, static_cast<Eigen::Index>(n_paths) * n_nodes);
  std::mt19937_64 rng(0x5E0Fu);
  std::normal_distribution<double> Z(0.0, 1.0);
  for (int j = 0; j < n_nodes; ++j) {
    const double t = node_time[static_cast<std::size_t>(j)];
    // Ornstein-Uhlenbeck (HW1F) stationary variance Var(t)=σ²(1−e^{−2κt})/(2κ) -- plateaus instead of
    // diverging like pure Brownian, so the curve-state (and the discount factors) stay in a plausible range.
    const double var = prm.kappa > 1e-9 ? prm.sigma * prm.sigma * (1.0 - std::exp(-2.0 * prm.kappa * t)) / (2.0 * prm.kappa)
                                         : prm.sigma * prm.sigma * t;
    const double sd = std::sqrt(var);
    for (int p = 0; p < n_paths; ++p) {
      const Eigen::Index col = static_cast<Eigen::Index>(j) * n_paths + p;
      const double dz = sd > 0.0 ? sd * Z(rng) : 0.0;  // one parallel shock for all knots at this state
      for (int k = 0; k < K; ++k) X(k, col) = x_cal[k] + dz;
    }
  }

  // The kernel + aggregation, PER NETTING SET (the legacy whole book is just the first "set" when present):
  // exposure aggregates WITHIN a set -- each portfolio gets its own colwise-net + EPE/ENE/PFE reduction --
  // never across sets. THE BOOK IS AGED TO EVERY NODE (E3-F2, 2026-09-10): at node t_j the surviving cashflows
  // are the ones paying after t_j, re-timed from the node (roll_book, the same helper pnl_explain's roll leg
  // uses), and priced off the node's simulated curve states. A netting set whose book has fully matured at a
  // node contributes zero there. The per-node compile sits inside the timed section.
  const auto t0 = std::chrono::steady_clock::now();
  const auto profile_of = [&](const pf::MultiCurveBook& b) {
    ExposureProfileResult r;
    Eigen::MatrixXd net(1, static_cast<Eigen::Index>(n_paths) * n_nodes);  // the netted value per state
    for (int j = 0; j < n_nodes; ++j) {
      const pf::Portfolio port = to_single_curve_portfolio(calibration::roll_book(b, node_time[static_cast<std::size_t>(j)], /*shift=*/true));
      const Eigen::Index c0 = static_cast<Eigen::Index>(j) * n_paths;
      if (port.positions.empty()) {
        net.middleCols(c0, n_paths).setZero();
        continue;
      }
      const pf::CompiledPortfolio cp(mods, port);
      const Eigen::MatrixXd Xj = X.middleCols(c0, n_paths);
      net.middleCols(c0, n_paths) = cp.npv_grid(Xj).colwise().sum();
    }
    const pf::Portfolio today = to_single_curve_portfolio(b);
    r.mtm = pf::CompiledPortfolio(mods, today).total_npv(x_cal);
    r.n = static_cast<int>(today.positions.size());
    r.profile = xva::exposure_profile(net, n_paths, n_nodes, node_time, prm.pfe_q);
    return r;
  };
  ExposureResult out;
  if (book) out.book = profile_of(*book);
  for (const ExposureSet& s : sets) out.sets.emplace_back(s.id, profile_of(s.book));
  out.wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  out.n_paths = n_paths;
  out.n_nodes = n_nodes;
  return out;
}

}  // namespace swaps::derive
