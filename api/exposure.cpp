// Exposure seam: the stateless "exposure" run_json verb — an EPE/ENE/PFE counterparty-exposure profile for a
// swap book off a CALIBRATED SOFR curve. Calibrates the bundle (like swaption_json), converts the book to the
// native single-curve Portfolio, simulates a Gaussian curve-state grid x(t,path)=x_cal+sigma·sqrt(t)·Z, and
// runs the MC-exposure kernel (CompiledPortfolio::npv_grid, ~1.9M full-book reprices/sec) + the EPE/ENE/PFE
// aggregation (swaps/xva/exposure.hpp). The whole book is one netting set. QuantLib-free.
//
// DEMO SCOPE: a single classic (Flat+Hermite) self-discounting curve; xccy / region / turn'd curves and
// non-swap positions are rejected/skipped (CompiledPortfolio is single-curve). The Gaussian curve-state proxy
// is illustrative — a calibrated LGM/HW1F is a later step; the kernel is model-agnostic (pure f(exp(-W·x))).
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/exposure.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/xva/exposure.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace pf = swaps::portfolio;
namespace xva = swaps::xva;

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
json::array vecf(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}
}  // namespace

std::string exposure_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o =
      top.contains("exposure") && top.at("exposure").is_object() ? top.at("exposure").as_object() : top;

  const int n_paths = std::max(1, static_cast<int>(jd(o, "n_paths", 4000)));
  const int n_nodes = std::max(2, static_cast<int>(jd(o, "n_nodes", 50)));
  const double horizon = jd(o, "horizon_years", 10.0);
  const double sigma = jd(o, "sigma", 0.008);   // absolute short-rate vol (~80 bp/yr)
  const double kappa = jd(o, "kappa", 0.08);    // mean reversion — bounds the curve-state dispersion (HW1F)
  const double pfe_q = jd(o, "pfe_q", 0.975);
  if (!o.contains("bundle")) throw std::invalid_argument("exposure: missing 'bundle'");
  if (!o.contains("book")) throw std::invalid_argument("exposure: missing 'book'");

  // Calibrate the bundle -> the discount curve we reprice against (same recipe as swaption_json).
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  for (int c = 0; c < static_cast<int>(sess.problem().curves.size()); ++c) reg.curves.push_back(c);
  sess.calibrate(flat_x0(sess.problem()), reg);

  // Guard: the CompiledPortfolio exposure kernel is a single classic Flat+Hermite self-discounting curve.
  const auto& C0 = sess.problem().curves[0];
  if (sess.problem().n_curves() != 1 || !C0.regions.empty() || !C0.turns.empty())
    throw std::invalid_argument("exposure: demo restricted to a single classic Flat+Hermite self-discounting "
                                "curve (no regions/turns/xccy)");
  const Eigen::VectorXd x_cal = sess.x();  // node-0 center; length = meeting+back knots

  // Book -> native single-curve Portfolio (a straight coupon-field copy; swap-kind, all-curve-0 positions).
  const pf::MultiCurveBook book = book_from_json(o.at("book"));
  pf::Portfolio port;
  for (const auto& b : book.positions) {
    if (b.kind != pf::MultiCurveBook::Kind::Swap) continue;
    if (b.fwd_curve != 0 || b.disc_curve != 0 || b.fixed_curve != 0) continue;
    port.positions.push_back({b.float_coupons, b.fixed_coupons, b.fixed_rate, b.notional});
  }
  if (port.positions.empty())
    throw std::invalid_argument("exposure: no single-curve swap positions in the book");
  const pf::CompiledPortfolio cp(C0.meeting, C0.back, port);

  // Node times 0..horizon (node 0 = today: sqrt(0)=0 -> deterministic, EPE(0)=max(MtM,0)).
  std::vector<double> node_time(n_nodes);
  for (int j = 0; j < n_nodes; ++j) node_time[j] = horizon * static_cast<double>(j) / (n_nodes - 1);

  // Single-factor Gaussian curve-state grid (HW1F-style parallel level move): one Brownian shock per
  // (path, node) shifts the whole knot vector — the illustrative desk proxy (a real LGM adds mean reversion
  // + a term structure of vol; the kernel is indifferent). NODE-MAJOR layout: column (path p, node j) at
  // j*n_paths + p (the layout exposure.hpp/npv_grid assume). Node 0 (t=0) has sd=0 -> deterministic today.
  const int K = static_cast<int>(x_cal.size());
  Eigen::MatrixXd X(K, static_cast<Eigen::Index>(n_paths) * n_nodes);
  std::mt19937_64 rng(0x5E0Fu);
  std::normal_distribution<double> Z(0.0, 1.0);
  for (int j = 0; j < n_nodes; ++j) {
    const double t = node_time[j];
    // Ornstein-Uhlenbeck (HW1F) stationary variance Var(t)=σ²(1−e^{−2κt})/(2κ) — plateaus instead of
    // diverging like pure Brownian, so the curve-state (and the discount factors) stay in a plausible range.
    const double var = kappa > 1e-9 ? sigma * sigma * (1.0 - std::exp(-2.0 * kappa * t)) / (2.0 * kappa)
                                     : sigma * sigma * t;
    const double sd = std::sqrt(var);
    for (int p = 0; p < n_paths; ++p) {
      const Eigen::Index col = static_cast<Eigen::Index>(j) * n_paths + p;
      const double dz = sd > 0.0 ? sd * Z(rng) : 0.0;  // one parallel shock for all knots at this state
      for (int k = 0; k < K; ++k) X(k, col) = x_cal[k] + dz;
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  const Eigen::MatrixXd& npvg = cp.npv_grid(X);
  const xva::ExposureProfile pr = xva::exposure_profile(npvg, n_paths, n_nodes, node_time, pfe_q);
  const double wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  const double mtm = cp.total_npv(x_cal);

  json::object out;
  out["node_time"] = vecf(pr.node_time);
  out["epe"] = vecf(pr.epe);
  out["ene"] = vecf(pr.ene);
  out["pfe"] = vecf(pr.pfe);
  out["mtm"] = mtm;
  out["wall_us"] = wall_us;
  out["n_paths"] = n_paths;
  out["n_nodes"] = n_nodes;
  out["n"] = static_cast<int>(port.positions.size());
  return json::serialize(json::value(std::move(out)));
}

}  // namespace swaps::api
