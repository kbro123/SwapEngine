// Exposure seam: the stateless "exposure" run_json verb — an EPE/ENE/PFE counterparty-exposure profile for a
// swap book off a CALIBRATED SOFR curve. Calibrates the bundle (like swaption_json), converts the book to the
// native single-curve Portfolio, simulates a Gaussian curve-state grid x(t,path)=x_cal+sigma·sqrt(t)·Z, and
// runs the MC-exposure kernel (CompiledPortfolio::npv_grid, ~1.9M full-book reprices/sec) + the EPE/ENE/PFE
// aggregation (swaps/xva/exposure.hpp). QuantLib-free.
//
// NETTING SETS: the legacy form ("book") treats the whole book as ONE netting set. The request may instead
// (or additionally) carry "netting_sets": typed trades grouped under a CSA (trade::NettingSet), where the
// CSA's collateral currency DECIDES the discount role (collateral OIS via trade::CSA) and the exposure
// profile is computed PER SET — netting aggregates within a set, never across sets. Same npv_grid kernel,
// same simulated state grid; only the AGGREGATION BOUNDARY moves. Schema (top-level or under "exposure"):
//   "value_date": "2026-09-04",                          required by netting_sets (trade materialization)
//   "curve_roles": {"USD-SOFR": 0},                      index id -> bundle curve index (as book_from_json)
//   "netting_sets": [{"id": "CP-1", "csa": {"collateral_currency": "USD"},
//                     "trades": [{id, notional, pay, fixed_rate, index, effective, maturity}, ...]}]
// Response gains "netting_sets": [{id, node_time, epe, ene, pfe, mtm, n}, ...]; the legacy whole-book
// fields are present (and unchanged) exactly when "book" is present.
//
// DEMO SCOPE: a single classic (Flat+Hermite) self-discounting curve; xccy / custom-region / turn'd curves
// and non-swap positions are rejected/skipped (CompiledPortfolio is single-curve). The Gaussian curve-state
// proxy is illustrative — a calibrated LGM/HW1F is a later step; the kernel is model-agnostic (pure
// f(exp(-W·x))).
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/exposure.hpp"
#include "swaps/build/date.hpp"
#include "swaps/calibration/pnl_explain.hpp"  // roll_book: age the book to each exposure node
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/trade/book.hpp"
#include "swaps/trade/csa.hpp"
#include "swaps/trade/trade.hpp"
#include "swaps/xva/exposure.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace pf = swaps::portfolio;
namespace cal = swaps::calibration;
namespace xva = swaps::xva;
namespace tr = swaps::trade;
namespace bld = swaps::build;

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
std::string js(const json::object& o, const char* k, const char* d) {
  return o.contains(k) && o.at(k).is_string() ? std::string(o.at(k).as_string()) : std::string(d);
}
json::array vecf(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}

// The single-curve filter shared by the legacy whole-book path and the per-netting-set path: keep only
// swap-kind positions priced entirely off curve 0 (CompiledPortfolio is single self-discounting curve).
pf::Portfolio to_single_curve_portfolio(const pf::MultiCurveBook& book) {
  pf::Portfolio port;
  for (const auto& b : book.positions) {
    if (b.kind != pf::MultiCurveBook::Kind::Swap) continue;
    if (b.fwd_curve != 0 || b.disc_curve != 0 || b.fixed_curve != 0) continue;
    port.positions.push_back({b.float_coupons, b.fixed_coupons, b.fixed_rate, b.notional});
  }
  return port;
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
  const bool has_book = o.contains("book");
  const bool has_sets = o.contains("netting_sets") && o.at("netting_sets").is_array();
  if (!has_book && !has_sets) throw std::invalid_argument("exposure: missing 'book'");

  // Calibrate the bundle -> the discount curve we reprice against (same recipe as swaption_json).
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  for (int c = 0; c < static_cast<int>(sess.problem().curves.size()); ++c) reg.curves.push_back(c);
  sess.calibrate(flat_x0(sess.problem()), reg);

  // Guard: the CompiledPortfolio exposure kernel is a single classic Flat+Hermite self-discounting curve.
  // (Checked via modules() — regions ARE the one curve representation now; "classic" = the two-region
  // Flat->Hermite layout.)
  const auto& C0 = sess.problem().curves[0];
  const auto mods = C0.modules();
  const bool classic = mods.size() == 2 && mods[0].scheme == swaps::curve::Scheme::Flat &&
                       mods[1].scheme == swaps::curve::Scheme::Hermite;
  if (sess.problem().n_curves() != 1 || !classic || !C0.turns.empty())
    throw std::invalid_argument("exposure: demo restricted to a single classic Flat+Hermite self-discounting "
                                "curve (no custom regions/turns/xccy)");
  const Eigen::VectorXd x_cal = sess.x();  // node-0 center; length = meeting+back knots

  // Legacy whole-book path (behaviour unchanged): book -> native single-curve Portfolio (a straight
  // coupon-field copy; swap-kind, all-curve-0 positions). The whole book is one netting set.
  pf::MultiCurveBook legacy_book;
  if (has_book) {
    legacy_book = book_from_json(o.at("book"));
    if (to_single_curve_portfolio(legacy_book).positions.empty())
      throw std::invalid_argument("exposure: no single-curve swap positions in the book");
  }

  // Netting-set path: each set = trade::NettingSet (trades under one CSA), materialized via
  // NettingSet::to_book(vd, csa_role) so the CSA's collateral-currency OIS DECIDES the discount role.
  // Role resolution mirrors book_from_json ("curve_roles"; an unmapped index fails loudly).
  struct ParsedSet {
    std::string id;
    pf::MultiCurveBook book;  // unrolled: aged to every node below
    int n = 0;
  };
  std::vector<ParsedSet> sets;
  if (has_sets) {
    const json::array& sarr = o.at("netting_sets").as_array();
    if (sarr.empty() && !has_book)
      throw std::invalid_argument("exposure: 'netting_sets' is empty");
    if (!sarr.empty() && !o.contains("value_date"))
      throw std::invalid_argument("exposure: netting_sets require a 'value_date' (ISO)");
    std::map<std::string, int> roles;
    if (o.contains("curve_roles"))
      for (const auto& kv : o.at("curve_roles").as_object())
        roles[std::string(kv.key())] = static_cast<int>(kv.value().as_int64());
    const auto role_of = [&](const std::string& id, const char* what) -> int {
      const auto it = roles.find(id);
      if (it == roles.end())
        throw std::invalid_argument(std::string("exposure: curve_roles has no entry for ") + what +
                                    " '" + id + "'");
      return it->second;
    };
    const bld::Date vd = sarr.empty() ? bld::Date{} : bld::Date::from_iso(js(o, "value_date", ""));
    for (const auto& se : sarr) {
      const auto& so = se.as_object();
      if (!so.contains("csa") || !so.at("csa").is_object())
        throw std::invalid_argument("exposure: every netting set needs a 'csa'");
      const tr::CSA csa = tr::CSA::cash(js(so.at("csa").as_object(), "collateral_currency", ""));
      const std::string disc_id = csa.discount_index_id();
      if (disc_id.empty())
        throw std::invalid_argument("exposure: netting-set CSA has an unknown collateral currency");
      const int csa_role = role_of(disc_id, "discount index");
      tr::NettingSet ns(js(so, "id", ""), csa);
      if (so.contains("trades") && so.at("trades").is_array())
        for (const auto& te : so.at("trades").as_array()) {
          const auto& to = te.as_object();
          const std::string index = js(to, "index", "");
          if (index.empty())
            throw std::invalid_argument("exposure: every netting-set trade needs an 'index'");
          ns.add(tr::Trade::vanilla_swap(
              js(to, "id", ""), jd(to, "notional", 1.0),
              js(to, "pay", "fixed") == "float" ? tr::Pay::Float : tr::Pay::Fixed,
              jd(to, "fixed_rate", 0.0), js(to, "currency", ""), index,
              bld::Date::from_iso(js(to, "effective", "")), bld::Date::from_iso(js(to, "maturity", "")),
              role_of(index, "trade index"), csa_role));
        }
      // THE CSA DECIDES: every trade's discounting is forced onto the collateral currency's OIS role.
      pf::MultiCurveBook book = ns.to_book(vd, csa_role);
      const int n = static_cast<int>(to_single_curve_portfolio(book).positions.size());
      if (n == 0)
        throw std::invalid_argument("exposure: netting set '" + ns.id() +
                                    "' has no single-curve swap positions");
      sets.push_back({ns.id(), std::move(book), n});
    }
  }

  // Node times 0..horizon (node 0 = today: sqrt(0)=0 -> deterministic, EPE(0)=max(MtM,0)).
  std::vector<double> node_time(n_nodes);
  for (int j = 0; j < n_nodes; ++j) node_time[j] = horizon * static_cast<double>(j) / (n_nodes - 1);

  // Single-factor Gaussian curve-state grid (HW1F-style parallel level move): one Brownian shock per
  // (path, node) shifts the whole knot vector — the illustrative desk proxy (a real LGM adds mean reversion
  // + a term structure of vol; the kernel is indifferent). NODE-MAJOR layout: column (path p, node j) at
  // j*n_paths + p (the layout exposure.hpp/npv_grid assume). Node 0 (t=0) has sd=0 -> deterministic today.
  // ONE grid serves every netting set: the curve states are market scenarios, shared across counterparties.
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

  // The kernel + aggregation, PER NETTING SET (the legacy whole book is just the first "set" when present):
  // exposure aggregates WITHIN a set — each portfolio gets its own colwise-net + EPE/ENE/PFE reduction —
  // never across sets. Same npv_grid kernel; only the aggregation boundary moved.
  // THE BOOK IS AGED TO EVERY NODE (E3-F2, 2026-09-10): at node t_j the surviving cashflows are the ones
  // paying after t_j, re-timed from the node (roll_book, the same helper pnl_explain's roll leg uses), and
  // priced off the node's simulated curve states. Until now every node repriced TODAY's cashflow set, so a
  // 2y swap still showed EPE/PFE at 4..10y and CVA carried the whole unmatured tail. A netting set whose
  // book has fully matured at a node contributes zero there. The per-node compile sits inside the timed
  // section (it is part of the exposure computation now).
  const auto t0 = std::chrono::steady_clock::now();
  const auto profile_of = [&](const pf::MultiCurveBook& book, double& mtm) {
    Eigen::MatrixXd net(1, static_cast<Eigen::Index>(n_paths) * n_nodes);  // the netted value per state
    for (int j = 0; j < n_nodes; ++j) {
      const pf::Portfolio port = to_single_curve_portfolio(cal::roll_book(book, node_time[j], /*shift=*/true));
      const Eigen::Index c0 = static_cast<Eigen::Index>(j) * n_paths;
      if (port.positions.empty()) {
        net.middleCols(c0, n_paths).setZero();
        continue;
      }
      const pf::CompiledPortfolio cp(mods, port);
      const Eigen::MatrixXd Xj = X.middleCols(c0, n_paths);
      net.middleCols(c0, n_paths) = cp.npv_grid(Xj).colwise().sum();
    }
    mtm = pf::CompiledPortfolio(mods, to_single_curve_portfolio(book)).total_npv(x_cal);
    return xva::exposure_profile(net, n_paths, n_nodes, node_time, pfe_q);
  };
  json::object out;
  if (has_book) {
    double mtm = 0.0;
    const xva::ExposureProfile pr = profile_of(legacy_book, mtm);
    out["node_time"] = vecf(pr.node_time);
    out["epe"] = vecf(pr.epe);
    out["ene"] = vecf(pr.ene);
    out["pfe"] = vecf(pr.pfe);
    out["mtm"] = mtm;
  }
  json::array sets_out;
  for (const auto& s : sets) {
    double mtm = 0.0;
    const xva::ExposureProfile pr = profile_of(s.book, mtm);
    json::object so;
    so["id"] = s.id;
    so["node_time"] = vecf(pr.node_time);
    so["epe"] = vecf(pr.epe);
    so["ene"] = vecf(pr.ene);
    so["pfe"] = vecf(pr.pfe);
    so["mtm"] = mtm;
    so["n"] = s.n;
    sets_out.push_back(std::move(so));
  }
  const double wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

  out["wall_us"] = wall_us;
  out["n_paths"] = n_paths;
  out["n_nodes"] = n_nodes;
  if (has_book) out["n"] = static_cast<int>(to_single_curve_portfolio(legacy_book).positions.size());
  if (has_sets) out["netting_sets"] = std::move(sets_out);
  return json::serialize(json::value(std::move(out)));
}

}  // namespace swaps::api
