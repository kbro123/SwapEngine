// Full-revaluation VaR / Expected-Shortfall verb (declared in include/swaps/api/var.hpp). Two modes:
//   REVAL    — calibrate a bundle ONCE, then fork the fitted market per historical/supplied MOVE
//              (market::Scenario) and reprice the book through the CACHED compiled twin
//              (portfolio::CompiledMultiCurveBook, the reprice_bound kernel) to build the P&L distribution.
//              This is full revaluation (every move genuinely re-prices), not a delta/Taylor approximation,
//              and it is cheap because the anchor is calibrated once and each cell is a µs fork + matvec.
//   SUPPLIED — quantile a P&L series handed in directly (an external full-reval or a realized-P&L history).
// Both then reduce the P&L to VaR and Expected-Shortfall at the requested confidence levels.
//
// The per-move fork is IDENTICAL to api/scenario.cpp / api/scenario_grid.cpp (curve shift added to each
// curve's interp forwards; an FX bump compounded into a scale on every xccy fx_spot), so a reval VaR built
// from parallel-shift moves matches the corresponding `scenario` npv_deltas to rounding.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/var.hpp"

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/portfolio/compiled_multi.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace cal = swaps::calibration;
namespace pf = swaps::portfolio;

namespace {


// One market move -> per-curve rate shift (bp/1e4) + a compounded fx factor. Same arithmetic as
// api/scenario.cpp: parallel_bp shifts every curve, shift_curve{role:bp} shifts one, bump_fx compounds.
void parse_move(const json::object& m, int n_curves, std::vector<double>& curve_delta, double& fx_factor) {
  curve_delta.assign(n_curves, 0.0);
  fx_factor = 1.0;
  if (m.contains("parallel_bp") && !m.at("parallel_bp").is_null()) {
    const double d = m.at("parallel_bp").to_number<double>() / 1e4;
    for (double& x : curve_delta) x += d;
  }
  if (m.contains("shift_curve") && m.at("shift_curve").is_object())
    for (const auto& kv : m.at("shift_curve").as_object()) {
      int role = 0;
      try {
        std::size_t pos = 0;
        role = std::stoi(std::string(kv.key()), &pos);
        if (pos != std::string(kv.key()).size()) throw std::invalid_argument("trailing");
      } catch (const std::exception&) {
        throw std::invalid_argument("var: shift_curve key '" + std::string(kv.key()) +
                                    "' is not an integer curve role");
      }
      if (role < 0 || role >= n_curves)
        throw std::invalid_argument("var: shift_curve role " + std::to_string(role) + " out of range");
      curve_delta[static_cast<std::size_t>(role)] += kv.value().to_number<double>() / 1e4;
    }
  if (m.contains("bump_fx") && m.at("bump_fx").is_array())
    for (const auto& fe : m.at("bump_fx").as_array())
      fx_factor *= (1.0 + jd(fe.as_object(), "rel", 0.0));
}

// numpy "type 7" linear-interpolation quantile of a SORTED-ascending vector at probability p in [0,1].
double quantile_sorted(const std::vector<double>& s, double p) {
  const int n = static_cast<int>(s.size());
  if (n == 1) return s[0];
  const double pos = p * (n - 1);
  const double lo = std::floor(pos);
  const int i = static_cast<int>(lo);
  if (i >= n - 1) return s[n - 1];
  const double frac = pos - lo;
  return s[i] * (1.0 - frac) + s[i + 1] * frac;
}

// VaR (loss) + ES (loss) at confidence q from a sorted-ascending P&L vector. Loss = −P&L; the (1−q)
// lower-tail P&L quantile is the negated VaR; ES = −mean of the m worst P&L order stats (m = round((1−q)N),
// >=1). Returns {var_pnl (signed), es_pnl (signed)}; the caller negates for the loss figures.
std::pair<double, double> var_es_pnl(const std::vector<double>& s, double q) {
  const int n = static_cast<int>(s.size());
  const double p = 1.0 - q;
  const double var_pnl = quantile_sorted(s, p);
  int m = static_cast<int>(std::lround(p * n));
  if (m < 1) m = 1;
  if (m > n) m = n;
  double sum = 0.0;
  for (int i = 0; i < m; ++i) sum += s[i];  // the m smallest (worst) P&Ls
  return {var_pnl, sum / m};
}

}  // namespace

std::string var_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o =
      (top.contains("var") && top.at("var").is_object()) ? top.at("var").as_object() : top;

  std::vector<double> quantiles = darr(o, "quantiles");
  if (quantiles.empty()) quantiles = {0.95, 0.99};
  for (double q : quantiles)
    if (!(q > 0.0 && q < 1.0)) throw std::invalid_argument("var: every quantile must be in (0, 1)");

  json::object out;
  std::vector<double> pnl;
  double reval_us = 0.0;

  // ---- SUPPLIED mode: a P&L series handed in directly --------------------------------------------
  if (o.contains("pnl") && o.at("pnl").is_array()) {
    pnl = darr(o, "pnl");
    if (pnl.empty()) throw std::invalid_argument("var: 'pnl' array is empty");
    out["mode"] = "supplied";
  } else {
    // ---- REVAL mode: calibrate once, fork+reprice the book under each move ------------------------
    if (!o.contains("bundle")) throw std::invalid_argument("var: needs either 'pnl' or 'bundle'+'scenarios'");
    if (!o.contains("scenarios") || !o.at("scenarios").is_array())
      throw std::invalid_argument("var: reval mode needs a 'scenarios' array of market moves");
    if (!o.contains("book")) throw std::invalid_argument("var: reval mode needs a 'book' to reprice");

    cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
    if (prob.n_curves() == 0) throw std::invalid_argument("var: bundle has no curves");
    BundleSession sess(std::move(prob));
    const cal::BundleProblem& P = sess.problem();

    Eigen::VectorXd x0;
    if (o.contains("x0")) {
      const std::vector<double> xv = darr(o, "x0");
      if (static_cast<int>(xv.size()) != P.n_knots())
        throw std::invalid_argument("var: x0 length does not match the bundle's knot count");
      x0 = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<Eigen::Index>(xv.size()));
    } else {
      x0 = flat_x0(P);
    }
    RegSpec reg;
    if (o.contains("regularize") && o.at("regularize").is_object()) {
      const auto& r = o.at("regularize").as_object();
      reg.lambda = jd(r, "lambda", 0.0);
      if (r.contains("curves") && r.at("curves").is_array())
        for (const auto& e : r.at("curves").as_array())
          reg.curves.push_back(static_cast<int>(e.to_number<long long>()));
      reg.tension = r.contains("tension") && r.at("tension").as_bool();
      reg.sigma = jd(r, "sigma", 0.0);
    }
    sess.calibrate(x0, reg);
    const Eigen::VectorXd x_base = sess.x();

    pf::MultiCurveBook book = book_from_json(o.at("book"));

    std::map<long long, std::unique_ptr<pf::CompiledMultiCurveBook>> cbook_by_factor;
    const auto factor_key = [](double f) { return static_cast<long long>(std::llround(f * 1e12)); };
    auto cbook_for = [&](double factor) -> const pf::CompiledMultiCurveBook& {
      const long long key = factor_key(factor);
      auto it = cbook_by_factor.find(key);
      if (it != cbook_by_factor.end()) return *it->second;
      if (factor == 1.0)
        return *(cbook_by_factor[key] = std::make_unique<pf::CompiledMultiCurveBook>(P.curves, book));
      pf::MultiCurveBook sbook = book;
      for (auto& p : sbook.positions)
        if (p.kind == pf::MultiCurveBook::Kind::Xccy) p.fx_spot *= factor;
      return *(cbook_by_factor[key] = std::make_unique<pf::CompiledMultiCurveBook>(P.curves, sbook));
    };
    const double base_npv = cbook_for(1.0).npv(x_base);

    const json::array& moves = o.at("scenarios").as_array();
    if (moves.empty()) throw std::invalid_argument("var: 'scenarios' is empty");
    pnl.reserve(moves.size());
    std::vector<double> curve_delta;
    double fx_factor = 1.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& me : moves) {
      parse_move(me.as_object(), P.n_curves(), curve_delta, fx_factor);
      Eigen::VectorXd xs = x_base;
      for (int c = 0; c < P.n_curves(); ++c) {
        const double d = curve_delta[static_cast<std::size_t>(c)];
        if (d == 0.0) continue;
        const int oc = P.offset(c);
        const int ni = P.curves[c].n_interp_knots();
        xs.segment(oc, ni).array() += d;
      }
      pnl.push_back(cbook_for(fx_factor).npv(xs) - base_npv);
    }
    reval_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

    out["mode"] = "reval";
    out["base_npv"] = base_npv;
    out["n_positions"] = static_cast<int>(book.positions.size());
    out["reval_us"] = reval_us;
  }

  // ---- reduce the P&L distribution to VaR / ES ---------------------------------------------------
  const int n = static_cast<int>(pnl.size());
  double mean = 0.0;
  for (double v : pnl) mean += v;
  mean /= n;
  double var_acc = 0.0;
  for (double v : pnl) var_acc += (v - mean) * (v - mean);
  const double stdev = n > 1 ? std::sqrt(var_acc / (n - 1)) : 0.0;

  std::vector<double> sorted = pnl;
  std::sort(sorted.begin(), sorted.end());

  out["n"] = n;
  out["mean_pnl"] = mean;
  out["stdev_pnl"] = stdev;
  out["pnl_sorted"] = vecf(sorted);

  json::array qout;
  for (double q : quantiles) {
    const auto [var_pnl, es_pnl] = var_es_pnl(sorted, q);
    json::object qo;
    qo["q"] = q;
    qo["var"] = -var_pnl;   // loss (positive)
    qo["es"] = -es_pnl;     // loss (positive)
    qo["var_pnl"] = var_pnl;
    qo["es_pnl"] = es_pnl;
    qout.push_back(std::move(qo));
  }
  out["quantiles"] = std::move(qout);

  json::object resp;
  resp["var"] = std::move(out);
  return json::serialize(resp);
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string var_json(const std::string& request) { return var_json(json::parse(request).as_object()); }
}  // namespace swaps::api
