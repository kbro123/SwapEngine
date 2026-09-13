// P&L EXPLAIN seam: the stateless "pnl" run_json verb (see api/pnl.hpp for the schema and
// include/swaps/calibration/pnl_explain.hpp for the exact carry/roll/market/residual contract).
//
// The verb wires existing primitives, nothing new:
//   * bundle0 -> a BundleSession, calibrated to q0 -> the state x0 and, via price_portfolio_risk(book),
//     the analytic delta ladder dP/dq at (x0, q0) — the SAME risk_operator / bucketed_delta path the
//     web risk feature uses (no bumping).
//   * bundle1 (same structure) -> calibrated to q1 -> the horizon state x1; dq = q1 - q0 from the two
//     bundles' market() targets.
//   * calibration::pnl_explain(prob0, book, x0, x1, dt, ladder, dq) does the reprices + attribution.
// Components SUM to total by construction (residual closes it). QuantLib-free.
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/pnl.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/calibration/pnl_explain.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace cal = swaps::calibration;
namespace pf = swaps::portfolio;

namespace {
Eigen::VectorXd to_vec(const std::vector<double>& v) {
  return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
}
}  // namespace

std::string pnl_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o =
      top.contains("pnl") && top.at("pnl").is_object() ? top.at("pnl").as_object() : top;

  if (!o.contains("bundle0") || !o.at("bundle0").is_object())
    throw std::invalid_argument("pnl: missing 'bundle0' object");
  if (!o.contains("book") || !o.at("book").is_object())
    throw std::invalid_argument("pnl: missing 'book' object");

  const double dt = jd(o, "dt_years", 0.0);
  const RegSpec reg = reg_from_json(o);
  const pf::MultiCurveBook book = book_from_json(o.at("book"));

  // bundle0: calibrate to q0 -> x0, and the analytic delta ladder dP/dq at (x0, q0).
  cal::BundleProblem prob0 = bundle_from_json(o.at("bundle0"));
  if (prob0.n_curves() == 0) throw std::invalid_argument("pnl: bundle0 has no curves");
  BundleSession sess0(std::move(prob0));
  const cal::BundleProblem& P0 = sess0.problem();
  {
    const std::vector<double> seed = darr(o, "x0");
    sess0.calibrate(static_cast<int>(seed.size()) == P0.n_knots() ? to_vec(seed) : flat_x0(P0), reg);
  }
  const PortfolioRisk risk = sess0.price_portfolio_risk(book, reg);  // ladder = dP/dq at (x0, q0)
  const Eigen::VectorXd q0 = P0.market();

  // bundle1 (optional, same structure): calibrate to q1 -> x1; dq = q1 - q0. Absent => pure carry/roll.
  Eigen::VectorXd x1 = sess0.x();
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(q0.size());
  if (o.contains("bundle1") && o.at("bundle1").is_object()) {
    cal::BundleProblem prob1 = bundle_from_json(o.at("bundle1"));
    if (prob1.n_knots() != P0.n_knots())
      throw std::invalid_argument("pnl: bundle1 must share bundle0's curve topology (knot count differs)");
    if (prob1.n_residuals() != P0.n_residuals())
      throw std::invalid_argument("pnl: bundle1 must share bundle0's instruments (residual count differs)");
    BundleSession sess1(std::move(prob1));
    const std::vector<double> seed1 = darr(o, "x1");
    sess1.calibrate(static_cast<int>(seed1.size()) == sess1.problem().n_knots()
                        ? to_vec(seed1) : flat_x0(sess1.problem()), reg);
    x1 = sess1.x();
    dq = sess1.problem().market() - q0;
  }

  // Optional explicit-state overrides for the decomposition reprice (advanced; the ladder stays at x0).
  {
    const std::vector<double> ox0 = darr(o, "x0"), ox1 = darr(o, "x1");
    Eigen::VectorXd x0 = sess0.x();
    if (static_cast<int>(ox0.size()) == P0.n_knots()) x0 = to_vec(ox0);
    if (static_cast<int>(ox1.size()) == P0.n_knots()) x1 = to_vec(ox1);

    const cal::PnlExplain e = cal::pnl_explain(P0, book, x0, x1, dt, risk.ladder, dq);

    json::object out;
    out["total"] = e.total;
    out["carry"] = e.carry;
    out["roll"] = e.roll;
    out["market"] = e.market;
    out["residual"] = e.residual;
    out["npv_t0"] = e.npv_t0;
    out["npv_t1"] = e.npv_t1;
    out["market_ladder"] = vecf(e.market_ladder);
    out["dq"] = vecf(dq);
    out["dt_years"] = dt;
    out["n"] = static_cast<int>(book.positions.size());
    json::object resp;
    resp["pnl"] = std::move(out);
    return json::serialize(resp);
  }
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string pnl_json(const std::string& request) { return pnl_json(json::parse(request).as_object()); }
}  // namespace swaps::api
