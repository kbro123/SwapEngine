// Implementation of GenerateRisk (include/swaps/api/generate_risk.hpp): one book, N bundles, all risk off
// bundle[0]'s DFs, with under-determined bundles rank-completed by self-quoted null pillars. Reuses the
// BundleSession public API (calibrate / model_quote / price_portfolio / price_portfolio_risk / jacobian).
// Boost.JSON's implementation is compiled in bundle_api.cpp; this TU includes only the declarations.
#include "swaps/api/generate_risk.hpp"
#include "swaps/calibration/residual_engine.hpp"  // kRankThreshold

#include <algorithm>
#include <stdexcept>

#include <Eigen/Eigenvalues>
#include <boost/json.hpp>

namespace swaps::api {

namespace json = boost::json;
namespace pf = swaps::portfolio;

namespace {

}  // namespace

// Left-multiply the book's curve gradient g = dP/dx by the RANK-COMPLETED risk operator. J is the calibration
// Jacobian dq/dx (n_res x n_knots). Any null direction of JᵀJ (a knot the quotes cannot resolve) is self-
// quoted: appended to J as a unit-pinned row, so J_full is full column rank and M_full = (J_fullᵀJ_full)⁻¹
// J_fullᵀ is well-posed with NO curvature penalty. Returns the length-(n_res + n_null) ladder; the trailing
// entries are the synthetic pillars, and `syn_knot` gets each pillar's dominant knot index (for labelling).
Eigen::VectorXd null_completed_ladder(const Eigen::MatrixXd& J, const Eigen::VectorXd& g,
                                      std::vector<int>& syn_knot) {
  const int n_res = static_cast<int>(J.rows());
  const int nk = static_cast<int>(J.cols());
  syn_knot.clear();
  // A direction is "unseen by the quotes" when its SINGULAR VALUE is null at the engine's ONE rank threshold
  // (kRankThreshold, relative to sigma_max) -- the same test calibrate()'s rank_deficiency and the streamer's
  // operator use. Until 2026-09-10 this squared J and cut the normal-equation eigenvalues at 1e-9 of the
  // largest, i.e. singular values at 3e-5 of sigma_max: a stiff-but-constrained direction (sigma ~1e-5, e.g.
  // a long knot two instruments barely separate) was self-quoted as if the quotes could not see it (E3-G5).
  int rank = 0;
  Eigen::MatrixXd V = Eigen::MatrixXd::Identity(nk, nk);
  if (n_res > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullV);
    svd.setThreshold(cal::kRankThreshold);
    rank = static_cast<int>(svd.rank());
    V = svd.matrixV();  // columns rank.. are the null directions (singular values descending)
  }
  const int n_null = nk - rank;
  Eigen::MatrixXd Jf(n_res + n_null, nk);
  if (n_res) Jf.topRows(n_res) = J;
  for (int j = 0; j < n_null; ++j) {
    const Eigen::VectorXd v = V.col(rank + j);  // a unit null direction in knot space
    Jf.row(n_res + j) = v.transpose();          // self-quote it (a direct pin on that direction)
    int idx = 0;
    v.cwiseAbs().maxCoeff(&idx);                // the knot it loads on most -> its label
    syn_knot.push_back(idx);
  }
  // Jf has full column rank by construction; the pseudo-inverse (rank-safe at the same threshold) is
  // (JfᵀJf)⁻¹Jfᵀ without forming the normal matrix.
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
  cod.setThreshold(cal::kRankThreshold);
  cod.compute(Jf);
  const Eigen::MatrixXd Mf = cod.pseudoInverse();  // n_knots x (n_res + n_null)
  return Mf.transpose() * g;                       // length n_res + n_null
}

namespace {

// A light calibration floor so RE-LEVELING an under-determined bundle converges (the risk operator itself is
// NOT smoothed — it is rank-completed). Matches the web's "light" tension preset.
RegSpec calibration_reg(const RegSpec& reg, const cal::BundleProblem& b) {
  if (reg.on()) return reg;
  RegSpec c;
  c.tension = true;
  c.lambda = 0.02;
  c.sigma = 0.0;
  c.curves.resize(b.curves.size());
  for (std::size_t i = 0; i < b.curves.size(); ++i) c.curves[i] = static_cast<int>(i);
  return c;
}

json::array vec_to_json(const Eigen::VectorXd& v) {
  json::array a;
  a.reserve(static_cast<std::size_t>(v.size()));
  for (int i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}

}  // namespace

ConsistentRisk generate_risk(const pf::MultiCurveBook& book,
                             const std::vector<cal::BundleProblem>& bundles, const RegSpec& reg) {
  if (bundles.empty()) throw std::invalid_argument("generate_risk: need at least one bundle");

  // Anchor: calibrate bundle[0] to ITS market -> the curve C0 everything is priced on.
  BundleSession anchor(bundles[0]);
  anchor.calibrate(flat_x0(bundles[0]), calibration_reg(reg, bundles[0]));
  const PortfolioReprice p0 = anchor.price_portfolio(book);

  ConsistentRisk out;
  out.n = static_cast<int>(book.positions.size());
  out.npv = p0.npv;
  out.pv01 = p0.pv01;

  for (std::size_t k = 0; k < bundles.size(); ++k) {
    cal::BundleProblem bk = bundles[k];  // copy: we may overwrite its markets to re-level onto C0
    if (k > 0) {
      if (!anchor.same_curve_set(bk))
        throw std::invalid_argument(
            "generate_risk: every bundle must share bundle[0]'s curve set (same currencies / outright-or-"
            "spread, same order) so one book is valid on all and their forwards imply off the anchor.");
      // RE-LEVEL: each quote takes the value the anchor curve C0 implies -> bk now describes C0, not its own
      // saved market. (Skipped for the anchor itself.)
      for (auto& ins : bk.instruments) ins.market = anchor.model_quote(ins);
    }
    BundleSession sk(bk);
    sk.calibrate(flat_x0(bk), calibration_reg(reg, bk));  // for k>0 the markets are C0-implied -> curve ~ C0

    const PortfolioRisk pk = sk.price_portfolio_risk(book);  // gives curve_grad = dP/dx_k on C0
    const PortfolioReprice pr = sk.price_portfolio(book);
    const Eigen::MatrixXd J = sk.jacobian();

    ConsistentBundleRisk item;
    std::vector<int> syn;
    const Eigen::VectorXd full = null_completed_ladder(J, pk.curve_grad, syn);
    const int n_res = static_cast<int>(J.rows());
    item.n_residuals = n_res;
    item.n_synthetic = static_cast<int>(syn.size());
    item.ladder = full.head(n_res);
    item.synthetic = full.tail(full.size() - n_res);
    item.synthetic_knot = syn;
    item.npv = pr.npv;
    item.pv01 = pr.pv01;
    item.ladder_dv01 = 1e-4 * full.sum();
    out.bundles.push_back(std::move(item));
  }
  return out;
}

std::string generate_risk_json(const json::object& request) {
  const json::object& o = request;
  const json::object& g = o.contains("generate_risk") ? o.at("generate_risk").as_object() : o;

  if (!g.contains("book")) throw std::invalid_argument("generate_risk: missing 'book'");
  if (!g.contains("bundles") || !g.at("bundles").is_array() || g.at("bundles").as_array().empty())
    throw std::invalid_argument("generate_risk: 'bundles' must be a non-empty array");

  const pf::MultiCurveBook book = book_from_json(g.at("book"));
  std::vector<cal::BundleProblem> bundles;
  for (const auto& b : g.at("bundles").as_array()) bundles.push_back(bundle_from_json(b));

  const RegSpec reg = reg_from_json(g);  // the one decoder (swaps/api/codec.hpp)

  const ConsistentRisk cr = generate_risk(book, bundles, reg);

  json::object out;
  out["npv"] = cr.npv;
  out["pv01"] = cr.pv01;
  out["n"] = cr.n;
  json::array barr;
  for (const auto& b : cr.bundles) {
    json::object bo;
    bo["ladder"] = vec_to_json(b.ladder);
    bo["synthetic"] = vec_to_json(b.synthetic);
    json::array sk;
    for (int i : b.synthetic_knot) sk.push_back(i);
    bo["synthetic_knot"] = std::move(sk);
    bo["npv"] = b.npv;
    bo["pv01"] = b.pv01;
    bo["ladder_dv01"] = b.ladder_dv01;
    bo["n_residuals"] = b.n_residuals;
    bo["n_synthetic"] = b.n_synthetic;
    barr.push_back(std::move(bo));
  }
  out["bundles"] = std::move(barr);
  return json::serialize(json::value(std::move(out)));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string generate_risk_json(const std::string& request) { return generate_risk_json(json::parse(request).as_object()); }
}  // namespace swaps::api
