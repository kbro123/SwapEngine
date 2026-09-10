// Calibration diagnostics: the stateless "calib_report" run_json verb (audit quick-win #10). Compiles +
// calibrates a bundle exactly as the `bundle` path does (BundleSession), then computes from the calibration
// Jacobian J = dq/dx at the solution a Jacobian condition number, its singular spectrum, the engine's own
// rank deficiency, and a per-quote IDENTIFIABILITY (how well-determined each pillar is). It also echoes the
// existing per-quote in-band diagnostics, so this one verb is a self-contained calibration-health report.
//
// This file is standalone (mirrors api/inflation.cpp / api/sabr_calibrate): it adds NO method to
// BundleSession and reuses only the session's PUBLIC surface -- jacobian(), risk_operator(),
// quote_diagnostics_json(), result() -- so it stays disjoint from the shared calibration internals.
//
// Measures (documented, defensible):
//   condition_number = sigma_max / sigma_min of J (Eigen JacobiSVD over the n_residuals x n_knots Jacobian).
//     Always >= 1 (sigma_max is the largest singular value). A well-posed curve gives a modest value; a
//     redundant/collinear instrument drives sigma_min toward 0 and the ratio up. If sigma_min is exactly 0
//     the ratio is reported as the largest finite double (rank_deficiency then flags WHY it blew up).
//   rank_deficiency = CalibrationResult::rank_deficiency -- the engine's own count of numerically
//     unconstrained state directions (n_knots - rank(J) at the shared kRankThreshold), reused verbatim.
//   identifiability_i = clamp(H(i,i), 0, 1), H = J*M the model-resolution (hat) matrix, M = risk_operator =
//     (JᵀJ + RᵀR)⁻¹Jᵀ = dx/dq. h_ii is quote i's leverage on its own fitted value: ~1 when instrument i
//     independently pins its pillar, ->0 when its information is redundant (a near-duplicate instrument
//     splits the leverage with the pillar it collides with, so both fall). trace(H) = rank(J).
//
// Additive, QuantLib-free, no new deps (Eigen already vendored). Units are model decimals (0.025 = 2.5%).
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/calib_report.hpp"
#include "swaps/api/json_util.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace cal = swaps::calibration;

namespace {
}  // namespace

std::string calib_report_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o = top.contains("calib_report") && top.at("calib_report").is_object()
                              ? top.at("calib_report").as_object()
                              : top;
  if (!o.contains("bundle")) throw std::invalid_argument("calib_report: missing 'bundle' object");

  // Build + calibrate exactly as the `bundle` verb does (same helpers, same seed/reg semantics).
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  if (prob.n_curves() == 0) throw std::invalid_argument("calib_report: bundle has no curves");
  BundleSession sess(std::move(prob));
  const cal::BundleProblem& P = sess.problem();

  Eigen::VectorXd x0;
  if (o.contains("x0") && o.at("x0").is_array()) {
    const auto& xa = o.at("x0").as_array();
    x0.resize(static_cast<Eigen::Index>(xa.size()));
    for (std::size_t i = 0; i < xa.size(); ++i) x0[static_cast<Eigen::Index>(i)] = xa[i].to_number<double>();
    if (x0.size() != P.n_knots())
      throw std::invalid_argument("calib_report: x0 length does not match the bundle's knot count");
  } else {
    x0 = flat_x0(P);
  }

  RegSpec reg;
  if (o.contains("regularize") && o.at("regularize").is_object()) {
    const auto& r = o.at("regularize").as_object();
    reg.lambda = jd(r, "lambda", 0.0);
    reg.curves = jia(r, "curves");
    reg.tension = jb(r, "tension", false);
    reg.sigma = jd(r, "sigma", 0.0);
  }

  const cal::CalibrationResult& res = sess.calibrate(x0, reg);

  // The calibration Jacobian at the solution and the model-resolution (hat) matrix H = J*M (M = dx/dq).
  const Eigen::MatrixXd J = sess.jacobian(reg);        // n_res x n_knots
  const Eigen::MatrixXd M = sess.risk_operator(reg);   // n_knots x n_res
  const Eigen::Index n_res = J.rows();
  const Eigen::Index n_knots = J.cols();

  // Singular spectrum of J and the condition number sigma_max / sigma_min (always >= 1).
  Eigen::VectorXd sv;
  double cond = 1.0;
  if (n_res > 0 && n_knots > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);          // singular values only, descending order
    sv = svd.singularValues();
    if (sv.size() > 0) {
      const double smax = sv[0];
      const double smin = sv[sv.size() - 1];
      cond = smin > 0.0 ? smax / smin : std::numeric_limits<double>::max();
      if (!std::isfinite(cond)) cond = std::numeric_limits<double>::max();
      if (cond < 1.0) cond = 1.0;  // guard tiny FP noise; sigma_max >= sigma_min by construction
    }
  }

  // Per-quote identifiability = clamp(diag(H), 0, 1), echoed alongside the in-band quote diagnostics so a
  // soft (banded) fit is never silent. quote_diagnostics_json() is one entry per instrument, in residual order.
  const json::value qd = json::parse(sess.quote_diagnostics_json());
  const json::array& diags = qd.as_array();
  json::array quotes;
  quotes.reserve(diags.size());
  for (std::size_t i = 0; i < diags.size(); ++i) {
    const json::object& d = diags[i].as_object();
    double ident = 0.0;
    const Eigen::Index ii = static_cast<Eigen::Index>(i);
    if (ii < n_res && ii < M.cols()) {
      ident = J.row(ii).dot(M.col(ii));   // (J*M)(i,i) without forming the full n_res x n_res product
      ident = std::clamp(ident, 0.0, 1.0);
    }
    json::object q;
    q["model"] = d.contains("model") ? d.at("model").to_number<double>() : 0.0;
    q["target"] = d.contains("target") ? d.at("target").to_number<double>() : 0.0;
    q["residual"] = d.contains("residual") ? d.at("residual").to_number<double>() : 0.0;
    q["weight"] = d.contains("weight") ? d.at("weight").to_number<double>() : 1.0;
    q["in_band"] = d.contains("in_band") && d.at("in_band").is_bool() ? d.at("in_band").as_bool() : false;
    q["soft"] = d.contains("soft") && d.at("soft").is_bool() ? d.at("soft").as_bool() : false;
    q["identifiability"] = ident;
    quotes.push_back(std::move(q));
  }

  json::object out;
  out["rms_residual"] = res.rms_residual;
  out["converged"] = res.converged;
  out["status"] = res.status;
  out["rank_deficiency"] = res.rank_deficiency;
  out["condition_number"] = cond;
  out["singular_values"] = vecf(sv);
  out["quotes"] = std::move(quotes);
  out["n"] = static_cast<int>(diags.size());
  return json::serialize(json::value(std::move(out)));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string calib_report_json(const std::string& request) { return calib_report_json(json::parse(request).as_object()); }
}  // namespace swaps::api
