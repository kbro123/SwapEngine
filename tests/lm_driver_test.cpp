// E5 taxonomy: T3 cross-path parity (the shipped LM driver against Eigen's MINPACK LM on the same analytic engine) | T2 calibration (status semantics, rank path)
// THE SHIPPED LM IS lm_normal_equations (2026-10-19, ASSUMPTIONS K50): Levenberg-Marquardt on the scaled normal
// equations with a Cholesky per damping trial, where calibrate_with ran Eigen's NonLinearOptimization LM (MINPACK
// lmder: a column-pivoted QR of J per iteration plus lmpar2's Givens trials -- 4.0 ms of a 5.9 ms iteration on the G4
// desk against 1.8 ms for the Jacobian). Pinned: on every ladder rung and the G4 Hermite desk the two drivers end at the
// same stationary point (equal in quote space to 1e-10, the same residual norm to 1e-12 relative, 1e-9 on the state off the G4 desk's flat CCP valley); the status codes keep MINPACK's
// meaning; m < n is refused as improper input so the rank path takes over, as before.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>

#include "g4_market.hpp"
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
using swaps::shapes::Shape;

namespace {
// Eigen's MINPACK LM over the same engine: the retired driver, kept here as the parity reference.
struct EngineFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
  const cal::HybridBundleResidual* eng;
  int n, m;
  int inputs() const { return n; }
  int values() const { return m; }
  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const { fvec = eng->residuals(x); return 0; }
  int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const { fjac = eng->jacobian(x); return 0; }
};
struct Fit { Eigen::VectorXd x; double fnorm; int info, iters; };
Fit minpack(const cal::HybridBundleResidual& h, int n, int m, const Eigen::VectorXd& x0) {
  EngineFunctor f{&h, n, m};
  Eigen::LevenbergMarquardt<EngineFunctor> lm(f);
  lm.parameters.xtol = 1e-14; lm.parameters.ftol = 1e-14; lm.parameters.maxfev = 4000;
  Fit r; r.x = x0; r.info = lm.minimize(r.x); r.iters = static_cast<int>(lm.iter); r.fnorm = h.residuals(r.x).norm();
  return r;
}
Fit ours(const cal::HybridBundleResidual& h, int n, int m, const Eigen::VectorXd& x0) {
  Fit r; r.x = x0; r.info = cal::lm_normal_equations(h, n, m, r.x, r.iters); r.fnorm = h.residuals(r.x).norm();
  return r;
}
bool converged_code(int info) { return info >= 1 && info <= 8 && info != 5; }
}  // namespace

TEST(LmDriver, EndsWhereMinpackEndsOnEveryRungAndTheG4Desk) {
  std::vector<Shape> shapes = swaps::shapes::ladder();
  swaps::shapes::g4::Opts o; o.back = cv::Scheme::Hermite;
  shapes.push_back(swaps::shapes::g4_desk(o));
  for (const Shape& s : shapes) {
    cal::BundleProblem p = s.prob;
    cal::add_smoothing_rows(p);  // the rows the session streams, where the region says so (the G4 desk)
    const cal::HybridBundleResidual h(p);
    const int n = p.n_knots(), m = p.n_residuals();
    // cold from the seed, and warm from the cold answer after the 25 bp move
    const Fit a = ours(h, n, m, s.x0), b = minpack(h, n, m, s.x0);
    EXPECT_TRUE(converged_code(a.info)) << s.name << " ours: " << cal::lm_status_text(a.info);
    EXPECT_TRUE(converged_code(b.info)) << s.name << " minpack: " << cal::lm_status_text(b.info);
    // Agreement is judged where the quotes see it: J (x_a - x_b) in residual space, the residual norm and the
    // stationarity. The G4 desk's CCP discount blocks are a flat valley (cond(J) 2e5: a 20 bp discount spread moves a par
    // rate 0.5 bp), so two drivers stopping on a 1e-14 scaled step end up to 4e-7 apart along it with the same objective
    // to 1e-12; on the well-conditioned rungs the states agree to 1e-9 as well.
    const auto same_point = [&](const cal::HybridBundleResidual& e, const Fit& p1, const Fit& p2, const char* what) {
      const Eigen::MatrixXd J = e.jacobian(p2.x);
      EXPECT_LT((J * (p1.x - p2.x)).cwiseAbs().maxCoeff(), 1e-10) << s.name << " " << what << ": the two answers differ in quote space";
      EXPECT_NEAR(p1.fnorm, p2.fnorm, 1e-12 * (1.0 + p2.fnorm)) << s.name << " " << what << " residual norm";
      const double st1 = (J.transpose() * e.residuals(p1.x)).cwiseAbs().maxCoeff();
      const double st2 = (J.transpose() * e.residuals(p2.x)).cwiseAbs().maxCoeff();
      EXPECT_LE(st1, std::max(1e-9, 10.0 * st2)) << s.name << " " << what << " stationarity ours " << st1 << " vs minpack " << st2;
      if (s.name.rfind("g4_", 0) != 0) EXPECT_LT((p1.x - p2.x).cwiseAbs().maxCoeff(), 1e-9) << s.name << " " << what;
    };
    same_point(h, a, b, "cold");
    EXPECT_LE(a.iters, 2 * b.iters + 4) << s.name << ": the new driver takes far more iterations";
    cal::BundleProblem moved = p;
    const Shape::Requote rq = s.requote(s.q_big);  // a banded row's band moves with its target
    for (int i = 0; i < s.prob.n_residuals(); ++i) {
      auto& ins = moved.instruments[static_cast<std::size_t>(i)];
      ins.market = rq.target[i]; ins.band_lower = rq.lower[i]; ins.band_upper = rq.upper[i]; ins.band_decay = rq.decay[i];
    }
    const cal::HybridBundleResidual hm(moved);
    const Fit aw = ours(hm, n, m, b.x), bw = minpack(hm, n, m, b.x);
    if (converged_code(bw.info)) {  // a rung where MINPACK itself converges on the move
      EXPECT_TRUE(converged_code(aw.info)) << s.name << " warm ours: " << cal::lm_status_text(aw.info);
      same_point(hm, aw, bw, "warm");
    }
  }
}

TEST(LmDriver, StatusCodesKeepTheirMeaning) {
  const Shape s = swaps::shapes::ois_lag();  // square: the solve ends at machine precision
  const cal::HybridBundleResidual h(s.prob);
  const int n = s.prob.n_knots(), m = s.prob.n_residuals();
  Eigen::VectorXd x = s.x0; int it = 0;
  const int info = cal::lm_normal_equations(h, n, m, x, it);
  EXPECT_TRUE(info == 1 || info == 2 || info == 3 || info == 6 || info == 7) << cal::lm_status_text(info);
  EXPECT_LT((x - s.x_true).cwiseAbs().maxCoeff(), 1e-10);
  EXPECT_GT(it, 0);
  // the evaluation cap is a status, not an exception
  cal::LmParams tight; tight.maxfev = 2;
  Eigen::VectorXd y = s.x0; int it2 = 0;
  EXPECT_EQ(cal::lm_normal_equations(h, n, m, y, it2, tight), 5);
  // m < n is improper input: refused with 0 and x untouched, so calibrate_with's rank path takes over
  Eigen::VectorXd z = s.x0; int it3 = 0;
  EXPECT_EQ(cal::lm_normal_equations(h, n, n - 1, z, it3), 0);
  EXPECT_EQ(z, s.x0);
  // and the whole calibrate_with reports the same converged picture as before
  const cal::CalibrationResult r = cal::calibrate(s.prob, s.x0);
  EXPECT_TRUE(r.converged) << r.status;
  EXPECT_EQ(r.rank_deficiency, 0);
  EXPECT_LT(r.stationarity, 1e-10);
  EXPECT_EQ(std::string(r.status), cal::lm_status_text(r.info));
}
