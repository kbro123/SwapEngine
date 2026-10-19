#pragma once
// Global Levenberg-Marquardt calibration of the knot forwards.
//
// The shipped solve (calibrate / calibrate_with) is lm_normal_equations below: Levenberg-Marquardt on the scaled
// normal equations with the engine's analytic Jacobian (2026-10-19). Eigen's NonLinearOptimization LM with a
// numerical-difference Jacobian stays as the INDEPENDENT optimizer the calibration tests cross-check against
// (calibrate(..., use_aad = false), CLAUDE.md section 3(b)). Because the calibration is over-determined (CLAUDE.md
// section 2), success is judged by FIRST-ORDER STATIONARITY (||J^T r||_inf), never by exact repricing.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/jacobian.hpp"        // aad_jacobian (re-exported here for back-compat)
#include "swaps/calibration/normal_op.hpp"       // NormalOp: the ONE rank rule (rank_deficiency)
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/residual_engine.hpp"  // residual_engine_t: the analytic fast path for cold calibrate

namespace swaps::calibration {

// Eigen NonLinearOptimization functor: fvec = r(x), rate units.
template <class Problem>
struct ResidualFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

  const Problem* prob;
  explicit ResidualFunctor(const Problem& p) : prob(&p) {}
  int inputs() const { return prob->n_knots(); }
  int values() const { return prob->n_residuals(); }

  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
    fvec = prob->template residuals<double>(x);
    return 0;
  }
};

struct CalibrationResult {
  Eigen::VectorXd x;
  int iterations = 0;
  int info = 0;             // Eigen::LevenbergMarquardtSpace::Status
  double rms_residual = 0;  // sqrt(||r||^2 / m)
  double stationarity = 0;  // ||J^T r||_inf  -- the over-determined optimality measure
  double solve_micros = 0;  // engine-measured wall time of the solve (excludes any marshalling); the
                            // caller (BundleSession) stamps this so callers report the ENGINE's own
                            // calibration time, not a language-boundary wall-clock.
  // Number of NUMERICALLY UNCONSTRAINED state directions (n_knots - rank(J) at the solution, at the
  // shared kRankThreshold). 0 for a well-posed problem. When > 0, `x` is the MINIMUM-NORM completion:
  // among the least-squares optima, the one closest to the seed x0 (see calibrate_with) -- the null
  // components sit exactly at their seed values instead of wherever the LM path wandered. A nonzero
  // value means the INSTRUMENT SET under-determines the curve (e.g. a front knot before the first
  // instrument, an unreached long knot): surface it to the user -- add an instrument or enable
  // smoothing -- rather than treating the completed values as market-implied.
  int rank_deficiency = 0;
  // Did the solve END at a stationary point -- LM's own xtol/ftol/gtol criteria, or "tolerance too small"
  // (= already at machine precision) -- with a finite x, rms and stationarity? false for the evaluation
  // cap (info 5), improper input (info 0) or a non-finite result. `status` names the stopping reason
  // (Eigen::LevenbergMarquardtSpace::Status). Non-finite INPUT (a NaN/inf quote or seed) never reaches
  // here: calibrate()/calibrate_with() throw std::invalid_argument up front (2026-09-10; before that a NaN
  // quote returned the seed curve with info 4 "converged" and rms = NaN -- probe C-lm-status).
  bool converged = false;
  const char* status = "not started";
};

inline const char* lm_status_text(int info) {
  switch (info) {
    case -2: return "not started";
    case -1: return "running";
    case 0: return "improper input parameters";
    case 1: return "relative reduction too small";
    case 2: return "relative error too small";
    case 3: return "relative error and reduction too small";
    case 4: return "cosine too small";
    case 5: return "too many function evaluations";
    case 6: return "ftol too small";
    case 7: return "xtol too small";
    case 8: return "gtol too small";
    case 9: return "user asked";
  }
  return "unknown";
}
inline void finish_status(CalibrationResult& res) {
  res.status = lm_status_text(res.info);
  const bool lm_ok = res.info >= 1 && res.info <= 8 && res.info != 5;
  res.converged = lm_ok && res.x.allFinite() && std::isfinite(res.rms_residual) && std::isfinite(res.stationarity);
}
// A NaN/inf quote, FX rate or seed is an INPUT error, not a solve outcome: refuse it here so no path can
// hand back a finite-looking "converged" seed curve for a poisoned market.
inline void require_finite_seed(const Eigen::VectorXd& x0) {
  if (!x0.allFinite()) throw std::invalid_argument("calibrate: the seed x0 contains a non-finite value");
}
inline void require_finite_residual(const Eigen::VectorXd& r0) {
  if (!r0.allFinite())
    throw std::invalid_argument("calibrate: the residual at the seed is non-finite -- a quote (market, band or FX rate) is NaN/inf");
}

// THE DRIVER: MINPACK's lmder on the SCALED NORMAL EQUATIONS (2026-10-19, ASSUMPTIONS K50).
//
// Until then calibrate_with ran Eigen's NonLinearOptimization LevenbergMarquardt (MINPACK lmder as is): a column-pivoted
// Householder QR of J every iteration plus lmpar's Givens re-triangularisations per trust-region trial. On the G4 desk
// (352 rows x 200 knots) that was 4.0 ms of a 5.9 ms iteration against 1.8 ms for the Jacobian itself. The CONTROL
// FLOW is kept exactly -- the trust region, lmpar's secular iteration for the damping parameter, the step bound
// updates, the acceptance threshold, the stopping tests and the status codes (lm_status_text) -- and only the
// factorisation under it changes: with D MINPACK's own column scaling (the running max of J's column norms) the
// damped step solves (D^-1 J^T J D^-1 + par I) ps = -D^-1 J^T r by a Cholesky (0.18 ms) of a matrix formed once per
// iteration (J^T J, 0.44 ms), instead of a QR (1.9 ms) plus Givens per trial. The squared condition number is harmless
// under the scaling: on the desk cond(J) is 2e5 but cond(D^-1 J) 7e2, and the step equals lmder's to 5e-14 on the
// CCP discount columns (three orders below the RFR columns). Why the control flow had to be MINPACK's and not a
// damping schedule: a Madsen-Nielsen-Tingleff mu (shed by at most a third per accepted step, doubled on a rejection)
// was measured taking 28..155 iterations on the mixed MonotoneCubic bundle's anchored solve where lmder takes 4..7,
// and parking at a Hyman-branch kink with the residual at 1e-6 from four seeds in ten -- after one rejection burst mu
// stays large for dozens of steps and the full Gauss-Newton step, which lmder retries as soon as its region allows,
// never comes. A rank-deficient J leaves the damped system singular only at par = 0: that Cholesky's failure is the
// signal to damp (as lmpar's rank-deficient branch), and the null components the path parks are what calibrate_with's
// seed-anchored re-solve already corrects.
struct LmParams {
  double xtol = 1e-14, ftol = 1e-14, gtol = 0.0, factor = 100.0;  // lmder's defaults as calibrate_with set them
  int maxfev = 4000;
};
namespace detail {
// lmpar on the scaled normal matrix: given As = D^-1 J^T J D^-1 (lower triangle), gs = D^-1 J^T r and the step bound
// delta, find par >= 0 and ps with (As + par I) ps = -gs such that either par = 0 and ||ps|| <= delta, or
// ||ps|| = delta to 10% (More's secular iteration, at most 10 steps -- MINPACK lmpar.f with a Cholesky where it has R).
struct Lmpar {
  Eigen::MatrixXd M;
  Eigen::VectorXd q;
  Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> llt;
  explicit Lmpar(int n) : M(n, n), q(n), llt(n) {}
  // One damped solve; false when As + par I is not positive definite (par == 0 on a rank-deficient J).
  bool solve(const Eigen::MatrixXd& As, const Eigen::VectorXd& gs, double par, Eigen::VectorXd& ps) {
    M = As;
    M.diagonal().array() += par;
    llt.compute(M);
    if (llt.info() != Eigen::Success) return false;
    ps = llt.solve(-gs);
    return true;
  }
  double run(const Eigen::MatrixXd& As, const Eigen::VectorXd& gs, double delta, double par, Eigen::VectorXd& ps) {
    constexpr double p1 = 0.1;
    double parl = 0.0, paru = gs.norm() / delta;  // paru: the gradient bound (lmpar.f)
    double dxnorm = 0.0, fp = 0.0;
    // the Gauss-Newton step is the answer when it fits the region
    if (solve(As, gs, 0.0, ps)) {
      dxnorm = ps.norm();
      fp = dxnorm - delta;
      if (fp <= p1 * delta) return 0.0;
      // the lower bound: Newton's step on the secular equation from par = 0 (full rank only, as lmpar.f)
      q = llt.matrixL().solve(ps);  // L^-1 ps
      const double t = q.norm() / dxnorm;
      parl = (fp / delta) / (t * t);
    }
    if (paru == 0.0) paru = std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
    par = std::clamp(par, parl, paru);
    if (par == 0.0) par = gs.norm() / (dxnorm > 0.0 ? dxnorm : delta);
    for (int it = 1;; ++it) {
      if (par == 0.0) par = std::max(std::numeric_limits<double>::min(), 0.001 * paru);
      while (!solve(As, gs, par, ps)) par = std::max(2.0 * par, std::numeric_limits<double>::min());  // never at par > 0 on finite data
      dxnorm = ps.norm();
      const double fp_old = fp;
      fp = dxnorm - delta;
      if (std::abs(fp) <= p1 * delta || (parl == 0.0 && fp <= fp_old && fp_old < 0.0) || it == 10) return par;
      q = llt.matrixL().solve(ps);
      const double t = q.norm() / dxnorm;
      const double parc = (fp / delta) / (t * t);
      if (fp > 0.0) parl = std::max(parl, par);
      else paru = std::min(paru, par);
      par = std::max(parl, par + parc);
    }
  }
};
}  // namespace detail
// Drives `x` from its seed to a stationary point of 1/2 ||r(x)||^2 over an engine with residuals(x) / jacobian(x).
// Returns the status (lm_status_text); `iters` counts the accepted steps (each one Jacobian) -- lmder's iter - 1.
template <class Engine>
int lm_normal_equations(const Engine& eng, int n, int m, Eigen::VectorXd& x, int& iters, const LmParams& prm = {}) {
  iters = 0;
  if (m < n || n == 0) return 0;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  Eigen::VectorXd r = eng.residuals(x);
  double fnorm = r.norm();
  int nfev = 1;
  Eigen::MatrixXd J, A(n, n), As(n, n);
  Eigen::VectorXd g(n), gs(n), diag(n), wa2(n), ps(n), p(n), xnew(n), rnew(m), Jp(m);
  detail::Lmpar lmpar(n);
  double delta = 0.0, xnorm = 0.0, par = 0.0;
  for (;;) {  // one Jacobian per pass
    J = eng.jacobian(x);
    A.setZero();
    A.template selfadjointView<Eigen::Lower>().rankUpdate(J.transpose());
    g.noalias() = J.transpose() * r;
    for (int j = 0; j < n; ++j) wa2[j] = J.col(j).norm();
    if (iters == 0) {  // the first iteration: scale by the seed Jacobian's column norms, open the step bound
      for (int j = 0; j < n; ++j) diag[j] = wa2[j] == 0.0 ? 1.0 : wa2[j];
      xnorm = diag.cwiseProduct(x).norm();
      delta = prm.factor * xnorm;
      if (delta == 0.0) delta = prm.factor;
    }
    double gnorm = 0.0;
    if (fnorm != 0.0)
      for (int j = 0; j < n; ++j)
        if (wa2[j] != 0.0) gnorm = std::max(gnorm, std::abs(g[j]) / (wa2[j] * fnorm));
    if (gnorm <= prm.gtol) return 4;
    diag = diag.cwiseMax(wa2);
    for (int j = 0; j < n; ++j)
      for (int i = j; i < n; ++i) As(i, j) = A(i, j) / (diag[i] * diag[j]);
    gs = g.cwiseQuotient(diag);
    for (;;) {  // the inner loop: trials on this Jacobian until a step is accepted
      par = lmpar.run(As, gs, delta, par, ps);
      p = ps.cwiseQuotient(diag);  // ps = D p
      const double pnorm = ps.norm();
      if (iters == 0) delta = std::min(delta, pnorm);
      xnew = x + p;
      rnew = eng.residuals(xnew);
      ++nfev;
      const double fnorm1 = rnew.norm();
      const double actred = (0.1 * fnorm1 < fnorm) ? 1.0 - (fnorm1 / fnorm) * (fnorm1 / fnorm) : -1.0;
      Jp.noalias() = J * p;
      const double temp1 = Jp.norm() / fnorm, temp2 = std::sqrt(par) * pnorm / fnorm;
      const double prered = temp1 * temp1 + temp2 * temp2 / 0.5;
      const double dirder = -(temp1 * temp1 + temp2 * temp2);
      const double ratio = prered != 0.0 ? actred / prered : 0.0;
      if (ratio <= 0.25) {  // shrink the region (lmder's quadratic shrink)
        double temp = actred >= 0.0 ? 0.5 : 0.5 * dirder / (dirder + 0.5 * actred);
        if (0.1 * fnorm1 >= fnorm || temp < 0.1) temp = 0.1;
        delta = temp * std::min(delta, pnorm / 0.1);
        par /= temp;
      } else if (par == 0.0 || ratio >= 0.75) {
        delta = pnorm / 0.5;
        par *= 0.5;
      }
      const bool accepted = ratio >= 1e-4;
      if (accepted) {
        x = xnew;
        r = rnew;
        xnorm = diag.cwiseProduct(x).norm();
        fnorm = fnorm1;
        ++iters;
      }
      int info = 0;
      if (std::abs(actred) <= prm.ftol && prered <= prm.ftol && 0.5 * ratio <= 1.0) info = 1;
      if (delta <= prm.xtol * xnorm) info += 2;
      if (info) return info;
      if (nfev >= prm.maxfev) return 5;
      if (std::abs(actred) <= eps && prered <= eps && 0.5 * ratio <= 1.0) return 6;
      if (delta <= eps * xnorm) return 7;
      if (gnorm <= eps) return 8;
      if (accepted) break;
    }
  }
}

// The LM loop over a PREBUILT residual engine. Identical control flow / tolerances / result stats to
// calibrate() below -- calibrate() IS this after constructing the engine -- but the engine's lifetime
// belongs to the caller, so a warm caller pays construction (the W-cache build) once, not per solve.
//
// MINIMUM-NORM completion: if the problem is rank-deficient at the solution (rank(J) < n_knots at the
// shared kRankThreshold), LM converges to A least-squares optimum but the null-direction components of
// x are path-dependent garbage (they change no residual, so LM parks them wherever its trajectory
// wandered -- forwards of -570% have been observed on an under-determined web spec). This snaps the
// result to the optimum CLOSEST TO THE SEED: x = x0 + J⁺J(x_lm - x0), the range-space projection via
// the same rank-thresholded complete orthogonal decomposition the streaming operator uses. For a warm
// re-solve the "seed" is the previous solution, so unconstrained states stay put tick to tick. A
// full-rank problem is a no-op (rank check only); the residual stats are refreshed at the snapped point.
template <class Engine>
CalibrationResult calibrate_with(const Engine& engine, int n_knots, int n_residuals,
                                 const Eigen::VectorXd& x0) {
  CalibrationResult res;
  res.x = x0;
  require_finite_seed(x0);
  {
    const Eigen::VectorXd r0 = engine.residuals(x0);  // a copy: engines may hand back an internal buffer
    require_finite_residual(r0);
  }
  res.info = lm_normal_equations(engine, n_knots, n_residuals, res.x, res.iterations);
  Eigen::MatrixXd J = engine.jacobian(res.x);

  NormalOp op(n_residuals, n_knots);
  op.decompose(J);  // the ONE rank rule (kRankThreshold), shared with the streamer and the risk operators
  res.rank_deficiency = n_knots - static_cast<int>(op.rank());
  if (res.rank_deficiency > 0) {
    // SEED-ANCHORED re-solve (rank-deficient problems ONLY; a determined problem never enters here).
    // A post-hoc null-space projection is NOT enough: the null combination of a nonlinear problem is
    // x-dependent, so a large projection moves the residual at second order and a re-polish wanders
    // back along the null. Instead re-run the SAME least squares with tiny anchor rows w·(x − x0)
    // appended: in the null directions the anchor is the only force (they land exactly at the seed);
    // in constrained directions its pull is (w/sigma)^2-suppressed — w = 1e-8·|max pivot| biases a
    // genuine direction by parts-per-billion of its value. For a WARM re-solve x0 is the previous
    // solution, so unconstrained states stay put tick to tick.
    const double w = 1e-8 * std::abs(op.max_pivot());
    // The anchor rows ARE regulariser rows centred on the seed: R = w·I, r = R·(x − x0) (constraint rows, stage B
    // 2026-10-09 -- the same composition a smoothing R enters a calibrate by, not a second hand-stacked engine).
    const Eigen::MatrixXd R_anchor = w * Eigen::MatrixXd::Identity(n_knots, n_knots);
    const RegularizedEngine<Engine> anchored(engine, R_anchor, x0);
    res.x = x0;  // the anchored problem is full-rank: one clean solve from the seed
    int anchored_iters = 0;
    res.info = lm_normal_equations(anchored, n_knots, n_residuals + n_knots, res.x, anchored_iters);  // the status of the solve that produced x (the first one refused m < n as "improper input")
    res.iterations += anchored_iters;
    J = engine.jacobian(res.x);
  }

  const Eigen::VectorXd r = engine.residuals(res.x);
  res.rms_residual = std::sqrt(r.squaredNorm() / r.size());
  res.stationarity = (J.transpose() * r).cwiseAbs().maxCoeff();
  finish_status(res);
  return res;
}

// use_aad = drive the LM with the ANALYTIC residual engine (default; compiled W-cache Jacobian for
// CalibrationProblem/BundleProblem, AAD otherwise) or Eigen NumericalDiff. The engine is built ONCE, so
// cold calibrate no longer does a per-iteration AAD sweep or a per-eval curve rebuild.
template <class Problem>
CalibrationResult calibrate(const Problem& prob, const Eigen::VectorXd& x0, bool use_aad = true) {
  CalibrationResult res;
  res.x = x0;

  if (use_aad) {
    const residual_engine_t<Problem> engine(prob);  // W-cache built once; analytic Jacobian per iter
    return calibrate_with(engine, prob.n_knots(), prob.n_residuals(), x0);
  }

  require_finite_seed(x0);
  require_finite_residual(prob.template residuals<double>(x0));
  ResidualFunctor<Problem> functor(prob);
  Eigen::NumericalDiff<ResidualFunctor<Problem>> num_diff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ResidualFunctor<Problem>>> lm(num_diff);
  lm.parameters.xtol = 1e-14;
  lm.parameters.ftol = 1e-14;
  lm.parameters.maxfev = 4000;
  res.info = lm.minimize(res.x);
  res.iterations = lm.iter;
  const Eigen::VectorXd r = prob.template residuals<double>(res.x);
  res.rms_residual = std::sqrt(r.squaredNorm() / r.size());
  const Eigen::MatrixXd J = aad_jacobian(prob, res.x);
  res.stationarity = (J.transpose() * r).cwiseAbs().maxCoeff();
  finish_status(res);
  return res;
}

}  // namespace swaps::calibration
