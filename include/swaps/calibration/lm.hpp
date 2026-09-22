#pragma once
// Global Levenberg-Marquardt calibration of the knot forwards.
//
// Phase 2 uses Eigen's numerical-difference Jacobian. Phase 3 replaces df() with the analytic AAD
// Jacobian (forward-mode vector-dual) with no change to the residual code. Because the calibration
// is over-determined (CLAUDE.md §2), success is judged by FIRST-ORDER STATIONARITY (||J^T r||_inf),
// never by exact repricing.

#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <cmath>
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

// LM functor over ANY residual engine (residuals(x) + jacobian(x)), not tied to residual_engine_t --
// what lets a caller drive the SAME LM loop with an engine it built once and keeps across solves
// (BundleSession's warm recalibrate/rebind), or with a composed engine (compiled + a constant
// regularizer block) that has no Problem type of its own.
template <class Engine>
struct AnyEngineFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

  const Engine* eng;
  int n_knots_, n_res_;
  AnyEngineFunctor(const Engine& e, int knots, int res) : eng(&e), n_knots_(knots), n_res_(res) {}
  int inputs() const { return n_knots_; }
  int values() const { return n_res_; }
  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
    fvec = eng->residuals(x);
    return 0;
  }
  int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const {
    fjac = eng->jacobian(x);
    return 0;
  }
};

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
  AnyEngineFunctor<Engine> functor(engine, n_knots, n_residuals);
  Eigen::LevenbergMarquardt<AnyEngineFunctor<Engine>> lm(functor);
  lm.parameters.xtol = 1e-14;
  lm.parameters.ftol = 1e-14;
  lm.parameters.maxfev = 4000;
  res.info = lm.minimize(res.x);
  res.iterations = lm.iter;
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
    struct Anchored {
      const Engine* base;
      const Eigen::VectorXd* x0;
      double w;
      int nk;
      Eigen::VectorXd residuals(const Eigen::VectorXd& x) const {
        const auto& r0 = base->residuals(x);
        Eigen::VectorXd r(r0.size() + nk);
        r.head(r0.size()) = r0;
        r.tail(nk) = w * (x - *x0);
        return r;
      }
      Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
        const Eigen::MatrixXd J0 = base->jacobian(x);
        Eigen::MatrixXd J(J0.rows() + nk, nk);
        J.topRows(J0.rows()) = J0;
        J.bottomRows(nk) = w * Eigen::MatrixXd::Identity(nk, nk);
        return J;
      }
    } anchored{&engine, &x0, w, n_knots};
    AnyEngineFunctor<Anchored> af(anchored, n_knots, n_residuals + n_knots);
    Eigen::LevenbergMarquardt<AnyEngineFunctor<Anchored>> alm(af);
    alm.parameters.xtol = 1e-14;
    alm.parameters.ftol = 1e-14;
    alm.parameters.maxfev = 4000;
    res.x = x0;  // the anchored problem is full-rank: one clean solve from the seed
    res.info = alm.minimize(res.x);  // the status of the solve that produced x (the first one refused m < n as "improper input")
    res.iterations += alm.iter;
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
