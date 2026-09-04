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

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/jacobian.hpp"        // aad_jacobian (re-exported here for back-compat)
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

// Same residual, but with the ANALYTIC AAD Jacobian supplied via df() (no numerical differencing).
template <class Problem>
struct ResidualFunctorAAD {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

  const Problem* prob;
  explicit ResidualFunctorAAD(const Problem& p) : prob(&p) {}
  int inputs() const { return prob->n_knots(); }
  int values() const { return prob->n_residuals(); }

  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
    fvec = prob->template residuals<double>(x);
    return 0;
  }
  int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const {
    fjac = aad_jacobian(*prob, x);
    return 0;
  }
};

// LM functor driven by a residual ENGINE (residual_engine_t<Problem>): residuals + the ANALYTIC
// Jacobian, both from the compiled W-cache when the problem has one (CalibrationProblem, BundleProblem)
// and from AAD otherwise. This is what makes cold calibrate() skip the per-iteration AAD sweep AND the
// per-eval curve rebuild -- the engine is built ONCE and reprices off DF = exp(-Wx).
template <class Problem>
struct EngineFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

  const residual_engine_t<Problem>* eng;
  int n_knots_, n_res_;
  EngineFunctor(const residual_engine_t<Problem>& e, int knots, int res)
      : eng(&e), n_knots_(knots), n_res_(res) {}
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

struct CalibrationResult {
  Eigen::VectorXd x;
  int iterations = 0;
  int info = 0;             // Eigen::LevenbergMarquardtSpace::Status
  double rms_residual = 0;  // sqrt(||r||^2 / m)
  double stationarity = 0;  // ||J^T r||_inf  -- the over-determined optimality measure
  double solve_micros = 0;  // engine-measured wall time of the solve (excludes any marshalling); the
                            // caller (BundleSession) stamps this so callers report the ENGINE's own
                            // calibration time, not a language-boundary wall-clock.
};

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
template <class Engine>
CalibrationResult calibrate_with(const Engine& engine, int n_knots, int n_residuals,
                                 const Eigen::VectorXd& x0) {
  CalibrationResult res;
  res.x = x0;
  AnyEngineFunctor<Engine> functor(engine, n_knots, n_residuals);
  Eigen::LevenbergMarquardt<AnyEngineFunctor<Engine>> lm(functor);
  lm.parameters.xtol = 1e-14;
  lm.parameters.ftol = 1e-14;
  lm.parameters.maxfev = 4000;
  res.info = lm.minimize(res.x);
  res.iterations = lm.iter;
  const Eigen::VectorXd r = engine.residuals(res.x);
  res.rms_residual = std::sqrt(r.squaredNorm() / r.size());
  const Eigen::MatrixXd J = engine.jacobian(res.x);
  res.stationarity = (J.transpose() * r).cwiseAbs().maxCoeff();
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
  return res;
}

}  // namespace swaps::calibration
