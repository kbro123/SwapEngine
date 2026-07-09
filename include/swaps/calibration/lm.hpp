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
#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

// Analytic Jacobian J[i][k] = d residual_i / d knot_k via forward-mode AAD, in one differentiated
// evaluation of the residual code. This is the north-star (CLAUDE.md §1) — no bump-and-reprice.
// `Problem` is any type exposing residuals<Scalar>(x), n_knots(), n_residuals() — i.e. either a
// CalibrationProblem or a SpreadCalibrationProblem.
template <class Problem>
Eigen::MatrixXd aad_jacobian(const Problem& prob, const Eigen::VectorXd& x) {
  const auto rd = prob.template residuals<ad::Dual>(ad::seed(x));
  const int n = prob.n_residuals(), m = prob.n_knots();
  Eigen::MatrixXd J(n, m);
  for (int i = 0; i < n; ++i) {
    // A residual with no curve dependence would have an empty gradient; guard defensively.
    if (rd[i].derivatives().size() == m)
      J.row(i) = rd[i].derivatives().transpose();
    else
      J.row(i).setZero();
  }
  return J;
}

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

struct CalibrationResult {
  Eigen::VectorXd x;
  int iterations = 0;
  int info = 0;             // Eigen::LevenbergMarquardtSpace::Status
  double rms_residual = 0;  // sqrt(||r||^2 / m)
  double stationarity = 0;  // ||J^T r||_inf  -- the over-determined optimality measure
};

// use_aad = analytic AAD Jacobian (default) or Eigen NumericalDiff. `Problem` is a
// CalibrationProblem or a SpreadCalibrationProblem.
template <class Problem>
CalibrationResult calibrate(const Problem& prob, const Eigen::VectorXd& x0, bool use_aad = true) {
  CalibrationResult res;
  res.x = x0;

  if (use_aad) {
    ResidualFunctorAAD<Problem> functor(prob);
    Eigen::LevenbergMarquardt<ResidualFunctorAAD<Problem>> lm(functor);
    lm.parameters.xtol = 1e-14;
    lm.parameters.ftol = 1e-14;
    lm.parameters.maxfev = 4000;
    res.info = lm.minimize(res.x);
    res.iterations = lm.iter;
  } else {
    ResidualFunctor<Problem> functor(prob);
    Eigen::NumericalDiff<ResidualFunctor<Problem>> num_diff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ResidualFunctor<Problem>>> lm(num_diff);
    lm.parameters.xtol = 1e-14;
    lm.parameters.ftol = 1e-14;
    lm.parameters.maxfev = 4000;
    res.info = lm.minimize(res.x);
    res.iterations = lm.iter;
  }

  const Eigen::VectorXd r = prob.template residuals<double>(res.x);
  res.rms_residual = std::sqrt(r.squaredNorm() / r.size());
  const Eigen::MatrixXd J = aad_jacobian(prob, res.x);
  res.stationarity = (J.transpose() * r).cwiseAbs().maxCoeff();
  return res;
}

}  // namespace swaps::calibration
