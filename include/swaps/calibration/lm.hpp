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

#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

// Eigen NonLinearOptimization functor: fvec = r(x), rate units.
struct ResidualFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

  const CalibrationProblem* prob;
  explicit ResidualFunctor(const CalibrationProblem& p) : prob(&p) {}
  int inputs() const { return prob->n_knots(); }
  int values() const { return prob->n_residuals(); }

  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
    fvec = prob->residuals<double>(x);
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

inline CalibrationResult calibrate(const CalibrationProblem& prob, const Eigen::VectorXd& x0) {
  ResidualFunctor functor(prob);
  Eigen::NumericalDiff<ResidualFunctor> num_diff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ResidualFunctor>> lm(num_diff);
  lm.parameters.xtol = 1e-14;
  lm.parameters.ftol = 1e-14;
  lm.parameters.maxfev = 4000;

  CalibrationResult res;
  res.x = x0;
  res.info = lm.minimize(res.x);
  res.iterations = lm.iter;

  const Eigen::VectorXd r = prob.residuals<double>(res.x);
  res.rms_residual = std::sqrt(r.squaredNorm() / r.size());

  Eigen::MatrixXd J(prob.n_residuals(), prob.n_knots());
  num_diff.df(res.x, J);
  res.stationarity = (J.transpose() * r).cwiseAbs().maxCoeff();
  return res;
}

}  // namespace swaps::calibration
