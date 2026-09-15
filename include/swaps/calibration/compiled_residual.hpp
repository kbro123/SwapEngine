#pragma once
// Single-curve calibration residual -- now a thin delegate to the multi-curve CompiledBundleResidual
// (a CalibrationProblem is a 1-curve bundle: one self-discounting outright curve, no basis). This keeps
// ONE compiled W-cache engine instead of two parallel copies, while preserving the exact interface the
// LM/warm/streaming code expects (residuals / model_rates / jacobian / n_residuals / n_times).
//
// The shared engine is pricing/compiled_book.hpp: DF = exp(-Wx) once, then the cheap per-type transform
// (futures DF-ratios + convexity, swap par rate = float_pv/annuity), and the analytic Jacobian
// J = -(dr/dDF diag(DF)) W. For one curve W is the single-curve integral-weight matrix -- identical math
// and (measured) identical speed to the former hand-rolled single-curve kernel.

#include <Eigen/Core>

#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

class CompiledResidual {
 public:
  explicit CompiledResidual(const CalibrationProblem& p) : impl_(single_curve_bundle(p)) {}

  int n_residuals() const { return impl_.n_residuals(); }
  int n_times() const { return impl_.n_times(); }
  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const { return impl_.model_rates(x); }
  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return impl_.residuals(x); }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return impl_.jacobian(x); }
  // Streaming against an arbitrary live market q (banded soft residual + consistent Jacobian).
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    return impl_.residuals_vs(x, q);
  }
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    return impl_.jacobian_vs(x, q);
  }
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const {
    impl_.jacobian_vs_into(x, q, J);
  }
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J, Eigen::VectorXd* r) const {
    impl_.jacobian_vs_into(x, q, J, r);
  }

 private:
  CompiledBundleResidual impl_;
};

}  // namespace swaps::calibration
