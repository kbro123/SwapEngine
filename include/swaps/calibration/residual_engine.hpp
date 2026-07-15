#pragma once
// Residual/Jacobian engine shared by the warm & streaming re-calibrators, so both drive ANY problem
// that exposes the calibration interface -- the single curve AND the multi-curve BundleProblem.
//
// A "residual engine" answers three questions about a parameter vector x:
//   residuals(x)   = model rates - market, in the LM/Jacobian residual order
//   jacobian(x)    = dr/dx (n_residuals x n_knots)
//   model_rates(x) = the model-implied market quotes at x (= residuals(x) + market)
//
// Two implementations, selected at compile time by residual_engine_t<Problem>:
//   * CalibrationProblem -> CompiledResidual: the vectorized DF = exp(-Wx) path with the ANALYTIC
//     Jacobian (no AAD in the hot loop). This is the microsecond warm/streaming fast path; keep it.
//   * any other Problem (e.g. BundleProblem) -> AadResidualEngine: the templated residuals<double>(x)
//     plus the AAD Jacobian. Correct and generic; the Jacobian refresh costs one AAD sweep instead of
//     the analytic scatter, but refreshes are rare (that is the whole point of the warm envelope).
//
// The engine is the ONLY thing that knew the concrete problem type, so templating it here is what lets
// WarmCalibrator / StreamingCalibrator become problem-generic without duplicating their control flow.

#include <Eigen/Dense>

#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"

namespace swaps::calibration {

// Generic engine for any Problem exposing residuals<Scalar>(x), n_residuals(), and market().
// The AAD Jacobian (lm.hpp) is templated on the problem, so the block-structured bundle Jacobian
// falls straight out of one differentiated sweep of the stacked residual.
template <class Problem>
class AadResidualEngine {
 public:
  explicit AadResidualEngine(const Problem& p) : p_(&p), market_(p.market()) {}

  int n_residuals() const { return p_->n_residuals(); }
  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const { return p_->template residuals<double>(x); }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return aad_jacobian(*p_, x); }
  // model_rates(x) = residuals(x) + market: residuals is (model - market) by construction, so adding
  // the stored targets back recovers the model-implied quotes the streaming feed compares against.
  Eigen::VectorXd model_rates(const Eigen::VectorXd& x) const { return residuals(x) + market_; }

 private:
  const Problem* p_;
  Eigen::VectorXd market_;
};

// CalibrationProblem -> the fast analytic CompiledResidual (which already exposes the same three ops);
// everything else -> the generic AAD engine.
template <class Problem>
struct residual_engine {
  using type = AadResidualEngine<Problem>;
};
template <>
struct residual_engine<CalibrationProblem> {
  using type = CompiledResidual;
};
template <class Problem>
using residual_engine_t = typename residual_engine<Problem>::type;

}  // namespace swaps::calibration
