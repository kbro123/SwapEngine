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

#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/jacobian.hpp"

namespace swaps::calibration {

// Generic engine for any Problem exposing residuals<Scalar>(x), n_residuals() (and, for streaming,
// market()). The AAD Jacobian is templated on the problem, so the block-structured bundle Jacobian
// falls straight out of one differentiated sweep of the stacked residual.
//
// market() is fetched LAZILY (only in model_rates), NOT at construction: cold calibrate() drives this
// engine too but never needs model_rates, so a Problem WITHOUT a market() (SpreadCalibrationProblem,
// BundleBlockProblem) still calibrates through the analytic engine -- model_rates is simply never
// instantiated for it. Only the streaming path calls model_rates, and only for problems that have market().
template <class Problem>
class AadResidualEngine {
 public:
  explicit AadResidualEngine(const Problem& p) : p_(&p) {}

  int n_residuals() const { return p_->n_residuals(); }
  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const { return p_->template residuals<double>(x); }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return aad_jacobian(*p_, x); }
  // model_rates(x) = residuals(x) + market: residuals is (model - market) by construction, so adding
  // the stored targets back recovers the model-implied quotes the streaming feed compares against.
  Eigen::VectorXd model_rates(const Eigen::VectorXd& x) const { return residuals(x) + p_->market(); }

 private:
  const Problem* p_;
};

// CalibrationProblem -> the single-curve analytic CompiledResidual; BundleProblem -> the multi-curve
// analytic CompiledBundleResidual (W_all fast path); anything else -> the generic AAD engine. All three
// expose the same residuals/jacobian/model_rates/n_residuals ops, so warm/streaming are oblivious.
template <class Problem>
struct residual_engine {
  using type = AadResidualEngine<Problem>;
};
template <>
struct residual_engine<CalibrationProblem> {
  using type = CompiledResidual;
};
template <>
struct residual_engine<BundleProblem> {
  using type = CompiledBundleResidual;
};
template <class Problem>
using residual_engine_t = typename residual_engine<Problem>::type;

}  // namespace swaps::calibration
