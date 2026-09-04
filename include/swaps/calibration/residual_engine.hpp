#pragma once
// Residual/Jacobian engine shared by the warm & streaming re-calibrators, so both drive ANY problem
// that exposes the calibration interface -- the single curve AND the multi-curve BundleProblem.
//
// A "residual engine" answers three questions about a parameter vector x:
//   residuals(x)   = model rates - market, in the LM/Jacobian residual order
//   jacobian(x)    = dr/dx (n_residuals x n_knots)
//   model_rates(x) = the model-implied market quotes at x (= residuals(x) + market)
//
// Three implementations, selected at compile time by residual_engine_t<Problem>:
//   * BundleProblem -> HybridBundleResidual: the analytic W_all fast path (DF = exp(-Wx) once, cheap
//     per-type transforms, analytic Jacobian -- no AAD in the hot loop) for the cacheable rows, PLUS a
//     width-reduced AAD block for any non-cacheable rows (FX/MtM). When nothing is non-cacheable it is a
//     zero-overhead delegate to a single CompiledBundleResidual -- the fast path is unchanged.
//   * CalibrationProblem -> CompiledResidual: NOT a second kernel -- a thin delegate that wraps the
//     single curve as a 1-curve bundle (single_curve_bundle) and runs the exact same CompiledBundleResidual.
//     Kept only so single-curve callers/tests get the simpler CalibrationProblem interface.
//   * anything ELSE (e.g. BundleBlockProblem for the staged solve, or a test-only problem such as the
//     fixed-base spread problem in tests/spread_reference.hpp) -> the generic AadResidualEngine:
//     templated residuals<double>(x) + the AAD Jacobian. Correct and generic, but a refresh here costs
//     one AAD sweep -- the only problem types that still AAD on refresh. Refreshes are rare, so fine.
//
// The engine is the ONLY thing that knew the concrete problem type, so templating it here is what lets
// WarmCalibrator / StreamingCalibrator become problem-generic without duplicating their control flow.

#include <Eigen/Dense>

#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"

namespace swaps::calibration {

// Generic engine for any Problem exposing residuals<Scalar>(x), n_residuals() (and, for streaming,
// market()). The AAD Jacobian is templated on the problem, so the block-structured bundle Jacobian
// falls straight out of one differentiated sweep of the stacked residual.
//
// market() is fetched LAZILY (only in model_rates), NOT at construction: cold calibrate() drives this
// engine too but never needs model_rates, so a Problem WITHOUT a market() (e.g. BundleBlockProblem, or a
// test-only problem) still calibrates through the analytic engine -- model_rates is simply never
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

// A residual engine COMPOSED with a constant pseudo-residual block R (rows x n_knots): appends R·x rows
// to the base engine's residual and R itself to its Jacobian. This is the engine-level counterpart of
// LinearRegularizedProblem (regularize.hpp): where that wraps the PROBLEM (and so routes the whole solve
// to the generic AAD engine), this wraps the ENGINE -- the instrument rows keep their compiled W-cache
// residual/Jacobian, and the regulariser costs a GEMV + a block copy. R and the base engine are held by
// reference; both must outlive the composition (BundleSession owns both).
template <class Engine>
class RegularizedEngine {
 public:
  RegularizedEngine(const Engine& base, const Eigen::MatrixXd& R) : base_(&base), R_(&R) {}

  int n_residuals() const { return base_->n_residuals() + static_cast<int>(R_->rows()); }
  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const {
    const auto& r0 = base_->residuals(x);
    Eigen::VectorXd r(r0.size() + R_->rows());
    r.head(r0.size()) = r0;
    r.tail(R_->rows()).noalias() = (*R_) * x;
    return r;
  }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    const Eigen::MatrixXd J0 = base_->jacobian(x);
    Eigen::MatrixXd J(J0.rows() + R_->rows(), J0.cols());
    J.topRows(J0.rows()) = J0;
    J.bottomRows(R_->rows()) = *R_;
    return J;
  }

 private:
  const Engine* base_;
  const Eigen::MatrixXd* R_;
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
  // Hybrid: W-cache for the cacheable rows + a width-reduced AAD block for any non-cacheable ones
  // (FX/MtM). A zero-overhead delegate to CompiledBundleResidual when nothing is non-cacheable.
  using type = HybridBundleResidual;
};
template <class Problem>
using residual_engine_t = typename residual_engine<Problem>::type;

}  // namespace swaps::calibration
