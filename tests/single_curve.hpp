#pragma once
// TEST FIXTURE (2026-11-01, review item 2.1): the SINGLE-CURVE calibration problem, moved out of the shipped object
// model. No verb or session ever built a CalibrationProblem -- the shipped problem is BundleProblem, of which this
// is a one-curve spelling (single_curve_bundle). It stays as the fixture the SOFR reference market, the shape-ladder
// oracles and the gated sofr_23k_* / risk_full_jacobian benches are written against, with its CompiledResidual
// delegate (a 1-curve CompiledBundleResidual, registered as its residual engine below) and the single-curve risk
// operators. Namespace swaps::testing, like spread_reference.hpp.
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/normal_op.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/curve/curve_module.hpp"

namespace swaps::testing {

namespace cal = swaps::calibration;
using cal::Instrument;
using cal::BundleProblem;
using cal::instrument_residual;
using cal::aad_jacobian;
using cal::NormalOp;
using cal::CompiledBundleResidual;

struct CalibrationProblem {
  // Curve topology (year fractions on the curve day count).
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  // Every calibration instrument is a generic Instrument (design §3). This problem has exactly ONE
  // curve, so every leg role resolves to it and the legs' curve indices are ignored (single_curve_bundle()
  // likewise forces them all to curve 0, so the templated and compiled paths cannot diverge).
  std::vector<Instrument> instruments;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const { return static_cast<int>(instruments.size()); }

  // r in rate units for an ARBITRARY curve (anything with `Scalar discount(double)`). RESIDUAL ORDER
  // is the instruments' INSERTION order (the Jacobian rows, the W-cache batches and the warm/streaming
  // feeds all index off it; CompiledResidual fills the same rows).
  template <class Scalar, class Curve>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> price_residuals(const Curve& c) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    const auto one_curve = [&c](int) -> const Curve& { return c; };  // every role IS this curve
    int i = 0;
    for (const auto& ins : instruments) r[i++] = instrument_residual<Scalar>(ins, one_curve);
    return r;
  }

  // THE single-curve layout (flat front over the meeting dates + Hermite back), built once here so every
  // consumer (residuals, risk.hpp's operators, tests) derives it from the same place (E6.1c: it used to be
  // re-derived in five headers).
  template <class Scalar>
  curve::ModularCurve<Scalar> make_curve() const {
    return curve::make_modular_curve<Scalar>(curve::flat_hermite(meeting_times, back_times));
  }

  // r(x) with x = the knot forwards of a standalone two-region curve.
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = make_curve<Scalar>();
    c.set_forwards(x);
    return price_residuals<Scalar>(c);
  }
};

// A single-curve CalibrationProblem IS a 1-curve bundle: one self-discounting outright curve, every
// instrument forecasting and discounting curve 0. This lets the single-curve CompiledResidual be a
// thin delegate to CompiledBundleResidual -- one compiled engine, not two.
inline BundleProblem single_curve_bundle(const CalibrationProblem& p) {
  BundleProblem b;
  b.curves.push_back({.regions = curve::flat_hermite(p.meeting_times, p.back_times)});  // base = -1
  for (Instrument ins : p.instruments) {  // by value: force every role onto the single curve, so the
    ins.fwd.forecast = ins.fwd.discount = 0;  // compiled path cannot disagree with price_residuals(),
    ins.bench.forecast = ins.bench.discount = 0;  // which resolves every role to the one curve.
    ins.fixed.discount = 0;
    ins.forecast = 0;
    b.instruments.push_back(std::move(ins));
  }
  return b;
}

class CompiledResidual {
 public:
  explicit CompiledResidual(const CalibrationProblem& p) : impl_(single_curve_bundle(p)) {}

  int n_residuals() const { return impl_.n_residuals(); }
  int n_times() const { return impl_.n_times(); }
  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const { return impl_.model_rates(x); }
  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return impl_.residuals(x); }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return impl_.jacobian(x); }
  // Streaming against an arbitrary live market q (banded soft residual + consistent Jacobian).
  const Eigen::VectorXd& last_model_rates() const { return impl_.last_model_rates(); }
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::VectorXd* model) const {
    const Eigen::VectorXd& r = impl_.residuals_vs(x, q);
    if (model) *model = impl_.last_model_rates();
    return r;
  }
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    return impl_.residuals_vs(x, q);
  }
  const std::vector<int>& dead_rows() const { return impl_.dead_rows(); }  // dead-zone quote rows (row_engine.hpp)
  void dead_row_into(int k, Eigen::Ref<Eigen::RowVectorXd, 0, Eigen::InnerStride<>> out) const { out = impl_.dead_J().row(k); }
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

// D = diag(−∂r_i/∂q_i) at x for a single-curve problem (see the header note).
inline Eigen::VectorXd residual_market_scale(const CalibrationProblem& prob, const Eigen::VectorXd& /*x*/) {
  return residual_market_scale(prob.instruments);  // problem.hpp: the one definition (D needs no curve)
}

// The IFT quote-sensitivity operator dx/dq = J⁺ D  (n_knots x n_residuals), rank-safe.
inline Eigen::MatrixXd ift_operator(const CalibrationProblem& prob, const Eigen::VectorXd& x) {
  const Eigen::MatrixXd J = aad_jacobian(prob, x);  // n_resid x n_knots
  NormalOp op(static_cast<int>(J.rows()), static_cast<int>(J.cols()));
  op.factor(J);
  return op.quote_sensitivity(residual_market_scale(prob, x));
}

// One AAD pass for d(NPV)/dx off the calibration curve, in a caller-chosen forward-AAD scalar.
template <class Scalar, class Portfolio>
inline Eigen::VectorXd book_curve_grad(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                                       const Portfolio& pf,
                                       const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& seeded) {
  auto c = prob.template make_curve<Scalar>();
  c.set_forwards(seeded);
  const Scalar pv = pf.template npv<Scalar>(c);
  return Eigen::VectorXd(pv.derivatives());  // length n_knots (empty for a curve-independent book)
}

// d(NPV)/dq_j for every market quote j, at the calibrated x. `pf` must expose
//   template <class Scalar, class Curve> Scalar npv(const Curve&) const.
template <class Portfolio>
Eigen::VectorXd bucketed_delta(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                               const Portfolio& pf) {
  // d(NPV)/dx via one AAD pass — R11 pooled (heap-free) for a narrow problem, heap Dual beyond.
  const Eigen::VectorXd dnpv_dx =
      prob.n_knots() <= ad::kPooledMaxW
          ? book_curve_grad<ad::DualPooled<ad::kPooledMaxW>>(prob, x, pf,
                                                             ad::seed_pooled<ad::kPooledMaxW>(x))
          : book_curve_grad<ad::Dual>(prob, x, pf, ad::seed(x));

  // Chain through the IFT: d(NPV)/dq = (dx/dq)ᵀ d(NPV)/dx = (J⁺ D)ᵀ dnpv_dx.
  return ift_operator(prob, x).transpose() * dnpv_dx;  // length n_resid = d(NPV)/dq per quote
}

}  // namespace swaps::testing

namespace swaps::calibration {
// The fixture's residual engine: the compiled delegate (the same registration the shipped BundleProblem has for
// its hybrid engine), so calibrate / StreamingCalibrator / aad_jacobian drive it on the analytic path.
template <>
struct residual_engine<swaps::testing::CalibrationProblem> {
  using type = swaps::testing::CompiledResidual;
};
}  // namespace swaps::calibration
