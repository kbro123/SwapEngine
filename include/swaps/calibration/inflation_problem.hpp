#pragma once
// calibration/inflation_problem.hpp — the breakeven-inflation calibration problem. Free variables x = the
// knot forwards of a breakeven ModularCurve (the instantaneous forward breakeven inflation rate); the
// residuals are ZCIS/YoY breakeven discrepancies. Additive, QuantLib-free, header-only.
//
// It exposes the STANDARD calibration interface — residuals<Scalar>(x), n_knots(), n_residuals() — so it
// rides the EXISTING LM+AAD path with no new machinery: residual_engine_t<InflationProblem> resolves to
// the generic AadResidualEngine (like the staged BundleBlockProblem and the test-only spread problem),
// and calibrate()/calibrate_with() in lm.hpp drive it unchanged. Nothing here touches the frozen W-cache
// hot path (compiled_bundle/compiled_book): the ZCIS power transform and the YoY index-ratio coupon are
// smooth AAD residuals, so the analytic-Jacobian AAD tier is exactly the right (and only needed) home.

#include <Eigen/Core>

#include <vector>

#include "swaps/build/inflation_instruments.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/inflation.hpp"

namespace swaps::calibration {

struct InflationProblem {
  // Breakeven-curve topology (year fractions). `meeting_times` is usually empty (inflation has no
  // meeting-date front); `back_times` are the ZCIS/YoY maturities interpolated with a local Hermite —
  // the same flat_hermite layout the nominal curve uses, so the curve machinery is shared verbatim.
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  double base = 100.0;                 // I(0)
  curve::Seasonality seasonality;      // optional; inactive => no seasonal adjustment
  std::vector<build::InflationInstrument> instruments;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const { return static_cast<int>(instruments.size()); }

  // r(x): build the breakeven curve from x, wrap it as an index curve, price each instrument's breakeven
  // discrepancy. RESIDUAL ORDER is the instruments' insertion order (Jacobian rows index off it).
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto bei = curve::make_modular_curve<Scalar>(curve::flat_hermite(meeting_times, back_times));
    bei.set_forwards(x);
    curve::InflationIndexCurve<Scalar> infl{base, &bei, seasonality.active ? &seasonality : nullptr};
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    for (int i = 0; i < n_residuals(); ++i) r[i] = instruments[i].template residual<Scalar>(infl);
    return r;
  }

  // The stored market quotes, in residual order (streaming/model_rates convenience; mirrors the other
  // problems' market()).
  Eigen::VectorXd market() const {
    Eigen::VectorXd m(n_residuals());
    for (int i = 0; i < n_residuals(); ++i) m[i] = instruments[i].market;
    return m;
  }
};

}  // namespace swaps::calibration
