#pragma once
// TEST-ONLY reference implementations for the fixed-base forward-spread validation (spread_test.cpp).
//
// These are NOT part of the shipped engine. Production spread curves are calibrated JOINTLY (base +
// spread solved together) via the bundle path's SpreadHandle over ModularCurve. The two types here model
// the SIMPLER textbook case -- a spread over a FIXED base -- purely so spread_test.cpp can pin the
// decomposition math (DF = base_DF · exp(-∫spread); forward = base + spread) and the AAD Jacobian against
// a bump reference. They live in tests/ and in namespace swaps::testing so they cannot leak into the
// library's object model. If a genuine fixed-base spread product is ever needed in production, promote
// them back into include/ deliberately.

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "swaps/calibration/problem.hpp"   // CalibrationProblem (instrument set + pricing) is reused
#include "swaps/curve/curve_module.hpp"

namespace swaps::testing {

// forward(t) = base(t) + spread(t); integral(t) = base + spread; DF = base_DF · exp(-∫spread). The free
// variables are the SPREAD forwards; the base is held fixed (double). Templated on Scalar so AAD flows
// through the spread. The spread rides the shipped flat_hermite layout, so it stays a linear map of its
// knots and AAD-differentiable.
template <class Scalar>
class SpreadCurve {
 public:
  SpreadCurve(const curve::ModularCurve<double>& base, const std::vector<double>& meeting_times,
              const std::vector<double>& back_times)
      : base_(&base),
        spread_(curve::make_modular_curve<Scalar>(curve::flat_hermite(meeting_times, back_times))) {}

  int n_knots() const { return spread_.n_knots(); }
  double max_time() const { return spread_.max_time(); }

  // Spread forwards at the knots (front spreads, then back spreads). May be negative.
  template <class Vec>
  void set_spreads(const Vec& s) {
    spread_.set_forwards(s);
  }

  Scalar forward(double t) const { return spread_.forward(t) + base_->forward(t); }
  // Scalar term (carries derivatives) leads; base double added second (AAD-safe).
  Scalar integral(double t) const { return spread_.integral(t) + base_->integral(t); }
  Scalar discount(double t) const {
    using std::exp;
    return exp(-integral(t));
  }
  Scalar zero(double t) const {
    if (t <= 0.0) return forward(0.0);
    return integral(t) / t;
  }
  const curve::ModularCurve<Scalar>& spread_component() const { return spread_; }

 private:
  const curve::ModularCurve<double>* base_;
  curve::ModularCurve<Scalar> spread_;
};

// Calibrating a spread to a FIXED base curve. Reuses the instrument set + pricing of a
// CalibrationProblem (its meeting_times/back_times are the SPREAD knot times); the free variables are the
// spread forwards s. Duck-types the calibration interface (residuals / n_knots / n_residuals), so the
// generic AAD-LM + risk code drives it unchanged (it falls to AadResidualEngine -- it has no compiled
// engine, which is fine for a test-only path).
struct SpreadCalibrationProblem {
  calibration::CalibrationProblem inst;         // instruments + spread knot times
  const curve::ModularCurve<double>* base;      // fixed base curve

  int n_knots() const { return inst.n_knots(); }
  int n_residuals() const { return inst.n_residuals(); }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& s) const {
    SpreadCurve<Scalar> c(*base, inst.meeting_times, inst.back_times);
    c.set_spreads(s);
    return inst.price_residuals<Scalar>(c);
  }
};

}  // namespace swaps::testing
