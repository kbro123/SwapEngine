#pragma once
// calibration/credit_problem.hpp — the hazard-rate (survival) calibration problem. Free variables x = the
// knot forwards of a hazard ModularCurve (the instantaneous forward hazard rate h); the residuals are par
// CDS-spread discrepancies. Additive, QuantLib-free, header-only. A direct sibling of
// calibration/inflation_problem.hpp — read that first.
//
// CALIBRATION TIER: this REUSES the existing residual/AAD calibration tier, no bespoke bootstrap. It
// exposes the STANDARD calibration interface — residuals<Scalar>(x), n_knots(), n_residuals(), market() —
// so residual_engine_t<CreditProblem> resolves to the generic AadResidualEngine (the unspecialized
// default, exactly as for InflationProblem and the test-only spread problem) and calibrate() /
// calibrate_with() in lm.hpp drive it unchanged: LM with an analytic AAD Jacobian, plus the seed-anchored
// minimum-norm completion that reports `rank_deficiency`. The CDS par spread is a smooth ratio of survival
// probabilities, so the AAD tier is exactly the right (and only needed) home; nothing here touches the
// frozen W-cache hot path.

#include <Eigen/Core>

#include <vector>

#include "swaps/calibration/credit_instrument.hpp"  // the row type only (E6.4: no build/ dependency)
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"

namespace swaps::calibration {

struct CreditProblem {
  // Hazard-curve topology (year fractions). `meeting_times` is usually empty (credit has no meeting-date
  // front); `back_times` are the CDS maturities interpolated with a local Hermite — the same flat_hermite
  // layout the nominal and breakeven curves use, so the curve machinery is shared verbatim.
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  std::vector<CdsInstrument> instruments;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const { return static_cast<int>(instruments.size()); }

  // r(x): build the hazard curve from x, wrap it as a survival curve, price each instrument's par-spread
  // discrepancy. RESIDUAL ORDER is the instruments' insertion order (Jacobian rows index off it).
  // Hazard-curve layout: PIECEWISE-FLAT hazard over every knot -- the ISDA standard-model shape. A smooth
  // (Hermite) hazard between knots overshoots into NEGATIVE hazard on ORDINARY humped / inverted strips
  // (1y100/3y250/5y150/7y140/10y130: h(10y) = -20%, 16% of the default mass negative), i.e. a non-monotone
  // survival and a negative default density. Flat is monotone by construction, reprices the strip exactly
  // (one free hazard per maturity), and is a linear map of the knots (W-cacheable). One layout, shared by
  // the residual (calibration) and the read-out (api/credit.cpp) so the two can never diverge.
  std::vector<curve::CurveModule> hazard_layout() const {
    std::vector<double> knots = meeting_times;
    knots.insert(knots.end(), back_times.begin(), back_times.end());
    return curve::flat_hermite(knots, {});
  }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto hz = curve::make_modular_curve<Scalar>(hazard_layout());
    hz.set_forwards(x);
    curve::SurvivalCurve<Scalar> surv{&hz};
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    for (int i = 0; i < n_residuals(); ++i) r[i] = instruments[i].template residual<Scalar>(surv);
    return r;
  }

  // The stored market par spreads, in residual order (streaming/model_rates convenience; mirrors the
  // other problems' market()).
  Eigen::VectorXd market() const {
    Eigen::VectorXd m(n_residuals());
    for (int i = 0; i < n_residuals(); ++i) m[i] = instruments[i].market;
    return m;
  }
};

}  // namespace swaps::calibration
