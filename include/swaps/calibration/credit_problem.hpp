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

#include "swaps/build/credit_instruments.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"

namespace swaps::calibration {

struct CreditProblem {
  // Hazard-curve topology (year fractions). `meeting_times` is usually empty (credit has no meeting-date
  // front); `back_times` are the CDS maturities interpolated with a local Hermite — the same flat_hermite
  // layout the nominal and breakeven curves use, so the curve machinery is shared verbatim.
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  std::vector<build::CdsInstrument> instruments;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const { return static_cast<int>(instruments.size()); }

  // r(x): build the hazard curve from x, wrap it as a survival curve, price each instrument's par-spread
  // discrepancy. RESIDUAL ORDER is the instruments' insertion order (Jacobian rows index off it).
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    // Local C1 Hermite hazard back end: reprices the strip EXACTLY (each maturity is a knot) and keeps
    // forwards local. NOTE: a smooth hazard can overshoot into slightly negative forward hazard between
    // knots on a NOISY / INVERTED strip, giving a mildly non-monotone survival at the long end -- a known
    // limitation of unconstrained smooth interpolation (a positivity-constrained bootstrap is the follow-up
    // for distressed/inverted names; a plain flat-forward layout is monotone-safe but under-fits here).
    auto hz = curve::make_modular_curve<Scalar>(curve::flat_hermite(meeting_times, back_times));
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
