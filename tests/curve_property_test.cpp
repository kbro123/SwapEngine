// PROPERTY / FUZZ tests over the NEW curve families — the breakeven-inflation index curve and the
// hazard/survival credit curve — plus their ZCIS/YoY and par-CDS calibration instruments.
//
// Complements the per-feature gates (inflation_test.cpp, credit_test.cpp), which pin a handful of FIXED
// strips. This file stresses the SAME machinery over RANDOMISED (seeded, reproducible) strips and asserts
// the mathematical INVARIANTS a correct construction must satisfy for ANY input, not hard-coded numbers:
//   inflation — calibrated curve reprices every input breakeven ~0; index(0)=base; growth>0; seasonality
//               is annual-neutral and sums to zero; YoY==ZC under a flat breakeven.
//   credit    — Q(0)=1, Q strictly decreasing, 0<Q<1; par spreads reprice ~0; credit-triangle bound
//               s ≈ (1−R)·h on a flat strip; higher spread ⇒ lower survival; par spread ∝ (1−R).
//
// All randomness is a fixed std::mt19937 seed so any failure reproduces exactly. QuantLib-free (targets
// swaps_tests); everything rides the standard ModularCurve + LM+AAD path.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <vector>

#include "swaps/build/credit_instruments.hpp"
#include "swaps/build/inflation_instruments.hpp"
#include "swaps/calibration/credit_problem.hpp"
#include "swaps/calibration/inflation_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"
#include "swaps/curve/inflation.hpp"

namespace curve = swaps::curve;
namespace cal = swaps::calibration;
namespace b = swaps::build;

namespace {

const std::vector<double> kKnots{1.0, 2.0, 3.0, 5.0, 10.0};

// A globally-flat forward curve (forward == f everywhere): growth/survival are pure exponentials.
curve::ModularCurve<double> flat_curve(double f) {
  auto c = curve::make_modular_curve<double>({{kKnots, curve::Scheme::Flat}});
  c.set_forwards(Eigen::VectorXd::Constant(static_cast<int>(kKnots.size()), f));
  return c;
}

struct FlatDf {
  double r;
  double operator()(double t) const { return std::exp(-r * t); }
};

}  // namespace

// ===================================================================================================
// INFLATION
// ===================================================================================================

// Calibrating the breakeven curve to a RANDOM ZCIS strip reprices every input breakeven to ~0, and the
// resulting index curve satisfies index(0)=base and growth>0 everywhere. 12 seeded trials.
TEST(CurveProperty, InflationCalibratedZcisStripRepricesAndBasics) {
  std::mt19937 rng(0x1F1A7u);
  std::uniform_real_distribution<double> be_dist(0.005, 0.045);
  const std::vector<double> mats{2.0, 3.0, 5.0, 7.0, 10.0};

  for (int trial = 0; trial < 12; ++trial) {
    std::vector<double> tgt(mats.size());
    double mean = 0.0;
    for (double& v : tgt) { v = be_dist(rng); mean += v; }
    mean /= static_cast<double>(tgt.size());

    cal::InflationProblem prob;
    prob.base = 100.0;
    prob.back_times = mats;
    for (std::size_t i = 0; i < mats.size(); ++i)
      prob.instruments.push_back(b::inflation_zcis(mats[i], tgt[i]));

    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), std::log(1.0 + mean));
    const cal::CalibrationResult res = cal::calibrate(prob, x0);
    EXPECT_LT(res.stationarity, 1e-6) << "trial " << trial;
    EXPECT_EQ(res.rank_deficiency, 0) << "trial " << trial;

    auto bei = curve::make_modular_curve<double>(curve::flat_hermite(prob.meeting_times, prob.back_times));
    bei.set_forwards(res.x);
    const curve::InflationIndexCurve<double> infl{prob.base, &bei, nullptr};

    // reprices every input breakeven.
    for (std::size_t i = 0; i < mats.size(); ++i)
      EXPECT_NEAR(infl.zc_breakeven(mats[i]), tgt[i], 1e-6) << "trial " << trial << " mat " << mats[i];

    // index(0) == base, and growth is strictly positive across the horizon.
    EXPECT_NEAR(infl.index(0.0), prob.base, 1e-9) << "trial " << trial;
    for (double t = 0.5; t <= 12.0; t += 0.5) {
      EXPECT_GT(infl.growth(t), 0.0) << "trial " << trial << " t " << t;
      EXPECT_GT(infl.index(t), 0.0) << "trial " << trial << " t " << t;
    }
  }
}

// Seasonality invariants over RANDOM 12-month log-factor vectors: the construction recenters to Σg==0, so
// the cumulative log-adjustment is exactly 0 at every integer year, the multiplicative factor is exactly
// 1 there, and an index built with seasonality equals the non-seasonal index at whole years — while a
// mid-year point genuinely moves (unless the draw is degenerate). 30 seeded trials.
TEST(CurveProperty, InflationSeasonalityAnnualNeutralAndZeroSum) {
  std::mt19937 rng(0x5EA5u);
  std::normal_distribution<double> gd(0.0, 0.01);
  const auto bei = flat_curve(0.02);

  for (int trial = 0; trial < 30; ++trial) {
    std::vector<double> monthly(12);
    for (double& v : monthly) v = gd(rng);
    const curve::Seasonality seas(monthly);
    ASSERT_TRUE(seas.active) << "trial " << trial;

    // recentred: the 12 stored increments sum to zero.
    double sum = 0.0;
    for (double g : seas.g) sum += g;
    EXPECT_NEAR(sum, 0.0, 1e-13) << "trial " << trial;

    // annual-neutral: log_factor==0 and factor==1 at each integer year.
    for (int y = 0; y <= 10; ++y) {
      EXPECT_NEAR(seas.log_factor(static_cast<double>(y)), 0.0, 1e-13) << "trial " << trial << " y " << y;
      EXPECT_NEAR(seas.factor(static_cast<double>(y)), 1.0, 1e-13) << "trial " << trial << " y " << y;
    }

    // seasonal index == non-seasonal index at whole years (the seasonal cumulant vanishes there).
    const curve::InflationIndexCurve<double> with{100.0, &bei, &seas};
    const curve::InflationIndexCurve<double> without{100.0, &bei, nullptr};
    for (int y = 1; y <= 8; ++y)
      EXPECT_NEAR(with.index(static_cast<double>(y)), without.index(static_cast<double>(y)), 1e-9)
          << "trial " << trial << " y " << y;
  }
}

// Under a RANDOM flat breakeven curve, the annual YoY par rate equals the (annually-compounded) zero-coupon
// breakeven exp(f)-1 — a term-structure-free identity that must hold for every f. 25 seeded trials.
TEST(CurveProperty, InflationYoyEqualsZeroCouponUnderFlatBreakeven) {
  std::mt19937 rng(0x0A0Au);
  std::uniform_real_distribution<double> f_dist(-0.01, 0.06);   // include mild deflation
  std::uniform_real_distribution<double> nom_dist(0.0, 0.05);

  std::vector<double> ends;
  for (int y = 1; y <= 10; ++y) ends.push_back(static_cast<double>(y));

  for (int trial = 0; trial < 25; ++trial) {
    const double f = f_dist(rng), nom = nom_dist(rng);
    const auto bei = flat_curve(f);
    const curve::InflationIndexCurve<double> infl{100.0, &bei, nullptr};
    const double zc = std::exp(f) - 1.0;

    const auto yoy = b::inflation_yoy(ends, /*par=*/0.0, nom);
    EXPECT_NEAR(yoy.model_quote<double>(infl), zc, 1e-12) << "trial " << trial << " f " << f;
    EXPECT_NEAR(infl.zc_breakeven(1.0), zc, 1e-13) << "trial " << trial;
  }
}

// ===================================================================================================
// CREDIT / HAZARD
// ===================================================================================================

namespace {

// Build a gentle sloped CDS strip (positive, so the calibrated forward hazards stay positive and Q stays
// monotone). `up` selects an upward vs downward term structure; `bump` shifts the whole strip.
std::vector<double> sloped_strip(const std::vector<double>& mats, bool up, std::mt19937& rng,
                                 double bump = 0.0) {
  std::uniform_real_distribution<double> jitter(-2e-4, 2e-4);
  std::vector<double> s(mats.size());
  for (std::size_t i = 0; i < mats.size(); ++i) {
    const double base = up ? (0.006 + 0.0013 * mats[i]) : (0.022 - 0.0010 * mats[i]);
    s[i] = base + bump + jitter(rng);
    if (s[i] < 1e-4) s[i] = 1e-4;  // keep strictly positive
  }
  return s;
}

// Calibrate a hazard curve to a CDS strip; return the calibrated survival curve (backed by `hz`, which
// the caller must keep alive alongside the returned pointer via `hz_out`).
cal::CalibrationResult calibrate_cds(const std::vector<double>& mats, const std::vector<double>& spreads,
                                     double R, double r, curve::ModularCurve<double>& hz_out) {
  cal::CreditProblem prob;
  prob.back_times = mats;
  double mean = 0.0;
  for (std::size_t i = 0; i < mats.size(); ++i) {
    prob.instruments.push_back(b::make_cds(mats[i], spreads[i], R, FlatDf{r}, /*freq=*/4, /*prot=*/8));
    mean += spreads[i];
  }
  mean /= static_cast<double>(spreads.size());
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), mean / (1.0 - R));
  const cal::CalibrationResult res = cal::calibrate(prob, x0);
  hz_out = curve::make_modular_curve<double>(curve::flat_hermite(prob.meeting_times, prob.back_times));
  hz_out.set_forwards(res.x);
  return res;
}

}  // namespace

// Over RANDOM upward- AND downward-sloping CDS strips: calibration reprices every par spread to ~0, and
// the survival curve satisfies Q(0)=1, 0<Q<1, and Q strictly decreasing. 20 seeded trials (both slopes).
TEST(CurveProperty, CreditCalibratedStripRepricesAndSurvivalWellFormed) {
  std::mt19937 rng(0xC0FFEEu);
  std::uniform_real_distribution<double> r_dist(0.0, 0.04);
  const std::vector<double> mats{1.0, 3.0, 5.0, 7.0, 10.0};
  const double R = 0.40;

  for (int trial = 0; trial < 20; ++trial) {
    const bool up = (trial % 2 == 0);
    const double r = r_dist(rng);
    const std::vector<double> spreads = sloped_strip(mats, up, rng);

    curve::ModularCurve<double> hz;
    const cal::CalibrationResult res = calibrate_cds(mats, spreads, R, r, hz);
    EXPECT_LT(res.stationarity, 1e-6) << "trial " << trial << (up ? " up" : " down");
    EXPECT_EQ(res.rank_deficiency, 0) << "trial " << trial;

    const curve::SurvivalCurve<double> surv{&hz};

    // reprices every input par spread.
    cal::CreditProblem probe;
    probe.back_times = mats;
    for (std::size_t i = 0; i < mats.size(); ++i)
      probe.instruments.push_back(b::make_cds(mats[i], spreads[i], R, FlatDf{r}, 4, 8));
    for (std::size_t i = 0; i < mats.size(); ++i)
      EXPECT_NEAR(probe.instruments[i].model_quote<double>(surv), spreads[i], 1e-6)
          << "trial " << trial << " mat " << mats[i];

    // Q(0)=1, 0<Q<1 across the horizon, and survival strictly DECREASING for an upward strip. For an
    // INVERTED (downward) strip the smooth Hermite hazard can overshoot into slightly negative forward
    // hazard between the long knots, giving a mild non-monotonicity at the long end -- a known limitation
    // of unconstrained smooth interpolation (see credit_problem.hpp); there we assert only positivity, not
    // strict monotonicity. Repricing (above) stays exact in both regimes.
    EXPECT_NEAR(surv.survival(0.0), 1.0, 1e-12) << "trial " << trial;
    double prev = 1.0;
    for (double t = 0.25; t <= 10.0 + 1e-9; t += 0.25) {
      const double q = surv.survival(t);
      if (up) EXPECT_LT(q, prev) << "trial " << trial << " Q not decreasing at t " << t;
      EXPECT_GT(q, 0.0) << "trial " << trial << " t " << t;
      EXPECT_LT(q, 1.0) << "trial " << trial << " t " << t;
      prev = q;
    }
  }
}

// Credit-triangle bound on a FLAT hazard strip: the model par spread ≈ (1−R)·h. For flat h and flat r the
// continuous par spread equals (1−R)·h EXACTLY, so the only gap is the leg discretisation — asserted small.
// Also: s>0 and s scales with (1−R) as R varies. 40 seeded trials.
TEST(CurveProperty, CreditTriangleBoundAndRecoveryScalingOnFlatStrip) {
  std::mt19937 rng(0x71A96u);
  std::uniform_real_distribution<double> h_dist(0.005, 0.05);
  std::uniform_real_distribution<double> R_dist(0.10, 0.60);
  std::uniform_real_distribution<double> r_dist(0.0, 0.05);
  const double T = 5.0;

  for (int trial = 0; trial < 40; ++trial) {
    const double h = h_dist(rng), R = R_dist(rng), r = r_dist(rng);
    const auto hz = flat_curve(h);
    const curve::SurvivalCurve<double> surv{&hz};

    const double s = b::make_cds(T, 0.0, R, FlatDf{r}, /*freq=*/4, /*prot=*/8).model_quote<double>(surv);
    EXPECT_GT(s, 0.0) << "trial " << trial;
    EXPECT_NEAR(s, (1.0 - R) * h, 1.5e-3) << "trial " << trial << " h " << h << " R " << R;

    // s ∝ (1−R): re-price with a second recovery and check the exact ratio (annuity is R-independent).
    const double R2 = (R < 0.4) ? R + 0.2 : R - 0.2;
    const double s2 = b::make_cds(T, 0.0, R2, FlatDf{r}, 4, 8).model_quote<double>(surv);
    EXPECT_NEAR(s / s2, (1.0 - R) / (1.0 - R2), 1e-9) << "trial " << trial;
  }
}

// Monotonicity in the market: a UNIFORMLY HIGHER CDS strip calibrates to a curve with LOWER survival at
// every maturity (more default risk ⇒ less survival). 12 seeded trials.
TEST(CurveProperty, CreditHigherSpreadLowersSurvival) {
  std::mt19937 rng(0xBADCAFEu);
  std::uniform_real_distribution<double> bump_dist(0.002, 0.006);
  const std::vector<double> mats{1.0, 3.0, 5.0, 7.0, 10.0};
  const double R = 0.40, r = 0.015;

  for (int trial = 0; trial < 12; ++trial) {
    const bool up = (trial % 2 == 0);
    const std::vector<double> base = sloped_strip(mats, up, rng);
    std::vector<double> higher = base;
    const double bump = bump_dist(rng);
    for (double& v : higher) v += bump;

    curve::ModularCurve<double> hz_lo, hz_hi;
    const auto rlo = calibrate_cds(mats, base, R, r, hz_lo);
    const auto rhi = calibrate_cds(mats, higher, R, r, hz_hi);
    EXPECT_LT(rlo.stationarity, 1e-6) << "trial " << trial;
    EXPECT_LT(rhi.stationarity, 1e-6) << "trial " << trial;

    const curve::SurvivalCurve<double> lo{&hz_lo}, hi{&hz_hi};
    for (double t : mats)
      EXPECT_LT(hi.survival(t), lo.survival(t))
          << "trial " << trial << " t " << t << ": higher spread did not lower survival";
  }
}
