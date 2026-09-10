// Inflation gate: the CPI index curve + ZCIS/YoY inflation-swap instruments (audit breadth item #3).
//
// Covered, each a place a wiring bug would hide:
//   1. a ZCIS reprices its own breakeven EXACTLY off a flat breakeven curve (residual == 0);
//   2. on a FLAT breakeven curve a YoY par rate equals the zero-coupon rate — and the leg-composed
//      QuoteKind::ParRate calibration::Instrument agrees with the dedicated YoY model quote to the bit;
//   3. seasonality shifts the MONTHLY index but leaves an ANNUAL zero-coupon breakeven (and integer-year
//      index points) untouched;
//   4. calibrating the breakeven curve to a STRIP of ZCIS recovers the input breakevens.
//
// QuantLib-free; header-only engine. Everything runs through the standard curve + LM+AAD machinery.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/build/inflation_instruments.hpp"
#include "swaps/calibration/inflation_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/inflation.hpp"

namespace curve = swaps::curve;
namespace cal = swaps::calibration;
namespace b = swaps::build;

namespace {

// A globally-flat breakeven curve: instantaneous forward breakeven == f everywhere, so growth(t)=exp(f·t).
curve::ModularCurve<double> flat_bei(double f, const std::vector<double>& knots) {
  auto c = curve::make_modular_curve<double>({{knots, curve::Scheme::Flat}});
  c.set_forwards(Eigen::VectorXd::Constant(static_cast<int>(knots.size()), f));
  return c;
}

const std::vector<double> kKnots{1.0, 2.0, 3.0, 5.0, 10.0};

}  // namespace

// 1. ZCIS reprices its breakeven exactly.
TEST(Inflation, ZcisRepricesBreakevenExactly) {
  const double f = 0.02;
  const auto bei = flat_bei(f, kKnots);
  const curve::InflationIndexCurve<double> infl{100.0, &bei, nullptr};

  const double T = 5.0;
  const double be = std::exp(f) - 1.0;  // growth=exp(fT) => (exp(fT))^(1/T)-1 = exp(f)-1
  EXPECT_NEAR(infl.zc_breakeven(T), be, 1e-13);
  EXPECT_NEAR(infl.index(T), 100.0 * std::exp(f * T), 1e-9);

  const auto ins = b::inflation_zcis(T, be);
  EXPECT_NEAR(ins.model_quote<double>(infl), be, 1e-13);
  EXPECT_NEAR(ins.residual<double>(infl), 0.0, 1e-13);
}

// 2. Flat breakeven => YoY par rate == the ZC rate; leg-composed ParRate instrument agrees exactly.
TEST(Inflation, FlatBreakevenYoyEqualsZeroCoupon) {
  const double f = 0.025, nom = 0.03;
  const auto bei = flat_bei(f, kKnots);
  const curve::InflationIndexCurve<double> infl{100.0, &bei, nullptr};
  const double zc = std::exp(f) - 1.0;

  std::vector<double> ends;
  for (int y = 1; y <= 10; ++y) ends.push_back(static_cast<double>(y));

  const auto yoy = b::inflation_yoy(ends, /*par=*/0.0, nom);
  EXPECT_NEAR(yoy.model_quote<double>(infl), zc, 1e-12);
  EXPECT_NEAR(infl.zc_breakeven(1.0), zc, 1e-13);

  // Leg-composed YoY as a QuoteKind::ParRate calibration::Instrument. Nominal is a flat ModularCurve so a
  // single role->curve map has one type; role 0 = nominal discount, role 1 = the breakeven forecast curve.
  const auto nomc = flat_bei(nom, kKnots);
  const auto par = b::yoy_par_swap_instrument(/*bei_role=*/1, /*nom_role=*/0, ends, /*market=*/0.0);
  const curve::ModularCurve<double>* roles[2] = {&nomc, &bei};
  const auto C = [&](int r) -> const curve::ModularCurve<double>& { return *roles[r]; };
  const double q_legs = cal::instrument_model_quote<double>(par, C);
  EXPECT_NEAR(q_legs, yoy.model_quote<double>(infl), 1e-12);
  EXPECT_NEAR(q_legs, zc, 1e-12);
}

// 3. Seasonality shifts the monthly index but not the annual ZC (or integer-year index points).
TEST(Inflation, SeasonalityShiftsMonthlyNotAnnual) {
  const double f = 0.02;
  const auto bei = flat_bei(f, kKnots);

  std::vector<double> monthly(12);
  for (int m = 0; m < 12; ++m) monthly[m] = (m < 6) ? 0.01 : -0.01;  // sums to 0; nonzero mid-year cumulant
  const curve::Seasonality seas(monthly);
  ASSERT_TRUE(seas.active);

  const curve::InflationIndexCurve<double> with{100.0, &bei, &seas};
  const curve::InflationIndexCurve<double> without{100.0, &bei, nullptr};

  // Annual zero-coupon breakeven: identical.
  EXPECT_NEAR(with.zc_breakeven(5.0), without.zc_breakeven(5.0), 1e-13);
  // Integer-year index level: identical (seasonal cumulant returns to 0 at whole years).
  EXPECT_NEAR(with.index(3.0), without.index(3.0), 1e-10);
  // Mid-year monthly index: genuinely shifted.
  EXPECT_GT(std::abs(with.index(1.5) - without.index(1.5)), 1e-4);
}

// 4. Calibrating the breakeven curve to a strip of ZCIS recovers the input breakevens.
TEST(Inflation, CalibrateZcisStripRecoversBreakevens) {
  const std::vector<double> mats{2.0, 3.0, 5.0, 7.0, 10.0};
  const std::vector<double> tgt{0.021, 0.022, 0.024, 0.025, 0.026};

  cal::InflationProblem prob;
  prob.base = 100.0;
  prob.back_times = mats;
  for (std::size_t i = 0; i < mats.size(); ++i)
    prob.instruments.push_back(b::inflation_zcis(mats[i], tgt[i]));

  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), std::log(1.02));
  const cal::CalibrationResult res = cal::calibrate(prob, x0);
  EXPECT_LT(res.stationarity, 1e-8);
  EXPECT_EQ(res.rank_deficiency, 0);

  auto bei = curve::make_modular_curve<double>(curve::flat_hermite(prob.meeting_times, prob.back_times));
  bei.set_forwards(res.x);
  const curve::InflationIndexCurve<double> infl{100.0, &bei, nullptr};
  for (std::size_t i = 0; i < mats.size(); ++i)
    EXPECT_NEAR(infl.zc_breakeven(mats[i]), tgt[i], 1e-8);
}

// 5. (E3-F4, 2026-09-10) The seasonal is anchored to the CALENDAR: `phase` is the calendar position (months
// from January) of curve time 0 -- the base reference month. With it, a whole-year maturity cancels for ANY
// phase, and the seasonal at t is the cumulant of the calendar month t years after the base month, relative to
// the base month. phase == 0 is byte-identical to the legacy January anchor.
TEST(Inflation, SeasonalityPhaseAnchorsToTheReferenceMonth) {
  std::vector<double> monthly{0.006, 0.004, 0.003, 0.002, 0.001, -0.001, -0.002, -0.003, -0.004, -0.003, -0.002, -0.001};
  curve::Seasonality jan(monthly), jul(monthly);
  jul.phase = 6.0;  // t = 0 is a July reference month
  ASSERT_TRUE(jan.active && jul.active);
  for (double y : {1.0, 2.0, 3.0, 7.0}) {
    EXPECT_NEAR(jan.log_factor(y), 0.0, 1e-15);
    EXPECT_NEAR(jul.log_factor(y), 0.0, 1e-15) << "a whole year cancels for any phase";
  }
  EXPECT_DOUBLE_EQ(jul.log_factor(0.0), 0.0);
  // Half a year after July is January: the seasonal relative to July is C(Jan) - C(Jul) = -C(Jul).
  EXPECT_NEAR(jul.log_factor(0.5), -jan.cumulant(6.0), 1e-15);
  // and 2.5 years after July is also January
  EXPECT_NEAR(jul.log_factor(2.5), -jan.cumulant(6.0), 1e-15);
  // The January anchor reproduces the legacy formula exactly for a range of t.
  for (double t : {0.1, 0.37, 1.5, 2.25, 4.9}) {
    const double u = t - std::floor(t);
    EXPECT_DOUBLE_EQ(jan.log_factor(t), jan.cumulant(u * 12.0));
  }
  // A fractional phase (a daily-interpolated index, mid-month base): continuous in the phase.
  curve::Seasonality mid(monthly);
  mid.phase = 6.5;
  EXPECT_NEAR(mid.log_factor(1.0), 0.0, 1e-15);
  EXPECT_NEAR(mid.log_factor(0.25), jan.cumulant(9.5) - jan.cumulant(6.5), 1e-15);
}
