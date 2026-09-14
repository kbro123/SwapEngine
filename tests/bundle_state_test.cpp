// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// calibration/bundle_state.hpp and portfolio/xccy_fx_scaled.hpp (E7 stage 5.1): sampling and forking a calibrated
// state, valuing a book at any state, and a book under an FX move -- the pieces scenario / scenario_grid / var each
// wrote by hand. Header-only (swaps_tests), so tools/mutate.py reaches them; the verbs end to end stay pinned byte for
// byte by tests/scenario_golden_test.cpp.
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/calibration/bundle_state.hpp"
#include "swaps/portfolio/xccy_fx_scaled.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

cal::BundleCurveSpec flat_curve(int currency) {
  cal::BundleCurveSpec s;
  s.currency = currency;
  crv::CurveModule m;
  m.scheme = crv::Scheme::Flat;
  m.knots = {1.0};
  s.regions = {m};
  return s;
}
px::FloatCoupon float_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
px::FixedCoupon fixed_coupon(double a, double b) {
  px::FixedCoupon c;
  c.pay = b;
  c.tau = b - a;
  return c;
}
pf::MultiCurveBook::Position one_period_swap(double fixed_rate) {
  pf::MultiCurveBook::Position p;
  p.notional = 1e6;
  p.fixed_rate = fixed_rate;
  p.fwd_curve = 0;
  p.disc_curve = 0;
  p.fixed_curve = 0;
  p.float_coupons = {float_coupon(0.0, 1.0)};
  p.fixed_coupons = {fixed_coupon(0.0, 1.0)};
  return p;
}
pf::MultiCurveBook::Position xccy_position() {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = 3e7;
  p.fwd_curve = 0;
  p.disc_curve = 0;
  p.float_coupons = {float_coupon(0.0, 1.0), float_coupon(1.0, 2.0)};
  p.mtm_coupons = p.float_coupons;
  for (auto& c : p.mtm_coupons) c.spread = 0.005;
  p.mtm_fwd_curve = 1;
  p.mtm_disc_curve = 1;
  p.mtm_reset_num = 1;
  p.mtm_reset_den = 0;
  p.fx_spot = 1.10;
  return p;
}

}  // namespace

TEST(SampleBundleCurves, AFlatCurveHasClosedFormDiscountsZerosAndForwards) {
  cal::BundleProblem p;
  p.curves = {flat_curve(7)};
  const Eigen::VectorXd x = Eigen::VectorXd::Constant(1, 0.03);
  const std::vector<cal::CurveSample> s = cal::sample_bundle_curves(p, x, {0.0, 0.5, 4.0});
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(s[0].currency, 7);
  EXPECT_EQ(s[0].t, (std::vector<double>{0.0, 0.5, 4.0}));
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(s[0].discount[k], std::exp(-0.03 * s[0].t[k]), 1e-15);
    EXPECT_NEAR(s[0].forward[k], 0.03, 1e-15);
    EXPECT_NEAR(s[0].zero[k], 0.03, 1e-15) << "at t = 0 the zero is the forward, never 0/0";
  }
  EXPECT_THROW((void)cal::sample_bundle_curves(p, Eigen::VectorXd::Zero(2), {1.0}), std::invalid_argument);
}

TEST(ShiftInterpForwards, AddsEachCurvesShiftToItsInterpolationForwardsAndNotItsTurns) {
  cal::BundleProblem p;
  cal::BundleCurveSpec turned = flat_curve(0);
  turned.regions[0].scheme = crv::Scheme::Linear;
  turned.regions[0].knots = {1.0, 2.0, 5.0};
  turned.turns = {px::Turn{0.10, 0.20}};
  p.curves = {turned, flat_curve(1)};
  ASSERT_EQ(p.n_knots(), 5);
  Eigen::VectorXd x(5);
  x << 0.01, 0.02, 0.03, 0.004, 0.05;  // curve 0: three forwards then its turn's jump; curve 1: one forward

  const Eigen::VectorXd up = cal::shift_interp_forwards(p, x, {1e-4, 0.0});
  EXPECT_EQ(up[0], 0.01 + 1e-4);
  EXPECT_EQ(up[1], 0.02 + 1e-4);
  EXPECT_EQ(up[2], 0.03 + 1e-4);
  EXPECT_EQ(up[3], 0.004) << "a turn's jump is not a level";
  EXPECT_EQ(up[4], 0.05) << "an unshifted curve is untouched";

  const Eigen::VectorXd down = cal::shift_interp_forwards(p, x, {0.0, -2.5e-3});
  EXPECT_TRUE(down.head(4) == x.head(4));
  EXPECT_EQ(down[4], 0.05 - 2.5e-3);

  EXPECT_THROW((void)cal::shift_interp_forwards(p, x, {1e-4}), std::invalid_argument);
  EXPECT_THROW((void)cal::shift_interp_forwards(p, Eigen::VectorXd::Zero(4), {1e-4, 0.0}), std::invalid_argument);
}

TEST(BookValueAt, AOnePeriodSwapOnAFlatCurveHasItsClosedForm) {
  cal::BundleProblem p;
  p.curves = {flat_curve(0)};
  const Eigen::VectorXd x = Eigen::VectorXd::Constant(1, 0.03);
  EXPECT_EQ(cal::book_value_at(pf::MultiCurveBook{}, p, x), 0.0);

  const double df1 = std::exp(-0.03);
  // Float leg = DF(0) - DF(1) = 1 - df1 per unit notional; the fixed leg's annuity is df1. So the value is linear in
  // the fixed rate with slope notional * df1, and zero at the par rate (1 - df1) / df1.
  const double v2 = cal::book_value_at(pf::MultiCurveBook{{one_period_swap(0.02)}}, p, x);
  const double v5 = cal::book_value_at(pf::MultiCurveBook{{one_period_swap(0.05)}}, p, x);
  EXPECT_NEAR(std::abs(v2 - v5), 1e6 * 0.03 * df1, 1e-6);
  EXPECT_NEAR(cal::book_value_at(pf::MultiCurveBook{{one_period_swap((1.0 - df1) / df1)}}, p, x), 0.0, 1e-8);
  EXPECT_THROW((void)cal::book_value_at(pf::MultiCurveBook{{one_period_swap(0.02)}}, p, Eigen::VectorXd::Zero(2)),
               std::invalid_argument);
}

TEST(XccyFxScaled, ScalesOnlyTheXccyPositionsFxSpot) {
  const pf::MultiCurveBook book{{one_period_swap(0.02), xccy_position()}};
  const pf::MultiCurveBook scaled = pf::xccy_fx_scaled(book, 1.05);
  EXPECT_EQ(scaled.positions[0].fx_spot, book.positions[0].fx_spot) << "a swap has no FX-reset notional";
  EXPECT_EQ(scaled.positions[1].fx_spot, 1.10 * 1.05);
  EXPECT_EQ(book.positions[1].fx_spot, 1.10) << "the input is untouched";
  EXPECT_EQ(pf::xccy_fx_scaled(book, 1.0).positions[1].fx_spot, 1.10);
}

TEST(XccyFxScaledBooks, CachesOneCompiledBookPerFactorAndPricesTheScaledBook) {
  cal::BundleProblem p;
  p.curves = {flat_curve(0), flat_curve(1)};
  Eigen::VectorXd x(2);
  x << 0.03, 0.02;
  const pf::MultiCurveBook book{{one_period_swap(0.02), xccy_position()}};
  pf::XccyFxScaledBooks books(p.curves, book);

  const pf::CompiledMultiCurveBook& one = books.at(1.0);
  EXPECT_EQ(&books.at(1.0), &one);
  EXPECT_EQ(&books.at(1.0 + 1e-14), &one) << "factors are keyed to 1e-12";
  const pf::CompiledMultiCurveBook& up = books.at(1.05);
  EXPECT_NE(&up, &one);

  EXPECT_EQ(one.npv(x), pf::CompiledMultiCurveBook(p.curves, book).npv(x));
  EXPECT_EQ(up.npv(x), pf::CompiledMultiCurveBook(p.curves, pf::xccy_fx_scaled(book, 1.05)).npv(x));
  EXPECT_NE(up.npv(x), one.npv(x)) << "the FX move reaches the xccy position";
  EXPECT_EQ(books.book().positions[1].fx_spot, 1.10);
}
