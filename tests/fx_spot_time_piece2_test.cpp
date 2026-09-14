// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T6 regression (fails on the reverted bug)
// O-X3 fx_spot_time, piece 2 (2026-09-14; owner answers). The builder quotes fx_spot for the product's SPOT date and fixes each MtM
// period's FX fx_reset_fixing_lag business days before its start (CARR 2021: Mon 7 June for the period from Wed 9 June), floored
// at the value date; a PASSED fixing makes the coupon seasoned (its notional is a number: reset_fx is required) while a future one
// still prices off the forward; roll_book ages the fixing time and keeps fx_spot_time (an unchanged market is the same spot quote
// for the new spot date). Expected dates are the CARR example's; the premises are asserted.
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/pnl_explain.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;
namespace tol = swaps::tol;

namespace {
b::Date d(const char* iso) { return b::Date::from_iso(iso); }
struct Flat {  // continuously-compounded flat curve: a closed-form reference (the kernels read integral and discount)
  double r;
  double integral(double t) const { return r * t; }
  double forward(double) const { return r; }
  double discount(double t) const { return std::exp(-r * t); }
};
}  // namespace

TEST(FxSpotTimePiece2, TheBuilderQuotesTheSpotDateAndFixesEachPeriodTwoBusinessDaysBefore) {
  const b::XccyConv x = b::xccy_conv("EURUSD");
  ASSERT_EQ(x.fx_reset_lag, 2) << "premise";
  const b::Date trade = d("2021-03-05");
  const cal::Instrument ins = b::xccy_mtm_basis(trade, x, b::resolve("3Y", trade, x.calendar, x.bdc, x.spot_lag), 0, 1, 2, 1.25, 0.0);
  EXPECT_DOUBLE_EQ(ins.mtm.fx_spot_time, b::curve_time(trade, d("2021-03-09"))) << "fx_spot is the quote for spot (Tue 9 March)";
  const auto& m = ins.mtm.coupons;
  ASSERT_GE(m.size(), 2u);
  for (const auto& c : m) {
    EXPECT_TRUE(c.fx_fixing_set);
    EXPECT_LT(c.reset_time, 0.0) << "the forward is still read at the period start";
  }
  EXPECT_DOUBLE_EQ(m[0].fx_fixing_time, 0.0) << "the first period fixes on the trade date";
  EXPECT_DOUBLE_EQ(m[1].fx_fixing_time, b::curve_time(trade, d("2021-06-07"))) << "CARR: FX reset Monday 7 June 2021";
  EXPECT_FALSE(px::mtm_coupon_is_seasoned(m[1])) << "a future fixing is not seasoned";
  // a weekend value date: RAW spot Tue 9 March minus 2 BD = Fri 5 March, before Saturday 6 March -> floored at the value date
  const b::Date sat = d("2021-03-06");
  ASSERT_FALSE(b::is_business_day(x.calendar, sat)) << "premise";
  const cal::Instrument w = b::xccy_mtm_basis(sat, x, b::resolve("1Y", sat, x.calendar, x.bdc, x.spot_lag), 0, 1, 2, 1.25, 0.0);
  EXPECT_DOUBLE_EQ(w.mtm.coupons.front().fx_fixing_time, 0.0);
  EXPECT_FALSE(px::mtm_coupon_is_seasoned(w.mtm.coupons.front()));
}

// Valued between a period's FX fixing and its start (the CARR 7 June .. 9 June window): the notional is KNOWN.
TEST(FxSpotTimePiece2, APassedFixingNeedsTheFixedRateAndAFutureOneIsTheForward) {
  const Flat num{0.02}, den{0.04}, fund{0.04};
  px::FloatCoupon c;
  c.obs.sub_start = {2.0 / 365.0};
  c.obs.sub_end = {2.0 / 365.0 + 0.25};
  c.obs.tau_index = 0.25;
  c.pay = 2.0 / 365.0 + 0.25 + 2.0 / 365.0;
  c.tau_pay = 0.25;
  c.spread = 0.01;
  c.accrual_set = true;
  c.accrual_start = c.obs.sub_start.front();
  c.accrual_end = c.obs.sub_end.back();
  c.fx_fixing_set = true;
  c.fx_fixing_time = -1.0 / 365.0;  // fixed yesterday
  ASSERT_TRUE(px::mtm_coupon_is_seasoned(c));
  EXPECT_THROW(px::xccy_mtm_leg_pv<double>(std::vector<px::FloatCoupon>{c}, 1.10, fund, fund, num, den), std::runtime_error)
      << "a curve-implied forward for a known fixing is not a price";
  px::FloatCoupon fixed = c;
  fixed.reset_fx = 1.1234;
  const double v = px::float_coupon_pv<double>(fixed, fund, fund) + fund.discount(c.accrual_end) - fund.discount(c.accrual_start);
  EXPECT_NEAR(px::xccy_mtm_leg_pv<double>(std::vector<px::FloatCoupon>{fixed}, 1.10, fund, fund, num, den), 1.1234 * v,
              tol::literal * std::abs(v));
  px::FloatCoupon future = c;
  future.fx_fixing_time = 1.0 / 365.0;  // fixes tomorrow: still the forward
  px::FloatCoupon unset = c;
  unset.fx_fixing_set = false;
  EXPECT_FALSE(px::mtm_coupon_is_seasoned(future));
  EXPECT_EQ(px::xccy_mtm_leg_pv<double>(std::vector<px::FloatCoupon>{future}, 1.10, fund, fund, num, den),
            px::xccy_mtm_leg_pv<double>(std::vector<px::FloatCoupon>{unset}, 1.10, fund, fund, num, den))
      << "a future fixing time does not move the forward";
}

TEST(FxSpotTimePiece2, RollBookAgesTheFixingTimeAndKeepsTheSpotTime) {
  pf::MultiCurveBook book;
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  px::FloatCoupon c;
  c.obs.sub_start = {0.25};
  c.obs.sub_end = {0.50};
  c.obs.tau_index = 0.25;
  c.pay = 0.51;
  c.tau_pay = 0.25;
  c.accrual_set = true;
  c.accrual_start = 0.25;
  c.accrual_end = 0.50;
  c.fx_fixing_set = true;
  c.fx_fixing_time = 0.245;
  p.float_coupons = {c};
  p.mtm_coupons = {c};
  p.fx_spot = 1.1;
  p.fx_spot_time = 2.0 / 365.0;
  book.positions.push_back(p);
  const pf::MultiCurveBook rolled = cal::roll_book(book, 0.30, /*shift=*/true);
  ASSERT_EQ(rolled.positions.size(), 1u);
  EXPECT_DOUBLE_EQ(rolled.positions[0].fx_spot_time, 2.0 / 365.0) << "an unchanged market is the same spot quote for the new spot date";
  EXPECT_DOUBLE_EQ(rolled.positions[0].mtm_coupons[0].fx_fixing_time, 0.245 - 0.30) << "the fixing ages (unfloored: passed = known)";
  EXPECT_TRUE(px::mtm_coupon_is_seasoned(rolled.positions[0].mtm_coupons[0]));
}

TEST(FxSpotTimePiece2, TheFxForwardBuilderCarriesTheSpotTime) {
  EXPECT_EQ(b::fx_forward(0, 1, 1.1, 0.5, 1.12).fx_spot_time, 0.0);
  EXPECT_EQ(b::fx_forward(0, 1, 1.1, 0.5, 1.12, 2.0 / 365.0).fx_spot_time, 2.0 / 365.0);
}
