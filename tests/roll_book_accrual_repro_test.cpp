// E5 taxonomy: T6 regression (fails on the reverted bug)
// PN2 REPRODUCTION (2026-09-14). calibration::roll_book ages a book by dt for pnl_explain's roll leg and the exposure
// profile. With shift = true it re-times pay, the observation brackets and the FX reset, but it leaves
//   (a) an MtM coupon's ACCRUAL period (accrual_start / accrual_end) -- where xccy_mtm_leg_pv books the notional
//       exchanges -DF(start) +DF(end) -- at the OLD times, and
//   (b) a position's principal_flows (dated principal exchanges) at the OLD times, never dropping a settled one.
// Expected values are written by hand from the inputs: every future time moves by -dt, an accrual start that falls
// before the new valuation date becomes NEGATIVE (the initial exchange has settled; cashflows.hpp reads s < 0 so),
// and a principal flow dated on/before dt is gone.
#include <gtest/gtest.h>

#include "swaps/calibration/pnl_explain.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

px::FloatCoupon mtm_coupon(double s, double e) {
  px::FloatCoupon c;
  c.obs.sub_start = {s};
  c.obs.sub_end = {e};
  c.obs.tau_index = e - s;
  c.tau_pay = e - s;
  c.pay = e + 2.0 / 365.0;
  c.accrual_set = true;
  c.accrual_start = s;
  c.accrual_end = e;
  return c;
}

pf::MultiCurveBook xccy_book() {
  pf::MultiCurveBook book;
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.float_coupons = {mtm_coupon(0.25, 0.50), mtm_coupon(0.50, 0.75), mtm_coupon(0.75, 1.00)};
  p.mtm_coupons = {mtm_coupon(0.25, 0.50), mtm_coupon(0.50, 0.75), mtm_coupon(0.75, 1.00)};
  p.principal_flows = {{0.25, -1.0}, {1.00, 1.0}};
  p.fx_spot = 1.10;
  book.positions.push_back(p);
  return book;
}

}  // namespace

TEST(RollBookAccrualRepro, ShiftMovesTheMtmAccrualPeriodWithTheValuationDate) {
  const double dt = 0.40;  // inside the first period (0.25, 0.50): its start has settled
  const pf::MultiCurveBook rolled = cal::roll_book(xccy_book(), dt, /*shift=*/true);
  ASSERT_EQ(rolled.positions.size(), 1u);
  const auto& m = rolled.positions[0].mtm_coupons;
  ASSERT_EQ(m.size(), 3u) << "premise: every coupon pays after dt";
  const double want_start[] = {0.25 - 0.40, 0.50 - 0.40, 0.75 - 0.40};
  const double want_end[] = {0.50 - 0.40, 0.75 - 0.40, 1.00 - 0.40};
  for (std::size_t i = 0; i < m.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_TRUE(m[i].accrual_set);
    EXPECT_DOUBLE_EQ(m[i].accrual_start, want_start[i]);
    EXPECT_DOUBLE_EQ(m[i].accrual_end, want_end[i]);
  }
  EXPECT_LT(m[0].accrual_start, 0.0) << "the first period's initial exchange settled before the new valuation date";
}

TEST(RollBookAccrualRepro, ShiftMovesAndDropsPrincipalFlows) {
  const pf::MultiCurveBook rolled = cal::roll_book(xccy_book(), 0.40, /*shift=*/true);
  ASSERT_EQ(rolled.positions.size(), 1u);
  const auto& flows = rolled.positions[0].principal_flows;
  ASSERT_EQ(flows.size(), 1u) << "the initial exchange at t = 0.25 settled before dt = 0.40";
  EXPECT_DOUBLE_EQ(flows[0].first, 1.00 - 0.40);
  EXPECT_DOUBLE_EQ(flows[0].second, 1.0);
}

TEST(RollBookAccrualRepro, ControlCarryLegKeepsTimes) {
  // shift = false (the carry leg) keeps every time and only drops paid coupons -- unchanged by the fix.
  const pf::MultiCurveBook kept = cal::roll_book(xccy_book(), 0.40, /*shift=*/false);
  ASSERT_EQ(kept.positions.size(), 1u);
  EXPECT_DOUBLE_EQ(kept.positions[0].mtm_coupons[0].accrual_start, 0.25);
  EXPECT_DOUBLE_EQ(kept.positions[0].mtm_coupons[0].accrual_end, 0.50);
}

TEST(RollBookAccrualRepro, CarryLegDropsTheSettledPrincipalExchangeAndKeepsTheRestsTime) {
  // The carry leg drops a PAID coupon; a principal exchange paid on or before dt is paid too, so keeping it would
  // count cash the holder no longer has. The surviving exchange keeps its time (no shift in the carry leg).
  const pf::MultiCurveBook kept = cal::roll_book(xccy_book(), 0.40, /*shift=*/false);
  ASSERT_EQ(kept.positions.size(), 1u);
  const auto& flows = kept.positions[0].principal_flows;
  ASSERT_EQ(flows.size(), 1u) << "the initial exchange at t = 0.25 settled before dt = 0.40";
  EXPECT_DOUBLE_EQ(flows[0].first, 1.00);
  EXPECT_DOUBLE_EQ(flows[0].second, 1.0);
}
