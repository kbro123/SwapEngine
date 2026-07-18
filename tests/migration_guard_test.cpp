// Stage 4 adoption guard: the reference market is built as GENERIC Instruments
// (reference_curve.hpp build_problem / build_square_problem). This pins the DATE fact that made the
// migration byte-safe, so a future change that breaks it fails loudly:
//
//   The generic overnight extractor reads valueDates().front()/back() where the retired legacy
//   extractor read accrualStartDate()/accrualEndDate(). On this market they coincide (no realized
//   prefix on any forward-starting coupon), which is the ONLY reason the generic swap reprices the old
//   OisSwap form bit-for-bit -- assert it on every coupon.
//
// (The bit-exact legacy-vs-generic residual comparison that lived here was removed with the legacy
// extractors; the equivalence it proved is now permanent -- there is only the generic path.)

#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include "reference_curve.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;

TEST(MigrationGuard, OvernightValueDatesCoincideWithAccrualDatesOnEveryCoupon) {
  RelinkableHandle<YieldTermStructure> h;
  const rb::Market mk = rb::build_market(h);

  int coupons = 0;
  for (const auto& swap : mk.swaps) {
    for (const auto& cf : swap->overnightLeg()) {
      auto c = ext::dynamic_pointer_cast<OvernightIndexedCoupon>(cf);
      ASSERT_TRUE(c) << "overnight leg cashflow is not an OvernightIndexedCoupon";
      // No realized prefix on this market => the generic extractor's forward sub-period is exactly
      // [valueDates().front(), valueDates().back()], which must equal the old accrual span.
      EXPECT_EQ(c->valueDates().front(), c->accrualStartDate())
          << "generic sub_start (valueDates.front) must equal accrualStart";
      EXPECT_EQ(c->valueDates().back(), c->accrualEndDate())
          << "generic sub_end (valueDates.back) must equal accrualEnd";
      ++coupons;
    }
  }
  EXPECT_GT(coupons, 0) << "expected at least one overnight coupon to check";
}
