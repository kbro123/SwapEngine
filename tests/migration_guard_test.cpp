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

#include <Eigen/Core>

#include <vector>

#include "reference_curve.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/pricing/cashflows.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cv = swaps::curve;
namespace px = swaps::pricing;

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

// Phase 0 multi-currency plumbing: the constant FX `scale` on a coupon/fixed-coupon must multiply its
// PV EXACTLY by scale, and (the regression-safety invariant) scale == 1.0 must be byte-identical to a
// coupon with no scale set. A power-of-2 scale keeps the linear-scaling assertion bit-exact
// (round(2x) == 2 round(x) in binary FP, including across the summed annuity), so EXPECT_EQ is honest.
TEST(FxScale, CouponAndAnnuityScaleExactlyAndDefaultIsIdentity) {
  auto curve = cv::make_calibration_curve<double>({0.25, 0.5}, {1, 2, 5, 10});
  Eigen::VectorXd x(6);
  x << 0.043, 0.044, 0.045, 0.046, 0.047, 0.05;
  curve.set_forwards(x);

  // A plain OIS-shape float coupon (one sub-period, tau_pay == tau_index, no spread/realized) -- the
  // shape that takes the fused fast path at scale == 1 and the general path at scale != 1.
  px::FloatCoupon fc;
  fc.obs.sub_start = {1.0};
  fc.obs.sub_end = {2.0};
  fc.obs.tau_index = 1.0;
  fc.pay = 2.0;
  fc.tau_pay = 1.0;
  const double pv1 = px::float_coupon_pv<double>(fc, curve, curve);  // scale defaults to 1.0
  fc.scale = 2.0;
  const double pv2 = px::float_coupon_pv<double>(fc, curve, curve);
  EXPECT_EQ(pv2, 2.0 * pv1) << "FX scale must multiply the coupon PV exactly by scale";
  EXPECT_NE(pv1, 0.0) << "the test coupon must have a non-trivial PV";

  // Annuity: a summed leg, so a power-of-2 scale is required for the sum to scale bit-exactly.
  std::vector<px::FixedCoupon> leg{{1.0, 1.0}, {2.0, 1.0}};
  const double a1 = px::annuity<double>(leg, curve);
  for (auto& c : leg) c.scale = 2.0;
  const double a2 = px::annuity<double>(leg, curve);
  EXPECT_EQ(a2, 2.0 * a1) << "FX scale must multiply the annuity exactly by scale";
  EXPECT_NE(a1, 0.0) << "the test annuity must be non-trivial";
}
