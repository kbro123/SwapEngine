// Design §2 gate: the GENERIC cashflow model's degrees of freedom.
//
// The generic kernel is validated against QuantLib to ~1e-16 by extract_test / pricing_test (via the
// reference market). This file pins what the generic model can express that a plain OIS coupon cannot:
// per-sub-period weights, an additive spread, tau_pay != tau_index (the k-form), and a fully fixed
// coupon summed with live siblings under AAD (the empty-derivative trap).
//
// (The legacy OisSwap/CompoundedFuture/AveragedFuture reduction tests that used to live here were
// removed with those structs -- the generic kernel is now the ONLY kernel.)

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace rm = swaps::refmkt;
namespace px = swaps::pricing;

namespace {

struct Generic : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::Curve curve = rb::reference_curve(mk);
};

}  // namespace

// ---- The new degrees of freedom ----------------------------------------------------------------

TEST_F(Generic, ExplicitUnitWeightsMatchEmptyWeights) {
  // A real averaging observation (per-business-day sub-periods, realized prefix) from the market.
  px::RateObservation implicit = rb::avg_future_obs(mk, mk.futures[0]);
  ASSERT_FALSE(implicit.sub_start.empty());
  px::RateObservation explicit_ones = implicit;
  explicit_ones.weight.assign(implicit.sub_start.size(), 1.0);
  // x * 1.0 is exact in IEEE-754, so the two paths must agree bitwise -- which is what licenses
  // "empty weight vector => all ones" as a pure optimisation rather than a second code path.
  EXPECT_EQ(px::rate<double>(explicit_ones, curve), px::rate<double>(implicit, curve));
}

TEST_F(Generic, WeightsScaleTheirSubPeriod) {
  px::RateObservation o;
  o.sub_start = {0.25, 0.50};
  o.sub_end = {0.50, 0.75};
  o.tau_index = 0.5;
  const double unweighted_a =
      curve.discount(0.25) / curve.discount(0.50) - 1.0;
  const double unweighted_b =
      curve.discount(0.50) / curve.discount(0.75) - 1.0;
  o.weight = {2.0, 3.0};
  const double got = px::rate<double>(o, curve);
  const double want = (2.0 * unweighted_a + 3.0 * unweighted_b) / 0.5;
  EXPECT_NEAR(got, want, 1e-15 * std::max(1.0, std::abs(want)));
}

// The k-form is an evaluation ORDER, not a different model: it must agree with the design doc's
// literal `DF(pay)*(rate+spread)*tau_pay` to ~1e-15 when tau_pay != tau_index (k != 1), where the
// two genuinely differ in rounding. (At k == 1 they agree bitwise -- that is the reduction above.)
TEST_F(Generic, KFormMatchesLiteralFormWithSpreadAndMixedDayCount) {
  px::FloatCoupon c;
  c.obs.sub_start = {0.25};
  c.obs.sub_end = {0.50};
  c.obs.tau_index = 0.25;   // ACT/360-ish index accrual
  c.pay = 0.50;
  c.tau_pay = 0.25 * 360.0 / 365.0;  // a DIFFERENT payment day count => k != 1
  c.spread = 25e-4;

  const double got = px::float_coupon_pv<double>(c, curve, curve);
  const double r = px::rate<double>(c.obs, curve);
  const double want = curve.discount(c.pay) * (r + c.spread) * c.tau_pay;
  EXPECT_NE(c.tau_pay, c.obs.tau_index);  // the test is not vacuous
  EXPECT_NEAR(got, want, 1e-15 * std::max(1.0, std::abs(want)));
}

TEST_F(Generic, SpreadIsAdditiveOutsideTheIndex) {
  px::FloatCoupon base;
  base.obs.sub_start = {0.25};
  base.obs.sub_end = {0.50};
  base.obs.tau_index = 0.25;
  base.pay = 0.50;
  base.tau_pay = 0.25;

  px::FloatCoupon spread = base;
  spread.spread = 50e-4;
  const double diff = px::float_coupon_pv<double>(spread, curve, curve) -
                      px::float_coupon_pv<double>(base, curve, curve);
  // A pure spread coupon is worth DF(pay)*spread*tau_pay -- no index dependence.
  EXPECT_NEAR(diff, curve.discount(base.pay) * 50e-4 * base.tau_pay, 1e-15);
}

TEST_F(Generic, FullyFixedCouponIsAConstant) {
  px::FloatCoupon c;
  c.obs.realized = 0.05 * 0.25;  // fixing * tau_index
  c.obs.tau_index = 0.25;
  c.pay = 0.50;
  c.tau_pay = 0.25;
  ASSERT_TRUE(c.obs.sub_start.empty());
  EXPECT_NEAR(px::rate<double>(c.obs, curve), 0.05, 1e-15);
  EXPECT_NEAR(px::float_coupon_pv<double>(c, curve, curve), curve.discount(0.50) * 0.05 * 0.25,
              1e-15);
}

// ---- AAD safety -------------------------------------------------------------------------------

// The trap (CLAUDE.md §7b): `Scalar(constant)` has an EMPTY derivative vector, and `+=`-ing a
// length-M term onto it is silent UB under -DNDEBUG. A fully fixed coupon inside an otherwise live
// leg is exactly that shape. `float_coupon_pv` must dodge it by keeping the constant a raw double
// and letting DF(pay) carry the derivatives -- so the leg sum stays correctly SIZED.
TEST_F(Generic, FullyFixedCouponInsideALiveLegIsAadSafe) {
  using swaps::ad::Dual;
  const int m = static_cast<int>(mk.meeting_times.size() + mk.back_times.size());

  Eigen::VectorXd x(m);
  {
    int i = 0;
    for (double f : rm::reference_front_forwards) x[i++] = f;
    for (double g : rm::reference_back_forwards) x[i++] = g;
  }
  auto dc = swaps::curve::make_calibration_curve<Dual>(mk.meeting_times, mk.back_times);
  dc.set_forwards(swaps::ad::seed(x));

  px::FloatCoupon fixed;  // already fixed: NO sub-periods
  fixed.obs.realized = 0.05 * 0.25;
  fixed.obs.tau_index = 0.25;
  fixed.pay = 0.50;
  fixed.tau_pay = 0.25;

  px::FloatCoupon live;
  live.obs.sub_start = {0.50};
  live.obs.sub_end = {0.75};
  live.obs.tau_index = 0.25;
  live.pay = 0.75;
  live.tau_pay = 0.25;

  // Fixed coupon FIRST: it seeds the leg accumulator, so if it carried an empty derivative vector
  // the `+=` of the live coupon would mismatch sizes.
  const Dual pv = px::float_leg_pv<Dual>({fixed, live}, dc, dc);
  ASSERT_EQ(pv.derivatives().size(), m);
  EXPECT_TRUE(pv.derivatives().allFinite());
  EXPECT_TRUE(std::isfinite(pv.value()));

  // The fixed coupon is not derivative-FREE (it still discounts), but its sensitivity must come
  // only through DF(pay) -- i.e. exactly d/dx of DF(0.50)*realized*k.
  const Dual only_fixed = px::float_leg_pv<Dual>({fixed}, dc, dc);
  ASSERT_EQ(only_fixed.derivatives().size(), m);
  const Dual df = dc.discount(0.50);
  EXPECT_TRUE(only_fixed.derivatives().isApprox(df.derivatives() * (0.05 * 0.25), 1e-14));
}
