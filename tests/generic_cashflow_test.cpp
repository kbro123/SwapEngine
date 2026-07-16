// Design §2 gate: the GENERIC cashflow model reduces to the legacy OIS/futures forms.
//
// This is the acceptance criterion for the generalization. The legacy kernel (`OisSwap`,
// `CompoundedFuture`, `AveragedFuture`) is validated against QuantLib to ~1e-16 by pricing_test;
// here we prove the generic kernel is BIT-IDENTICAL to it on the same data, which transitively
// inherits that oracle without re-running it. Anything weaker than bitwise equality would let a
// silent 1-ulp drift into every existing number (design §2 backward-compat invariant).
//
// We also cover what the legacy forms CANNOT express, so the new degrees of freedom are pinned
// before the compiled engine consumes them: weights, spread, tau_pay != tau_index, and a fully
// fixed coupon summed with live siblings under AAD (the empty-derivative trap).

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

// ---- The legacy -> generic mapping (design §1). This lives in the TEST because it is a statement
// about what the legacy index-flavoured shapes MEAN, not engine logic.

// Compounded OIS coupon: ONE sub-period (daily compounding telescopes to the DF ratio), no
// realized, no spread, tau_pay == tau_index. Note the legacy OisSwap carries no float accrual at
// all -- the DF ratio IS the coupon amount -- which is exactly the k == 1 case: any tau works as
// long as tau_pay and tau_index are the SAME double. We use 1.0 to make that explicit.
std::vector<px::FloatCoupon> as_float_leg(const px::OisSwap& s) {
  std::vector<px::FloatCoupon> leg;
  for (std::size_t i = 0; i < s.float_acc_start.size(); ++i) {
    px::FloatCoupon c;
    c.obs.sub_start = {s.float_acc_start[i]};
    c.obs.sub_end = {s.float_acc_end[i]};
    c.obs.tau_index = 1.0;
    c.pay = s.float_pay[i];
    c.tau_pay = 1.0;
    leg.push_back(std::move(c));
  }
  return leg;
}

std::vector<px::FixedCoupon> as_fixed_leg(const px::OisSwap& s) {
  std::vector<px::FixedCoupon> leg;
  for (std::size_t i = 0; i < s.fixed_pay.size(); ++i)
    leg.push_back({s.fixed_pay[i], s.fixed_accrual[i]});
  return leg;
}

px::RateObservation as_obs(const px::CompoundedFuture& f) {
  px::RateObservation o;
  o.sub_start = {f.start};
  o.sub_end = {f.end};
  o.tau_index = f.accrual;
  return o;
}

px::RateObservation as_obs(const px::AveragedFuture& f) {
  px::RateObservation o;
  o.sub_start = f.sub_start;
  o.sub_end = f.sub_end;
  o.realized = f.realized_sum;
  o.tau_index = f.period_yf;
  return o;
}

struct Generic : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::Curve curve = rb::reference_curve(mk);
};

}  // namespace

// ---- Reduction: bitwise, not approximately ----------------------------------------------------

TEST_F(Generic, OisParRateReducesBitExact) {
  ASSERT_FALSE(mk.swaps.empty());
  for (const auto& swap : mk.swaps) {
    const auto sched = swaps::qlx::extract_ois_swap(*swap, mk.today, mk.dc);
    const double legacy = px::ois_par_rate<double>(sched, curve);
    const double generic =
        px::par_rate<double>(as_float_leg(sched), as_fixed_leg(sched), curve, curve);
    // Bitwise. EXPECT_DOUBLE_EQ would tolerate 4 ULP and hide exactly the drift we are excluding.
    EXPECT_EQ(generic, legacy) << " swap maturity " << swap->maturityDate();
  }
}

TEST_F(Generic, CompoundedFutureReducesBitExact) {
  int n = 0;
  for (const auto& f : mk.futures) {
    if (!f.quarterly) continue;
    const auto sched = swaps::qlx::extract_compounded_future(f.start, f.end, mk.today, mk.dc);
    const double conv = mk.convexity(f);
    const double legacy = px::compounded_future_rate<double>(sched, curve) + conv;
    const double generic = px::future_rate<double>(as_obs(sched), conv, curve);
    EXPECT_EQ(generic, legacy) << " 3M future " << f.start << ".." << f.end;
    ++n;
  }
  ASSERT_EQ(n, 8);
}

TEST_F(Generic, AveragedFutureReducesBitExact) {
  int n = 0, with_realized = 0;
  for (const auto& f : mk.futures) {
    if (f.quarterly) continue;
    const auto sched = swaps::qlx::extract_averaged_future(mk.sofr, f.start, f.end, mk.today, mk.dc);
    const double conv = mk.convexity(f);
    const double legacy = px::averaged_future_rate<double>(sched, curve) + conv;
    const double generic = px::future_rate<double>(as_obs(sched), conv, curve);
    EXPECT_EQ(generic, legacy) << " 1M future " << f.start << ".." << f.end;
    if (sched.realized_sum != 0.0) ++with_realized;
    ++n;
  }
  ASSERT_EQ(n, 12);
  // The current-month contract straddles the evaluation date (CLAUDE.md §2), so the realized-days
  // path is genuinely exercised above and not vacuously zero.
  EXPECT_GT(with_realized, 0);
}

// ---- The new degrees of freedom ----------------------------------------------------------------

TEST_F(Generic, ExplicitUnitWeightsMatchEmptyWeights) {
  const auto sched = swaps::qlx::extract_averaged_future(mk.sofr, mk.futures[0].start,
                                                         mk.futures[0].end, mk.today, mk.dc);
  px::RateObservation implicit = as_obs(sched);
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

// The generic par rate must be bit-identical to the legacy one under AAD too -- value AND every
// derivative -- so migrating the residual to the generic kernel cannot move the Jacobian.
TEST_F(Generic, OisParRateReducesBitExactUnderAad) {
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

  for (const auto& swap : mk.swaps) {
    const auto sched = swaps::qlx::extract_ois_swap(*swap, mk.today, mk.dc);
    const Dual legacy = px::ois_par_rate<Dual>(sched, dc);
    const Dual generic = px::par_rate<Dual>(as_float_leg(sched), as_fixed_leg(sched), dc, dc);
    ASSERT_EQ(generic.derivatives().size(), m);
    EXPECT_EQ(generic.value(), legacy.value()) << " swap maturity " << swap->maturityDate();
    for (int k = 0; k < m; ++k)
      EXPECT_EQ(generic.derivatives()[k], legacy.derivatives()[k])
          << " swap maturity " << swap->maturityDate() << " knot " << k;
  }
}
