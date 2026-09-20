// E5 taxonomy: T4 invariant (an engine-internal property: the stamp is the hash twin of structure_equal)
// E8 prototype: the STRUCTURAL STAMP is the hash twin of structure_equal.
//
// The whole design rests on one claim: stamp equality means the same structure. These tests are that claim --
// every structural edit structure_equal rejects must change the stamp, every non-structural difference it
// accepts (quotes, bands, and crucially FIXINGS RESOLUTION) must leave the stamp alone.
#include <gtest/gtest.h>

#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/fixings.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {

// A small two-curve bundle: an outright SOFR-shaped curve + a spread curve, par and basis instruments, one
// coupon carrying a FIXING SCHEDULE (the resolution-sensitive case) and one plain averaged coupon.
cal::BundleProblem make_bundle() {
  cal::BundleProblem p;
  p.curves.resize(2);
  p.curves[0] = px::CurveStructure{.base = -1, .regions = cv::flat_hermite({0.25, 0.5}, {1.0, 2.0, 5.0})};
  p.curves[1] = px::CurveStructure{.base = 0, .regions = cv::flat_hermite({0.25, 0.5}, {1.0, 2.0, 5.0})};
  for (int c = 0; c < 2; ++c) {
    for (double T : {1.0, 2.0, 5.0}) {
      cal::Instrument in;
      in.quote = c == 0 ? cal::QuoteKind::ParRate : cal::QuoteKind::ParSpread;
      std::vector<px::FloatCoupon> flt;
      std::vector<px::FixedCoupon> fix;
      for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
        px::FloatCoupon fc;
        fc.obs.sub_start = {u - 1.0};
        fc.obs.sub_end = {u};
        fc.obs.tau_index = 1.0;
        fc.pay = u;
        fc.tau_pay = 1.0;
        flt.push_back(fc);
        fix.push_back({u, 1.0});
      }
      in.fwd = {flt, c, 0};
      if (c == 1) in.bench = {flt, 0, 0};
      in.fixed = {fix, 0};
      in.market = 0.03 + 0.001 * T;
      p.instruments.push_back(std::move(in));
    }
  }
  return p;
}

std::uint64_t stamp_of(const cal::BundleProblem& p) { return cal::structural_stamp(p); }

// Every mutation below is one structure_equal REJECTS; the stamp must reject it too.
struct Edit {
  const char* what;
  void (*apply)(cal::BundleProblem&);
};
const Edit kEdits[] = {
    {"a knot moves", [](cal::BundleProblem& p) { p.curves[0].regions[1].knots[0] = 1.5; }},
    {"a knot is added", [](cal::BundleProblem& p) { p.curves[0].regions[1].knots.push_back(7.0); }},
    {"an interpolation scheme changes", [](cal::BundleProblem& p) { p.curves[0].regions[1].scheme = cv::Scheme::MonotoneCubic; }},
    {"a curve's base changes", [](cal::BundleProblem& p) { p.curves[1].base = -1; }},
    {"a currency tag changes", [](cal::BundleProblem& p) { p.curves[1].currency = 2; }},
    {"a turn is added", [](cal::BundleProblem& p) { p.curves[0].turns.push_back({0.9, 0.92}); }},
    {"a region's regularisation changes", [](cal::BundleProblem& p) { p.curves[0].regions[1].reg_lambda = 0.5; }},
    {"a quote kind changes", [](cal::BundleProblem& p) { p.instruments[0].quote = cal::QuoteKind::Rate; }},
    {"a forecast role changes", [](cal::BundleProblem& p) { p.instruments[3].fwd.forecast = 0; }},
    {"a discount role changes", [](cal::BundleProblem& p) { p.instruments[0].fixed.discount = 1; }},
    {"a pay date moves", [](cal::BundleProblem& p) { p.instruments[2].fwd.coupons[1].pay += 1.0 / 365.0; }},
    {"an accrual factor changes", [](cal::BundleProblem& p) { p.instruments[2].fixed.coupons[0].tau = 0.99; }},
    {"an observation window shifts", [](cal::BundleProblem& p) { p.instruments[2].fwd.coupons[0].obs.sub_end[0] += 0.01; }},
    {"a coupon is dropped", [](cal::BundleProblem& p) { p.instruments[2].fwd.coupons.pop_back(); }},
    {"an instrument is dropped", [](cal::BundleProblem& p) { p.instruments.pop_back(); }},
    {"a spread is added to a coupon", [](cal::BundleProblem& p) { p.instruments[1].fwd.coupons[0].spread = 1e-4; }},
    {"convexity changes", [](cal::BundleProblem& p) { p.instruments[1].convexity = 1e-5; }},
    {"an mtm leg appears", [](cal::BundleProblem& p) { p.instruments[1].mtm = p.instruments[1].fwd; }},
};

}  // namespace

TEST(StructuralStamp, EveryStructuralEditChangesTheStamp) {
  const cal::BundleProblem base = make_bundle();
  const std::uint64_t s0 = stamp_of(base);
  for (const Edit& e : kEdits) {
    cal::BundleProblem p = base;
    e.apply(p);
    EXPECT_FALSE(cal::structure_equal(base, p)) << e.what << ": structure_equal must reject it (test premise)";
    EXPECT_NE(s0, stamp_of(p)) << e.what << ": the stamp must change with it";
  }
}

TEST(StructuralStamp, QuotesAndBandsDoNotChangeIt) {
  const cal::BundleProblem base = make_bundle();
  cal::BundleProblem p = base;
  for (auto& in : p.instruments) {
    in.market += 25e-4;
    in.band_lower = in.market - 1e-4;
    in.band_upper = in.market + 1e-4;
    in.band_decay = 0.5;
  }
  EXPECT_TRUE(cal::structure_equal(base, p));
  EXPECT_EQ(stamp_of(base), stamp_of(p)) << "a requote is not a structural change";
}

// THE case the design depends on: a session resolves its fixing schedules in place (set_fixings /
// set_evaluation_date rewrite realized, realized_factor and the forecast sub-periods, but NEVER the carried
// accrual period). structure_equal is blind to that, and the stamp must be too -- otherwise a client's
// unresolved document could never rebind onto a session carrying resolved schedules.
TEST(StructuralStamp, FixingResolutionDoesNotChangeIt) {
  cal::BundleProblem unresolved = make_bundle();
  auto& c = unresolved.instruments[1].fwd.coupons[0];
  // As build::instruments stamps them: the accrual period is carried, independent of any observation shift.
  c.accrual_set = true;
  c.accrual_start = 0.0;
  c.accrual_end = 1.0;
  for (int k = 0; k < 4; ++k)
    c.obs.fixing_schedule.push_back(px::FixingDay{.fixing_date = 20000 + k, .accrual = 0.25, .t_start = 0.25 * k, .t_end = 0.25 * (k + 1)});
  c.obs.fixing_index = "USD-SOFR";

  cal::BundleProblem resolved = unresolved;  // what the session looks like after resolve_fixings()
  auto& o = resolved.instruments[1].fwd.coupons[0].obs;
  o.realized = 0.0125;   // the past fixings, baked
  o.realized_factor = 1.0125;
  o.sub_start = {0.5};   // the window shrank to the first FUTURE fixing
  o.sub_end = {1.0};

  ASSERT_TRUE(cal::structure_equal(unresolved, resolved)) << "test premise: equality is resolution-insensitive";
  EXPECT_EQ(stamp_of(unresolved), stamp_of(resolved)) << "the stamp must be resolution-insensitive too";
}

// The converse, and the honest limit of that blindness: WITHOUT a carried accrual period the effective
// exchange dates are the observation window, which resolution moves -- so structure_equal rejects the pair.
// The stamp is the twin either way: it must reject it too (never accept what the equality check refuses).
TEST(StructuralStamp, WithoutACarriedAccrualPeriodResolutionIsAStructuralChangeForBoth) {
  cal::BundleProblem unresolved = make_bundle();
  auto& c = unresolved.instruments[1].fwd.coupons[0];
  ASSERT_FALSE(c.accrual_set) << "test premise: this fixture's coupons carry no accrual period";
  for (int k = 0; k < 4; ++k)
    c.obs.fixing_schedule.push_back(px::FixingDay{.fixing_date = 20000 + k, .accrual = 0.25, .t_start = 0.25 * k, .t_end = 0.25 * (k + 1)});

  cal::BundleProblem resolved = unresolved;
  auto& o = resolved.instruments[1].fwd.coupons[0].obs;
  o.sub_start = {0.5};
  o.sub_end = {1.0};

  EXPECT_FALSE(cal::structure_equal(unresolved, resolved));
  EXPECT_NE(stamp_of(unresolved), stamp_of(resolved)) << "the stamp tracks the equality check, both ways";
}

TEST(StructuralStamp, MakeStampedCarriesTheStampOfItsOwnProblem) {
  const cal::StampedBundle b = cal::make_stamped(make_bundle());
  EXPECT_EQ(b.stamp, cal::structural_stamp(b.problem));
  EXPECT_NE(b.stamp, 0u);
}
