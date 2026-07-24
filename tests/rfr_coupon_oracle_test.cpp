// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// QuantLib 1.35 oracle for RFR conventions on compounded overnight coupons: lookback days, lockout
// days, and observation shift (added to OvernightIndexedCoupon in 1.35). These break the telescoping
// that collapses a plain compounded coupon to a single DF ratio, so the engine prices them in its new
// COMPOUNDED (product) mode (RateObservation.compounded). Composite oracle: our curve is wired in as the
// QuantLib term structure, so QuantLib's own OvernightIndexedCouponPricer and our extract+price run off
// IDENTICAL discount factors -- any disagreement is a pricing bug, not a curve mismatch (CLAUDE.md §3).
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace px = swaps::pricing;
namespace qlx = swaps::qlx;

namespace {

struct RfrCoupon : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::Curve curve = rb::reference_curve(mk);
  ext::shared_ptr<qlx::CurveTermStructure<rb::Curve>> ts;

  void SetUp() override {
    ts = ext::make_shared<qlx::CurveTermStructure<rb::Curve>>(mk.today, mk.dc, &curve);
    ts->enableExtrapolation();
    h.linkTo(ts);
  }

  ext::shared_ptr<OvernightIndexedCoupon> make(const Date& start, const Date& end, Natural lookback,
                                               Natural lockout, bool shift) {
    const Date pay = mk.sofr->fixingCalendar().advance(end, 2, Days);
    return ext::make_shared<OvernightIndexedCoupon>(
        pay, 1.0, start, end, mk.sofr, 1.0, 0.0, Date(), Date(), mk.dc,
        /*telescopicValueDates=*/false, RateAveraging::Compound, lookback, lockout, shift);
  }

  // The engine's rate for the coupon, via the RFR-aware extractor + the compounded-product kernel.
  double ours(const ext::shared_ptr<OvernightIndexedCoupon>& c) {
    const px::RateObservation o = qlx::extract_overnight_obs(*c, mk.today, mk.dc);
    return px::rate<double>(o, curve);
  }
};

}  // namespace

TEST_F(RfrCoupon, PlainCompoundedStillTelescopesNotCompoundedMode) {
  // Guard: a coupon with NONE of the RFR features must keep the existing telescoped single-sub-period
  // path (compounded == false) -- i.e. calibration coupons are completely unaffected by this feature.
  const Calendar cal = mk.sofr->fixingCalendar();
  const Date start = cal.advance(mk.today, 3, Months), end = cal.advance(start, 1, Months);
  const px::RateObservation o = qlx::extract_overnight_obs(*make(start, end, 0, 0, false), mk.today, mk.dc);
  EXPECT_FALSE(o.compounded) << "no lookback/lockout/shift => telescoped arithmetic path, not product mode";
  EXPECT_EQ(o.sub_start.size(), 1u) << "plain compounded OIS is ONE telescoped sub-period";
}

TEST_F(RfrCoupon, MatchesQuantLibAcrossRfrVariants) {
  const Calendar cal = mk.sofr->fixingCalendar();
  const Date start = cal.advance(mk.today, 3, Months), end = cal.advance(start, 1, Months);
  struct Case { const char* name; Natural lookback, lockout; bool shift; };
  const std::vector<Case> cases{
      {"lookback-no-shift", 5, 0, false},
      {"lookback-obs-shift", 5, 0, true},
      {"lockout-only", 0, 3, false},
      {"lookback+shift+lockout", 2, 2, true},
  };
  double worst = 0.0;
  for (const auto& c : cases) {
    auto cpn = make(start, end, c.lookback, c.lockout, c.shift);
    const px::RateObservation o = qlx::extract_overnight_obs(*cpn, mk.today, mk.dc);
    ASSERT_TRUE(o.compounded) << c.name << ": must route to the compounded-product mode";
    const double ql = cpn->rate(), our = px::rate<double>(o, curve);
    const double rel = std::abs(our - ql) / std::max(1.0, std::abs(ql));
    std::cout << "  [rfr:" << c.name << "] ql=" << ql << " ours=" << our << " rel=" << rel << "\n";
    EXPECT_LT(rel, 1e-12) << c.name << ": engine compounded-product rate must match QuantLib 1.35";
    worst = std::max(worst, rel);
  }
  std::cout << "  [rfr] worst rel = " << worst << "\n";
}

TEST_F(RfrCoupon, PartiallyRealizedLookbackLockoutMatchesQuantLib) {
  // A coupon straddling today: past days fold into realized_factor (the multiplicative Π_past(1+f·dt)),
  // future days stay in the product. Exercises the realized-prefix path with lookback + lockout. Past
  // SOFR fixings are seeded by build_market (today-20..today @ 0.0430).
  const Calendar cal = mk.sofr->fixingCalendar();
  const Date start = cal.advance(mk.today, -8, Days), end = cal.advance(mk.today, 20, Days);
  auto cpn = make(start, end, 2, 2, /*shift=*/true);
  const px::RateObservation o = qlx::extract_overnight_obs(*cpn, mk.today, mk.dc);
  ASSERT_TRUE(o.compounded);
  EXPECT_NE(o.realized_factor, 1.0) << "a partially-past coupon must carry a non-trivial realized product";
  const double ql = cpn->rate(), our = px::rate<double>(o, curve);
  const double rel = std::abs(our - ql) / std::max(1.0, std::abs(ql));
  std::cout << "  [rfr-partial] ql=" << ql << " ours=" << our << " realized_factor=" << o.realized_factor
            << " rel=" << rel << "\n";
  EXPECT_LT(rel, 1e-12) << "partially-realized compounded RFR coupon must match QuantLib 1.35";
}
