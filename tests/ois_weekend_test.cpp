// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// INDEPENDENT verification that overnight (OIS) compounding accrues the right fixing over weekends and
// holidays -- checking QuantLib AND our engine against a from-first-principles calendar walk, not just
// against each other. The failure mode this targets: a Friday fixing must compound over the 3-day weekend
// (dt = 3/360), and the Friday before a Monday holiday over 4 days (dt = 4/360); a naive engine that
// counts only business days (dt = 1/360 each, dropping weekend/holiday days) gets a materially wrong rate.
//
// The reference is a hand-rolled walk of the index's own fixing calendar: step business day to business
// day, accrue each fixing over the ACTUAL calendar span to the next business day. The decisive invariant
// is that these spans TILE every calendar day exactly once (Sum dt * 360 == total calendar days).
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"

using namespace QuantLib;
namespace qlx = swaps::qlx;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
// A deterministic, date-varying overnight fixing so the compounding is a genuine product, not f * Sum dt.
double fix(const Date& d) { return 0.0400 + 0.00004 * (d.serialNumber() % 13); }
}  // namespace

TEST(OisWeekend, RealizedCompoundingTilesEveryCalendarDay) {
  const Date eval(1, July, 2026);
  Settings::instance().evaluationDate() = eval;
  RelinkableHandle<YieldTermStructure> h;
  h.linkTo(ext::make_shared<FlatForward>(eval, 0.04, Actual365Fixed(), Continuous));  // unused: all realized
  const auto sofr = ext::make_shared<Sofr>(h);
  const Calendar cal = sofr->fixingCalendar();
  const DayCounter idc = sofr->dayCounter();  // Actual/360

  // Hermetic: SOFR fixing history is GLOBAL (IndexManager), so clear any fixings a prior test seeded before
  // we lay down ours -- otherwise a duplicate-with-different-value in this span makes addFixing throw and
  // this test fails only in a full-suite run (it passes in isolation).
  IndexManager::instance().clearHistory(sofr->name());

  // A coupon fully in the PAST spanning many weekends AND Memorial Day (Mon 25 May 2026) + Juneteenth
  // (Fri 19 Jun 2026): both create long accrual spans a naive engine would mishandle.
  const Date start = cal.adjust(Date(15, May, 2026)), end = cal.adjust(Date(26, June, 2026));
  for (Date d = start - 8; d <= end + 3; ++d)
    if (sofr->isValidFixingDate(d)) sofr->addFixing(d, fix(d));
  const Date pay = cal.advance(end, 2, Days);
  const auto cpn = ext::make_shared<OvernightIndexedCoupon>(pay, 1.0, start, end, sofr);  // Compound default

  // ---- FIRST-PRINCIPLES reference: walk the fixing calendar, accrue each fixing over its real span ----
  std::vector<Date> walk_vd{start};
  double hand = 1.0;
  long walk_days = 0;
  int weekend3 = 0, long4plus = 0;
  for (Date d = start; d < end;) {
    const Date nxt = cal.advance(d, 1, Days);
    const double dt = idc.yearFraction(d, nxt);
    hand *= (1.0 + fix(d) * dt);
    const int span = nxt - d;
    walk_days += span;
    if (span == 3) ++weekend3;
    if (span >= 4) ++long4plus;
    walk_vd.push_back(nxt);
    d = nxt;
  }
  const double tau = idc.yearFraction(start, end);
  const double hand_rate = (hand - 1.0) / tau;

  // The decisive invariant: the business-day spans partition [start,end] with NO calendar day lost or
  // double-counted. A "business-days-only" bug (dt=1/360 each) would make Sum(dt)*360 < total days.
  EXPECT_EQ(walk_days, end - start) << "accrual spans must tile every calendar day exactly once";
  EXPECT_GT(weekend3, 0) << "the period must actually contain weekends (3-day Fri->Mon spans)";
  EXPECT_GT(long4plus, 0) << "the period must actually contain a holiday (>=4-day span)";
  std::cout << "  [ois-weekend] calendar days=" << walk_days << " weekends(3d)=" << weekend3
            << " holidays(>=4d)=" << long4plus << " hand_rate=" << hand_rate << "\n";

  // QuantLib's own value-date construction must equal the business-day walk (this is the thing that could
  // be wrong): same value dates, and Sum(dt)*360 == calendar days.
  EXPECT_EQ(cpn->valueDates(), walk_vd) << "QuantLib value dates must equal an independent business-day walk";
  double ql_sum = 0;
  for (double d : cpn->dt()) ql_sum += d;
  EXPECT_NEAR(ql_sum * 360.0, static_cast<double>(walk_days), 1e-9) << "QuantLib dt() must tile all calendar days";

  // Both QuantLib and our engine must reproduce the first-principles compounded rate.
  const double ql_rate = cpn->rate();
  const px::RateObservation o = qlx::extract_overnight_obs(*cpn, eval, Actual365Fixed());
  ASSERT_TRUE(o.sub_start.empty()) << "a fully-realized coupon carries no forecast sub-periods";
  auto dummy = cv::make_modular_curve<double>(cv::flat_hermite({0.5}, {1, 2, 5, 10}));  // unused (subs empty)
  Eigen::VectorXd xd(5); xd << 0.04, 0.04, 0.04, 0.04, 0.04; dummy.set_forwards(xd);
  const double our_rate = px::rate<double>(o, dummy);

  std::cout << "  [ois-weekend] hand=" << hand_rate << " quantlib=" << ql_rate << " ours=" << our_rate << "\n";
  EXPECT_NEAR(ql_rate, hand_rate, 1e-12) << "QuantLib compounding must match first principles over weekends/holidays";
  EXPECT_NEAR(our_rate, hand_rate, 1e-12) << "our compounding must match first principles over weekends/holidays";

  // Sanity that the test bites: a naive business-days-only compound (dt=1/360 each) is materially different.
  double naive = 1.0;
  for (Date d = start; d < end; d = cal.advance(d, 1, Days)) naive *= (1.0 + fix(d) * (1.0 / 360.0));
  const double naive_rate = (naive - 1.0) / tau;
  EXPECT_GT(std::abs(naive_rate - hand_rate), 1e-4)
      << "dropping weekend/holiday days would move the rate by >1bp -- the test genuinely exercises it";
}

TEST(OisWeekend, ForecastTelescopingSpansWeekendsAndHolidays) {
  // The FORWARD path: a future OIS coupon over a holiday weekend. We telescope DF(v0)/DF(vn); QuantLib does
  // the same. Wire OUR curve in as the term structure so both price off identical discount factors, and
  // confirm (a) the value dates tile every calendar day, (b) our rate == QuantLib to machine precision.
  const Date eval(1, July, 2026);
  Settings::instance().evaluationDate() = eval;
  RelinkableHandle<YieldTermStructure> h;
  auto curve = cv::make_modular_curve<double>(cv::flat_hermite({0.25, 0.5}, {1, 2, 3, 5, 10}));
  Eigen::VectorXd x(7); x << 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.05; curve.set_forwards(x);
  auto ts = ext::make_shared<qlx::CurveTermStructure<cv::ModularCurve<double>>>(eval, Actual365Fixed(), &curve);
  ts->enableExtrapolation();
  h.linkTo(ts);
  const auto sofr = ext::make_shared<Sofr>(h);
  const Calendar cal = sofr->fixingCalendar();
  const DayCounter idc = sofr->dayCounter();

  const Date start = cal.adjust(Date(1, September, 2026)), end = cal.adjust(Date(1, October, 2026));  // Labor Day 7 Sep
  const Date pay = cal.advance(end, 2, Days);
  const auto cpn = ext::make_shared<OvernightIndexedCoupon>(pay, 1.0, start, end, sofr);

  long days = 0; int long4 = 0;
  for (Date d = start; d < end;) { const Date n = cal.advance(d, 1, Days); days += (n - d); if (n - d >= 4) ++long4; d = n; }
  double qlsum = 0; for (double d : cpn->dt()) qlsum += d;
  EXPECT_NEAR(qlsum * 360.0, static_cast<double>(days), 1e-9) << "forward coupon dt() must tile every calendar day";
  EXPECT_GT(long4, 0) << "the period must contain the Labor Day long weekend";

  const double ql_rate = cpn->rate();
  const px::RateObservation o = qlx::extract_overnight_obs(*cpn, eval, Actual365Fixed());
  const double our_rate = px::rate<double>(o, curve);
  std::cout << "  [ois-weekend-fwd] days=" << days << " quantlib=" << ql_rate << " ours=" << our_rate
            << " rel=" << std::abs(our_rate - ql_rate) / std::max(1.0, std::abs(ql_rate)) << "\n";
  EXPECT_NEAR(our_rate, ql_rate, 1e-11) << "our telescoped forward must match QuantLib over the holiday weekend";
}
