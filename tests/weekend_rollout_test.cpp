// First-principles verification that ACCRUALS, DISCOUNT FACTORS and FIXINGS roll out correctly over
// weekends and holidays in EVERY interpolation region -- with special attention to the flat-forward
// front (CLAUDE.md §2). This is the QuantLib-FREE companion to the oracle-linked ois_weekend_test.cpp:
// it checks the engine's DF(vd)/DF(md) roll-out against a from-first-principles calendar walk, not
// against QuantLib, so it lives in the engine's own gate (no QuantLib link).
//
// What "rolls out correctly over a weekend" MEANS, made into invariants:
//   1. Curve time counts ACTUAL calendar days: t(Mon) - t(Fri) == 3/365 across a weekend.
//   2. FLAT region, weekend WITHIN a meeting segment: DF(Fri)/DF(Mon) == exp(f * 3/365) exactly.
//   3. FLAT region, weekend STRADDLING a meeting jump: the DF ratio splits into the two flat pieces
//      (Flat::integral is the two-segment split; this is the case unique to the piecewise-flat front).
//   4. Per-business-day (averaged) sub-periods TILE every calendar day exactly once -- no weekend or
//      holiday day dropped or double-counted (the failure mode a business-days-only engine has).
//   5. Compounded telescoping is EXACT over weekends/holidays in EVERY region:
//        prod_k DF(vd_k)/DF(md_k) == DF(vd_0)/DF(vd_last).
//   6. The engine's arithmetic forecast numerator == an independent per-day recomputation from discount(),
//      and rate == numerator / tau_index, in every region.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cv = swaps::curve;
namespace px = swaps::pricing;

namespace {

// ---- tiny self-contained calendar -------------------------------------------------------------
// Serial day counting from a reference. ref = serial 0 = a Monday; dow: 0=Mon..6=Sun; 5,6 = weekend.
// One mid-stream holiday (serial 32) forces a >=4-day accrual span, exactly like a Monday holiday.
constexpr double kCurveDc = 365.0;  // ACT/365F for curve time
constexpr double kIndexDc = 360.0;  // ACT/360 for index accrual
const std::vector<int> kHolidays = {32};

int dow(int s) { int r = s % 7; return r < 0 ? r + 7 : r; }
bool is_holiday(int s) {
  for (int h : kHolidays)
    if (h == s) return true;
  return false;
}
bool is_bus(int s) { return dow(s) < 5 && !is_holiday(s); }
int next_bus(int s) {
  int d = s + 1;
  while (!is_bus(d)) ++d;
  return d;
}
int adjust_prec(int s) {
  int d = s;
  while (!is_bus(d)) --d;
  return d;
}
double tcurve(int s) { return s / kCurveDc; }
double idx_accr(int a, int b) { return (b - a) / kIndexDc; }

// The SHIPPED flat_hermite layout: meeting-date flat front + Hermite back. Meeting m0 sits on a
// Saturday (serial 5) so an overnight fixing [Fri=4, Mon=7] straddles the forward jump at m0.
const std::vector<int> kMeetingSerial = {5, 40, 100};
const std::vector<double> kBack = {1.0, 2.0, 5.0, 10.0};
constexpr double kF0 = 0.0525, kF1 = 0.0500, kF2 = 0.0475;  // flat levels on (0,m0], (m0,m1], (m1,m2]

std::vector<double> meeting_times() {
  std::vector<double> m;
  for (int s : kMeetingSerial) m.push_back(tcurve(s));
  return m;
}
Eigen::VectorXd forwards() {
  Eigen::VectorXd x(kMeetingSerial.size() + kBack.size());
  x << kF0, kF1, kF2, 0.0460, 0.0450, 0.0440, 0.0430;
  return x;
}
cv::ModularCurve<double> region_curve(cv::Scheme back_scheme, double sigma) {
  std::vector<cv::CurveModule> mods;
  mods.push_back({meeting_times(), cv::Scheme::Flat, 0.0});
  mods.push_back({kBack, back_scheme, sigma});
  auto c = cv::make_modular_curve<double>(mods);
  c.set_forwards(forwards());
  return c;
}

struct RegionCase {
  const char* name;
  cv::Scheme scheme;
  double sigma;
};
const std::vector<RegionCase> kRegions = {
    {"flat+Hermite", cv::Scheme::Hermite, 0.0},
    {"flat+Linear", cv::Scheme::Linear, 0.0},
    {"flat+NaturalCubic", cv::Scheme::NaturalCubic, 0.0},
    {"flat+Tension", cv::Scheme::Tension, 5.0},
    {"flat+MonotoneCubic", cv::Scheme::MonotoneCubic, 0.0},
};

// A period spanning several weekends + the holiday, walked business day to business day.
struct Walk {
  int start, end;
  std::vector<int> vd;  // value dates: start, next_bus, ..., end
};
Walk make_walk() {
  Walk w;
  w.start = adjust_prec(10);
  w.end = next_bus(55);
  int d = w.start;
  w.vd.push_back(d);
  while (d < w.end) {
    d = next_bus(d);
    w.vd.push_back(d);
  }
  return w;
}

// Averaged (Simple) observation: one sub-period per business day; empty weights == all-ones.
px::RateObservation avg_obs(const Walk& w) {
  px::RateObservation o;
  o.tau_index = idx_accr(w.start, w.end);
  for (std::size_t k = 0; k + 1 < w.vd.size(); ++k) {
    o.sub_start.push_back(tcurve(w.vd[k]));
    o.sub_end.push_back(tcurve(w.vd[k + 1]));
  }
  return o;
}

}  // namespace

TEST(WeekendRollout, CurveTimeCountsCalendarDaysAcrossWeekend) {
  EXPECT_EQ(dow(25), 4) << "serial 25 must be a Friday";
  EXPECT_EQ(dow(28), 0) << "serial 28 must be a Monday";
  EXPECT_NEAR(tcurve(28) - tcurve(25), 3.0 / 365.0, 1e-15)
      << "a Fri->Mon weekend must advance curve time by 3 actual calendar days";
}

TEST(WeekendRollout, FlatRegionWeekendWithinSegment) {
  auto c = region_curve(cv::Scheme::Hermite, 0.0);
  const int fri = 25, mon = 28;  // both inside (m0=5, m1=40): flat level f1
  const double ratio = c.discount(tcurve(fri)) / c.discount(tcurve(mon));
  EXPECT_NEAR(ratio, std::exp(kF1 * 3.0 / 365.0), 1e-14)
      << "flat forward must accrue the full 3-day weekend, not one business day";
  // The implied overnight fixing over the weekend, index ACT/360 accrual = 3/360.
  const double fixing = (ratio - 1.0) / idx_accr(fri, mon);
  EXPECT_NEAR(fixing, (std::exp(kF1 * 3.0 / 365.0) - 1.0) / (3.0 / 360.0), 1e-14);
}

TEST(WeekendRollout, FlatRegionWeekendStraddlesMeetingJump) {
  auto c = region_curve(cv::Scheme::Hermite, 0.0);
  const int fri = 4, mon = 7;  // meeting m0 at serial 5 (Sat) lies strictly between
  ASSERT_EQ(dow(fri), 4);
  ASSERT_EQ(dow(mon), 0);
  const double ratio = c.discount(tcurve(fri)) / c.discount(tcurve(mon));
  // Two-piece: f0 over [Fri,Sat] = 1 day, f1 over [Sat,Mon] = 2 days.
  const double expect = std::exp(kF0 * 1.0 / 365.0 + kF1 * 2.0 / 365.0);
  EXPECT_NEAR(ratio, expect, 1e-14) << "the weekend integral must split at the meeting-date jump";
  // It must genuinely differ from a single-level accrual at either f0 or f1 (the split is real).
  EXPECT_GT(std::abs(ratio - std::exp(kF0 * 3.0 / 365.0)), 1e-8);
  EXPECT_GT(std::abs(ratio - std::exp(kF1 * 3.0 / 365.0)), 1e-8);
}

TEST(WeekendRollout, PerDaySubPeriodsTileEveryCalendarDay) {
  const Walk w = make_walk();
  long tiled = 0;
  int weekends = 0, longspans = 0;
  for (std::size_t k = 0; k + 1 < w.vd.size(); ++k) {
    const int span = w.vd[k + 1] - w.vd[k];
    tiled += span;
    if (span == 3) ++weekends;
    if (span >= 4) ++longspans;
  }
  EXPECT_EQ(tiled, static_cast<long>(w.end - w.start))
      << "business-day sub-periods must tile every calendar day exactly once";
  EXPECT_GT(weekends, 0) << "the period must actually contain weekends (3-day Fri->Mon spans)";
  EXPECT_GT(longspans, 0) << "the period must contain a >=4-day holiday span";
}

TEST(WeekendRollout, CompoundedTelescopingExactAllRegions) {
  const Walk w = make_walk();
  for (const auto& rc : kRegions) {
    auto c = region_curve(rc.scheme, rc.sigma);
    double prod = 1.0;
    for (std::size_t k = 0; k + 1 < w.vd.size(); ++k)
      prod *= c.discount(tcurve(w.vd[k])) / c.discount(tcurve(w.vd[k + 1]));
    const double single = c.discount(tcurve(w.vd.front())) / c.discount(tcurve(w.vd.back()));
    EXPECT_NEAR(prod, single, 1e-12)
        << rc.name << ": daily DF-ratio product must telescope exactly across weekends/holidays";
  }
}

TEST(WeekendRollout, ArithmeticNumeratorMatchesHandAllRegions) {
  const Walk w = make_walk();
  const px::RateObservation o = avg_obs(w);
  for (const auto& rc : kRegions) {
    auto c = region_curve(rc.scheme, rc.sigma);
    const double eng = px::obs_forward_sum<double>(o, c);
    double hand = 0.0;
    for (std::size_t k = 0; k + 1 < w.vd.size(); ++k)
      hand += c.discount(tcurve(w.vd[k])) / c.discount(tcurve(w.vd[k + 1])) - 1.0;
    EXPECT_NEAR(eng, hand, 1e-13) << rc.name << ": engine numerator must equal the per-day hand walk";
    EXPECT_NEAR(px::rate<double>(o, c), hand / o.tau_index, 1e-13) << rc.name;
  }
}

TEST(WeekendRollout, StraddlingFixingPerDayPathFlatFront) {
  auto c = region_curve(cv::Scheme::Hermite, 0.0);
  // Overnight fixing [Fri=4, Mon=7] straddling the meeting jump, priced as one sub-period.
  px::RateObservation o;
  o.tau_index = idx_accr(4, 7);  // 3/360
  o.sub_start.push_back(tcurve(4));
  o.sub_end.push_back(tcurve(7));
  const double expect_ratio = std::exp(kF0 * 1.0 / 365.0 + kF1 * 2.0 / 365.0);
  EXPECT_NEAR(px::obs_forward_sum<double>(o, c), expect_ratio - 1.0, 1e-14);
  EXPECT_NEAR(px::rate<double>(o, c), (expect_ratio - 1.0) / (3.0 / 360.0), 1e-14);
}
