// Priority-1 schedule-roller gate: the OPTIONAL ISDA richness (EOM snap, stub location, roll-day anchor)
// added to build::swap_periods_to. The load-bearing guarantee is (a) — a DEFAULT ScheduleRule reproduces
// the legacy short-back-stub forward roll BYTE-FOR-BYTE — cross-checked here against an inline transcription
// of the legacy algorithm (the documented oracle) over several tenors/frequencies, plus hand-computed dates
// for the EOM / front-stub / roll-day-anchor opt-in paths. QuantLib-free (swaps_tests).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "swaps/build/schedule.hpp"

namespace b = swaps::build;

namespace {
// The legacy roller, transcribed verbatim from the pre-extension swap_periods_to body — the independent
// oracle for the byte-identical regression (assertion (a)).
std::vector<b::Period> legacy_periods(const b::Date& vd, const std::string& cal, const b::Date& mat,
                                      const std::string& freq, const std::string& bdc, int spot_lag) {
  const b::Date spot = b::spot_date(vd, cal, spot_lag);
  if (mat <= spot) return {{spot, mat}};
  const int step_m = b::tok_months(freq);
  std::vector<b::Date> bounds{spot};
  for (int m = step_m;; m += step_m) {
    const b::Date d = b::adjust(cal, b::add_period(spot, std::to_string(m) + "M"), bdc);
    if (d >= mat) break;
    bounds.push_back(d);
  }
  bounds.push_back(mat);
  std::vector<b::Period> out;
  for (std::size_t i = 0; i + 1 < bounds.size(); ++i) out.emplace_back(bounds[i], bounds[i + 1]);
  return out;
}

void expect_same(const std::vector<b::Period>& a, const std::vector<b::Period>& c) {
  ASSERT_EQ(a.size(), c.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(b::iso(a[i].first), b::iso(c[i].first)) << "period " << i << " start";
    EXPECT_EQ(b::iso(a[i].second), b::iso(c[i].second)) << "period " << i << " end";
  }
}
}  // namespace

// (a) Default == current output, byte-for-byte, across tenors and frequencies. The no-arg call AND an
// explicitly default-constructed ScheduleRule both route to the legacy fast path.
TEST(ScheduleStubEom, DefaultIsByteIdenticalToLegacy) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const std::string cal = "USD-SOFR", bdc = "ModifiedFollowing";
  const int spot_lag = 2;
  for (const char* tenor : {"1y", "2y", "5y", "7y", "10y", "18m", "30y"}) {
    for (const char* freq : {"3M", "6M", "1Y"}) {
      const b::Date mat = b::resolve(tenor, vd, "NONE", "Following", 0);
      const auto oracle = legacy_periods(vd, cal, mat, freq, bdc, spot_lag);
      const auto no_arg = b::swap_periods_to(vd, cal, mat, freq, bdc, spot_lag);
      const auto def_rule = b::swap_periods_to(vd, cal, mat, freq, bdc, spot_lag, b::ScheduleRule{});
      SCOPED_TRACE(std::string(tenor) + "/" + freq);
      expect_same(oracle, no_arg);
      expect_same(oracle, def_rule);
    }
  }
}

// (b) EOM: a month-end anchor (spot) snaps every rolled boundary to month-end. vd 2026-09-28 -> spot
// 2026-09-30 (Wed, a month-end) on TARGET; semiannual to 2027-12-31. Chosen so every month-end boundary
// is a business day (adjust is a no-op) and the assertion pins the raw month-end dates.
TEST(ScheduleStubEom, EomSnapsMonthEndStart) {
  const b::Date vd = b::Date::from_iso("2026-09-28");
  const b::Date spot = b::spot_date(vd, "EUR", 2);
  ASSERT_EQ(b::iso(spot), "2026-09-30");
  ASSERT_TRUE(b::is_month_end(spot));
  const b::Date mat = b::Date::from_iso("2027-12-31");

  b::ScheduleRule rule;
  rule.eom_auto = true;  // infer EOM because the anchor is a month-end
  const auto per = b::swap_periods_to(vd, "EUR", mat, "6M", "ModifiedFollowing", 2, rule);

  ASSERT_EQ(per.size(), 3u);
  EXPECT_EQ(b::iso(per[0].first), "2026-09-30");
  EXPECT_EQ(b::iso(per[0].second), "2027-03-31");  // Mar-end, not the 30th (would be plain-DOM roll)
  EXPECT_EQ(b::iso(per[1].second), "2027-09-30");
  EXPECT_EQ(b::iso(per[2].second), "2027-12-31");  // final stub ends exactly at maturity
  EXPECT_TRUE(b::is_month_end(per[0].second));
  EXPECT_TRUE(b::is_month_end(per[1].second));

  // Contrast: WITHOUT eom the same start rolls on the anchor's own DOM (30), giving Mar-30 not Mar-31.
  const auto plain = b::swap_periods_to(vd, "EUR", mat, "6M", "ModifiedFollowing", 2);
  EXPECT_EQ(b::iso(plain[0].second), "2027-03-30");
}

// (c) A FRONT stub puts the odd (short) period FIRST; a back stub puts it LAST. spot 2026-09-30, annual
// to 2028-03-31 -> one regular 12m period and a 6m stub.
TEST(ScheduleStubEom, FrontStubOddPeriodFirst) {
  const b::Date vd = b::Date::from_iso("2026-09-28");
  const b::Date mat = b::Date::from_iso("2028-03-31");

  b::ScheduleRule front;
  front.side = b::StubSide::Front;
  const auto f = b::swap_periods_to(vd, "EUR", mat, "1Y", "ModifiedFollowing", 2, front);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(b::iso(f[0].first), "2026-09-30");
  EXPECT_EQ(b::iso(f[0].second), "2027-03-31");   // the odd (short ~6m) period is FIRST
  EXPECT_EQ(b::iso(f[1].second), "2028-03-31");
  EXPECT_LT(f[0].second - f[0].first, f[1].second - f[1].first);  // first period is the short one

  // Back stub (default) on the same span puts the odd (short) period LAST.
  const auto back = b::swap_periods_to(vd, "EUR", mat, "1Y", "ModifiedFollowing", 2);
  ASSERT_EQ(back.size(), 2u);
  EXPECT_EQ(b::iso(back[0].second), "2027-09-30");  // regular 12m first
  EXPECT_EQ(b::iso(back[1].second), "2028-03-31");
  EXPECT_LT(back[1].second - back[1].first, back[0].second - back[0].first);  // last period is the short one
}

// (d) Roll-day anchor: regular boundaries land on the chosen day-of-month (15), not the anchor's own DOM.
TEST(ScheduleStubEom, RollDayAnchorLandsOnDom) {
  const b::Date vd = b::Date::from_iso("2026-09-28");  // spot 2026-09-30 (DOM 30)
  const b::Date mat = b::Date::from_iso("2028-01-20");

  b::ScheduleRule rule;
  rule.roll_dom = 15;
  const auto per = b::swap_periods_to(vd, "EUR", mat, "6M", "ModifiedFollowing", 2, rule);
  ASSERT_GE(per.size(), 2u);
  // Every interior boundary (all but the final maturity end) is on the 15th (dates chosen so adjust is a
  // no-op business day).
  EXPECT_EQ(b::iso(per[0].second), "2027-03-15");
  EXPECT_EQ(b::iso(per[1].second), "2027-09-15");
  for (std::size_t i = 0; i + 1 < per.size(); ++i)
    EXPECT_EQ(per[i].second.day(), 15u) << "interior boundary " << i;
  EXPECT_EQ(b::iso(per.back().second), "2028-01-20");  // final period still ends exactly at maturity
}
