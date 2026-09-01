// FixingSeries: the per-index realized-fixing time series with an as-of (on-or-before) query — the
// semantic the flat pricing/fixings.hpp FixingTable lacks. These tests pin exact lookup, the as-of query
// across a deliberate gap in the series, the before-first boundary, and latest()/count().
#include <gtest/gtest.h>

#include <stdexcept>

#include "swaps/build/date.hpp"
#include "swaps/market/fixing_series.hpp"

namespace mkt = swaps::market;
namespace b = swaps::build;

TEST(FixingSeries, AsOfSpansGapAndExactLookup) {
  mkt::FixingSeries s("USD-SOFR");

  // Ascending SOFR fixings with a deliberate GAP on 2026-08-05 (weekend/holiday-style hole).
  const b::Date d0803 = b::Date::from_iso("2026-08-03");
  const b::Date d0804 = b::Date::from_iso("2026-08-04");
  const b::Date d0805 = b::Date::from_iso("2026-08-05");  // NO fixing here
  const b::Date d0806 = b::Date::from_iso("2026-08-06");

  s.add(d0803, 0.0431);
  s.add(d0804, 0.0432);
  s.add(d0806, 0.0433);

  EXPECT_EQ(s.count(), 3);
  EXPECT_FALSE(s.empty());

  // Exact lookup returns the exact rate on that exact date.
  EXPECT_NEAR(s.at(d0803), 0.0431, 1e-12);
  EXPECT_NEAR(s.at(d0804), 0.0432, 1e-12);
  EXPECT_NEAR(s.at(d0806), 0.0433, 1e-12);

  // Exact-hole date has no exact fixing.
  EXPECT_FALSE(s.has(d0805));
  EXPECT_THROW(s.at(d0805), std::out_of_range);

  // as_of across the gap: 08-05 has no fixing, so the most recent ON OR BEFORE is 08-04's value.
  EXPECT_TRUE(s.has_as_of(d0805));
  EXPECT_NEAR(s.as_of(d0805), 0.0432, 1e-12);

  // as_of on an exact date returns that date's value.
  EXPECT_NEAR(s.as_of(d0806), 0.0433, 1e-12);

  // as_of after the last known fixing carries the last value forward.
  EXPECT_NEAR(s.as_of(b::Date::from_iso("2026-08-10")), 0.0433, 1e-12);

  // Before the first fixing: nothing is available as-of, and as_of throws.
  const b::Date before = b::Date::from_iso("2026-08-01");
  EXPECT_FALSE(s.has_as_of(before));
  EXPECT_THROW(s.as_of(before), std::out_of_range);

  // latest() / latest_date() track the most recent entry.
  EXPECT_NEAR(s.latest(), 0.0433, 1e-12);
  EXPECT_EQ(s.latest_date(), int(d0806.serial()));
}

TEST(FixingSeries, EmptySeries) {
  mkt::FixingSeries s("USD-SOFR");
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.count(), 0);
  EXPECT_FALSE(s.has_as_of(b::Date::from_iso("2026-08-04")));
  EXPECT_THROW(s.latest(), std::out_of_range);
  EXPECT_THROW(s.latest_date(), std::out_of_range);
}

TEST(FixingSeries, AddOverwritesAndSerialOverloads) {
  mkt::FixingSeries s("USD-SOFR");
  const int d = int(b::Date::from_iso("2026-08-04").serial());
  s.add(d, 0.0432);
  s.add(d, 0.0400);  // upsert overwrites, does not duplicate
  EXPECT_EQ(s.count(), 1);
  EXPECT_NEAR(s.at(d), 0.0400, 1e-12);
  EXPECT_NEAR(s.as_of(d), 0.0400, 1e-12);
}
