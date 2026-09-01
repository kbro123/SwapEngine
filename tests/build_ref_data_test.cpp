// Phase 3: the first-class reference-data OBJECTS (build/ref_data.hpp) delegate EXACTLY to the parity-tested
// free functions they wrap — so introducing them changes no behaviour, and the model can hold typed
// Calendar / DayCount / Index / Convention instead of raw string ids.
#include <gtest/gtest.h>

#include "swaps/build/ref_data.hpp"

namespace b = swaps::build;

TEST(RefData, CalendarDelegatesToFreeFunctions) {
  const b::Calendar cal{"USD"};
  const b::Date d = b::Date::from_iso("2026-07-03");
  EXPECT_EQ(cal.is_business_day(d), b::is_business_day("USD", d));
  EXPECT_EQ(cal.adjust(d).serial(), b::adjust("USD", d).serial());
  EXPECT_EQ(cal.advance_bd(d, 2).serial(), b::advance_bd("USD", d, 2).serial());
  EXPECT_TRUE(b::Calendar{""}.empty());
}

TEST(RefData, DayCountDelegatesToFreeFunction) {
  const b::DayCount dc{"ACT/360"};
  const b::Date d1 = b::Date::from_iso("2026-01-15"), d2 = b::Date::from_iso("2026-07-15");
  EXPECT_DOUBLE_EQ(dc.year_frac(d1, d2), b::year_frac("ACT/360", d1, d2));
}

TEST(RefData, IndexExposesItsConventions) {
  const b::Index sofr("USD-SOFR");
  EXPECT_TRUE(sofr.known());
  EXPECT_TRUE(sofr.is_overnight());
  EXPECT_EQ(sofr.currency(), "USD");
  EXPECT_EQ(sofr.day_count().id, b::index_day_count("USD-SOFR"));     // typed sub-object matches the free fn
  EXPECT_EQ(sofr.fixing_calendar().id, b::index_calendar("USD-SOFR"));

  const b::Index unknown("NOT-AN-INDEX");
  EXPECT_FALSE(unknown.known());
  EXPECT_EQ(unknown.day_count().id, "ACT/360");  // sensible fallback, same as the free function
  EXPECT_TRUE(unknown.fixing_calendar().empty());
}

TEST(RefData, ConventionResolvesToSwapConv) {
  const b::Convention c{"USD", "USD-SOFR", 0.0};
  const b::SwapConv sc = c.resolve();
  const b::SwapConv ref = b::swap_conv("USD", 0.0, "USD-SOFR");
  EXPECT_EQ(sc.calendar, ref.calendar);
  EXPECT_EQ(sc.fixed_dc, ref.fixed_dc);
  EXPECT_EQ(sc.float_dc, ref.float_dc);
  EXPECT_EQ(sc.spot_lag, ref.spot_lag);
}

TEST(RefData, IndexProducesItsConventionObjectToObject) {
  const b::Index sofr("USD-SOFR");
  const b::Convention conv = sofr.par_convention();  // Index -> Convention, no raw swap_conv() call
  const b::SwapConv ref = b::swap_conv("USD", 0.0, "USD-SOFR");
  // The Convention resolves to the same SwapConv as the procedural path...
  EXPECT_EQ(conv.resolve().calendar, ref.calendar);
  EXPECT_EQ(conv.index, "USD-SOFR");
  EXPECT_EQ(conv.currency, "USD");
  // ...and its typed accessors expose each convention as its own object.
  EXPECT_EQ(conv.calendar().id, ref.calendar);
  EXPECT_EQ(conv.fixed_day_count().id, ref.fixed_dc);
  EXPECT_EQ(conv.float_day_count().id, ref.float_dc);
  EXPECT_EQ(conv.spot_lag(), ref.spot_lag);
}
