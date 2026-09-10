// E5 taxonomy: T6 regression (fails on the reverted bug)
// Gate for the DATA-driven holiday calendars (conventions/conventions.json `calendars[].holidays`,
// codegen'd to kCalendars/kHolidayRules, interpreted by swaps/build/calendar.hpp). Every golden value below
// was captured from the PRE-MIGRATION hard-coded calendar.hpp (itself the parity-pinned transcription of
// server/calendars.py) BEFORE the rules moved to data — so this test pins the migration bit-for-bit:
// spot holidays/non-holidays per calendar, the Juneteenth from_year, the Good-Friday bond-vs-Fed split, the
// EURUSD join, the observance year-bucket quirk, and an exhaustive per-year holiday count over 2024-2036.
// QuantLib-free (swaps_tests binary). Companion of build_calendar_test.cpp (which must pass unchanged).
#include <gtest/gtest.h>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/build/calendar.hpp"
#include "swaps/conventions_data.hpp"

namespace b = swaps::build;
namespace cvd = swaps::conventions;

namespace {
std::set<long> serials(const std::string& csv) {  // "2025-01-01,2025-01-20,..." -> set of day serials
  std::set<long> out;
  std::stringstream ss(csv);
  std::string iso;
  while (std::getline(ss, iso, ',')) out.insert(b::Date::from_iso(iso).serial());
  return out;
}
void check_holidays(const std::string& cal, int y, const std::string& golden) {
  EXPECT_EQ(b::holidays_serial(cal, y), serials(golden)) << cal << " " << y;
}
bool bd(const std::string& cal, const std::string& iso) {
  return b::is_business_day(cal, b::Date::from_iso(iso));
}
}  // namespace

// Raw per-year holiday sets (including any that fall on a weekend), captured from the old implementation.
TEST(CalendarData, GoldenHolidaySets) {
  // USD = SIFMA bond market. SIFMA recommended the Juneteenth close from 2022 (QuantLib GovernmentBond agrees);
  // 2021-06-18 is therefore OPEN. Good Friday 2021-04-02 is the FIRST Friday of April (jobs report) = early close,
  // not a closure. A Veterans Day on a Saturday (2028-11-11) is NOT moved to the Friday (raw set keeps the Saturday).
  check_holidays("USD", 2020, "2020-01-01,2020-01-20,2020-02-17,2020-04-10,2020-05-25,2020-07-03,2020-09-07,2020-10-12,2020-11-11,2020-11-26,2020-12-25");
  check_holidays("USD", 2021, "2021-01-01,2021-01-18,2021-02-15,2021-05-31,2021-07-05,2021-09-06,2021-10-11,2021-11-11,2021-11-25,2021-12-24");
  // Jan-1-2022 is a Saturday: the observed Friday 2021-12-31 lands in 2022's year-bucket (legacy quirk).
  check_holidays("USD", 2022, "2021-12-31,2022-01-17,2022-02-21,2022-04-15,2022-05-30,2022-06-20,2022-07-04,2022-09-05,2022-10-10,2022-11-11,2022-11-24,2022-12-26");
  check_holidays("USD", 2024, "2024-01-01,2024-01-15,2024-02-19,2024-03-29,2024-05-27,2024-06-19,2024-07-04,2024-09-02,2024-10-14,2024-11-11,2024-11-28,2024-12-25");
  check_holidays("USD", 2028, "2027-12-31,2028-01-17,2028-02-21,2028-04-14,2028-05-29,2028-06-19,2028-07-04,2028-09-04,2028-10-09,2028-11-11,2028-11-23,2028-12-25");
  check_holidays("USD", 2033, "2032-12-31,2033-01-17,2033-02-21,2033-04-15,2033-05-30,2033-06-20,2033-07-04,2033-09-05,2033-10-10,2033-11-11,2033-11-24,2033-12-26");
  // USD-SOFR = the SIFMA rule set WITHOUT the first-Friday Good Friday exception: SOFR is never published on
  // Good Friday (2021-04-02 closed here, open on the bond market), otherwise identical.
  check_holidays("USD-SOFR", 2021, "2021-01-01,2021-01-18,2021-02-15,2021-04-02,2021-05-31,2021-07-05,2021-09-06,2021-10-11,2021-11-11,2021-11-25,2021-12-24");
  check_holidays("USD-SOFR", 2028, "2027-12-31,2028-01-17,2028-02-21,2028-04-14,2028-05-29,2028-06-19,2028-07-04,2028-09-04,2028-10-09,2028-11-11,2028-11-23,2028-12-25");
  // USD-FED = Fedwire: NO Good Friday; a Saturday holiday stays put (2021-06-19, 2021-12-25 unshifted).
  check_holidays("USD-FED", 2020, "2020-01-01,2020-01-20,2020-02-17,2020-05-25,2020-07-04,2020-09-07,2020-10-12,2020-11-11,2020-11-26,2020-12-25");
  check_holidays("USD-FED", 2021, "2021-01-01,2021-01-18,2021-02-15,2021-05-31,2021-06-19,2021-07-05,2021-09-06,2021-10-11,2021-11-11,2021-11-25,2021-12-25");
  check_holidays("USD-FED", 2022, "2022-01-01,2022-01-17,2022-02-21,2022-05-30,2022-06-20,2022-07-04,2022-09-05,2022-10-10,2022-11-11,2022-11-24,2022-12-26");
  check_holidays("USD-FED", 2028, "2028-01-01,2028-01-17,2028-02-21,2028-05-29,2028-06-19,2028-07-04,2028-09-04,2028-10-09,2028-11-11,2028-11-23,2028-12-25");
  // EUR = TARGET: fixed set + Easter pair, never observance-shifted.
  check_holidays("EUR", 2021, "2021-01-01,2021-04-02,2021-04-05,2021-05-01,2021-12-25,2021-12-26");
  check_holidays("EUR", 2024, "2024-01-01,2024-03-29,2024-04-01,2024-05-01,2024-12-25,2024-12-26");
  check_holidays("EUR", 2033, "2033-01-01,2033-04-15,2033-04-18,2033-05-01,2033-12-25,2033-12-26");
}

TEST(CalendarData, JuneteenthFromYear) {
  EXPECT_TRUE(bd("USD", "2020-06-19"));       // a Friday, but pre-2021: not yet a holiday
  EXPECT_TRUE(bd("USD-FED", "2020-06-19"));
  EXPECT_TRUE(bd("USD", "2021-06-18"));       // 2021: not yet a SIFMA close (from 2022; a Saturday anyway)
  EXPECT_TRUE(bd("USD-FED", "2021-06-18"));
  EXPECT_FALSE(bd("USD", "2024-06-19"));      // a Wednesday: closed everywhere
  EXPECT_FALSE(bd("USD-FED", "2024-06-19"));
  EXPECT_FALSE(bd("USD", "2027-06-18"));      // 2027: Sat again — same split as 2021
  EXPECT_TRUE(bd("USD-FED", "2027-06-18"));
}

TEST(CalendarData, GoodFridayBondVsFed) {
  // 2026-04-03 is the FIRST Friday of April (jobs report): SIFMA recommends an early close, not a closure
  // (QuantLib GovernmentBond: Good Friday closed only when d > 7); SOFR is still not published that day.
  EXPECT_TRUE(bd("USD", "2026-04-03"));
  EXPECT_FALSE(bd("USD-SOFR", "2026-04-03"));
  EXPECT_FALSE(bd("USD", "2027-03-26"));       // a Good Friday that is not the first Friday: closed
  EXPECT_TRUE(bd("USD-FED", "2026-04-03"));    // Fedwire is open
  EXPECT_FALSE(bd("USD", "2024-03-29"));
  EXPECT_TRUE(bd("USD-FED", "2024-03-29"));
  EXPECT_FALSE(bd("EUR", "2024-03-29"));       // TARGET closes too
}

TEST(CalendarData, EurusdJoin) {
  EXPECT_FALSE(bd("EURUSD", "2026-05-01"));  // TARGET May Day (a Friday): USD open, join closed
  EXPECT_TRUE(bd("USD", "2026-05-01"));
  EXPECT_FALSE(bd("EURUSD", "2026-04-06"));  // Easter Monday: EUR-only holiday closes the join
  EXPECT_TRUE(bd("USD", "2026-04-06"));
  EXPECT_FALSE(bd("EURUSD", "2026-11-26")); // US Thanksgiving: EUR open, join closed
  EXPECT_TRUE(bd("EUR", "2026-11-26"));
  EXPECT_TRUE(bd("EURUSD", "2026-05-04"));  // an ordinary Monday: open on both
}

// The observance year-bucket quirk, kept bit-for-bit: Jan-1-2022 (Saturday) is observed 2021-12-31 in the
// 2022 rule bucket, but business-day lookup buckets by the queried date's own year — so 2021-12-31 stays a
// business day on the bond calendars, and Jan-3-2022 (Monday) is an ordinary business day.
TEST(CalendarData, ObservanceYearBucketQuirk) {
  EXPECT_TRUE(bd("USD", "2021-12-31"));
  EXPECT_TRUE(bd("USD-SOFR", "2021-12-31"));
  EXPECT_TRUE(bd("USD", "2022-01-03"));
  EXPECT_TRUE(bd("USD-FED", "2021-12-31"));  // Fed never shifts a Saturday holiday at all
  EXPECT_TRUE(bd("USD-FED", "2022-01-03"));  // Fed shifts SUNDAY only: Sat Jan-1 stays put, Monday is open
  // Contrast with a SUNDAY New Year (2023): every US calendar observes Monday Jan-2.
  EXPECT_FALSE(bd("USD", "2023-01-02"));
  EXPECT_FALSE(bd("USD-FED", "2023-01-02"));
}

// (The "ExhaustiveYearlyHolidayCounts" table that lived here was captured from the ENGINE's own output and
// enshrined wrong rules — GBP with no Saturday substitution, Juneteenth 2021, Veterans Day Saturdays. It was
// deleted on 2026-09-09 and replaced by tests/calendar_ql_oracle_test.cpp: every calendar day-by-day against
// QuantLib. See tests/oracle_assertions.lock for the deliberate reduction.)
