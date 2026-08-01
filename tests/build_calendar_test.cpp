// Golden-value gate for the QuantLib-free construction date engine (include/swaps/build/{date,calendar,
// day_count,schedule}.hpp). Every expected value below is the OUTPUT of the Python compiler it transcribes
// (server/calendars.py + dates.py) — so this pins C++/Python parity for holidays, IMM, schedule rolling,
// day counts, and token resolution. QuantLib-free (in the swaps_tests binary).
#include <gtest/gtest.h>

#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/build/calendar.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"

namespace b = swaps::build;

namespace {
std::set<long> serials(const std::string& csv) {  // "2025-01-01,2025-01-20,..." -> set of day serials
  std::set<long> out;
  std::stringstream ss(csv);
  std::string iso;
  while (std::getline(ss, iso, ',')) out.insert(b::Date::from_iso(iso).serial());
  return out;
}
void check_holidays(const std::string& cal, int y, const std::string& golden) {
  // Direct mirror of Python calendars._holidays (the raw set, INCLUDING any that fall on a weekend).
  EXPECT_EQ(b::holidays_serial(cal, y), serials(golden)) << cal << " " << y;
}
}  // namespace

TEST(BuildCalendar, HolidaysMatchPython) {
  // USD-SOFR = SIFMA bond market (closes Good Friday). Note 2026 Jul-3 (Jul-4 is Sat -> observed Fri).
  check_holidays("USD-SOFR", 2025, "2025-01-01,2025-01-20,2025-02-17,2025-04-18,2025-05-26,2025-06-19,2025-07-04,2025-09-01,2025-10-13,2025-11-11,2025-11-27,2025-12-25");
  check_holidays("USD-SOFR", 2026, "2026-01-01,2026-01-19,2026-02-16,2026-04-03,2026-05-25,2026-06-19,2026-07-03,2026-09-07,2026-10-12,2026-11-11,2026-11-26,2026-12-25");
  check_holidays("USD-SOFR", 2027, "2027-01-01,2027-01-18,2027-02-15,2027-03-26,2027-05-31,2027-06-18,2027-07-05,2027-09-06,2027-10-11,2027-11-11,2027-11-25,2027-12-24");
  // USD-FED = Fedwire: NO Good Friday, and a Saturday holiday is NOT taken on the Friday (2026 has no Jul-3).
  check_holidays("USD-FED", 2025, "2025-01-01,2025-01-20,2025-02-17,2025-05-26,2025-06-19,2025-07-04,2025-09-01,2025-10-13,2025-11-11,2025-11-27,2025-12-25");
  check_holidays("USD-FED", 2026, "2026-01-01,2026-01-19,2026-02-16,2026-05-25,2026-06-19,2026-07-04,2026-09-07,2026-10-12,2026-11-11,2026-11-26,2026-12-25");
  check_holidays("USD-FED", 2027, "2027-01-01,2027-01-18,2027-02-15,2027-05-31,2027-06-19,2027-07-05,2027-09-06,2027-10-11,2027-11-11,2027-11-25,2027-12-25");
  // EUR = TARGET.
  check_holidays("EUR", 2025, "2025-01-01,2025-04-18,2025-04-21,2025-05-01,2025-12-25,2025-12-26");
  check_holidays("EUR", 2026, "2026-01-01,2026-04-03,2026-04-06,2026-05-01,2026-12-25,2026-12-26");
  check_holidays("EUR", 2027, "2027-01-01,2027-03-26,2027-03-29,2027-05-01,2027-12-25,2027-12-26");
}

TEST(BuildCalendar, ResolveMatchesPython) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  auto R = [&](const std::string& t) { return b::iso(b::resolve(t, vd)); };
  EXPECT_EQ(R("U27"), "2027-09-15");     // IMM 3rd-Wed Sep-2027
  EXPECT_EQ(R("M28"), "2028-06-21");
  EXPECT_EQ(R("3m"), "2026-10-08");
  EXPECT_EQ(R("2y"), "2028-07-10");
  EXPECT_EQ(R("18m"), "2028-01-10");
  EXPECT_EQ(R("on"), "2026-07-09");
  EXPECT_EQ(R("tn"), "2026-07-10");
  EXPECT_EQ(R("sp"), "2026-07-08");
  EXPECT_EQ(R("2027-09-16"), "2027-09-16");
}

TEST(BuildCalendar, ScheduleAndDayCounts) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::Date mat = b::resolve("2y", vd);
  const auto per = b::swap_periods_to(vd, "USD-SOFR", mat, "1Y", "ModifiedFollowing", 2);
  ASSERT_EQ(per.size(), 2u);
  EXPECT_EQ(b::iso(per[0].first), "2026-07-10");
  EXPECT_EQ(b::iso(per[0].second), "2027-07-12");
  EXPECT_EQ(b::iso(per[1].first), "2027-07-12");
  EXPECT_EQ(b::iso(per[1].second), "2028-07-10");

  EXPECT_NEAR(b::year_frac("ACT/360", b::Date::from_iso("2026-07-10"), b::Date::from_iso("2027-07-12")),
              1.0194444444444444, 1e-15);
  EXPECT_NEAR(b::year_frac("30E/360", b::Date::from_iso("2026-07-31"), b::Date::from_iso("2027-01-31")),
              0.5, 1e-15);
  EXPECT_EQ(b::iso(b::adjust("EURUSD", b::Date::from_iso("2025-04-18"), "ModifiedFollowing")), "2025-04-22");
  EXPECT_EQ(b::iso(b::advance_bd("USD-SOFR", b::Date::from_iso("2026-07-08"), 2)), "2026-07-10");
}
