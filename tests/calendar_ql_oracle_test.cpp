// @oracle-test — EVERY conventions-DB calendar vs QuantLib's calendar of the same market, DAY BY DAY
// E5 taxonomy: T1 oracle (engine number vs an independent number)
// (engine is_business_day vs QuantLib::Calendar::isBusinessDay), 2024-01-01 .. the oracle coverage year.
// DO NOT DELETE OR WEAKEN without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
//
// This is the independent source the calendar data was missing (PRINCIPLES.md P8/P11): until 2026-09-09 the
// per-year holiday-count goldens were captured from the engine's OWN output, so a wrong observance rule
// (e.g. GBP with no Saturday-to-Monday substitution) was enshrined rather than caught. The DB row's
// `quantlib` field names the QuantLib calendar; the coverage year is where QuantLib's own explicit
// tabulation ends for markets it tabulates rather than rules (China 2024, India 2025, Saudi 2029, Turkey
// 2034; rule-based calendars are compared through 2035).
#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include <ql/quantlib.hpp>

#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"
#include "swaps/conventions_data.hpp"

namespace b = swaps::build;
namespace cvd = swaps::conventions;

namespace {

struct Oracle {
  QuantLib::Calendar cal;
  int coverage_year;  // last year QuantLib is authoritative for this market
};

std::map<std::string, Oracle> oracles() {
  using namespace QuantLib;
  return {
      {"USD",      {UnitedStates(UnitedStates::GovernmentBond), 2035}},
      {"USD-SOFR", {UnitedStates(UnitedStates::SOFR), 2035}},
      {"USD-FED",  {UnitedStates(UnitedStates::FederalReserve), 2035}},
      {"EUR",      {TARGET(), 2035}},
      {"EURUSD",   {JointCalendar(UnitedStates(UnitedStates::GovernmentBond), TARGET(), JoinHolidays), 2035}},
      {"GBP",      {UnitedKingdom(UnitedKingdom::Settlement), 2035}},
      {"JPY",      {Japan(), 2035}},
      {"AUD",      {Australia(Australia::Settlement), 2035}},
      {"CAD",      {Canada(Canada::Settlement), 2035}},
      {"CHF",      {Switzerland(), 2035}},
      {"CNY",      {China(China::IB), 2024}},
      {"INR",      {India(), 2025}},
      {"BRL",      {Brazil(Brazil::Settlement), 2035}},
      {"MXN",      {Mexico(), 2035}},
      {"KRW",      {SouthKorea(SouthKorea::Settlement), 2035}},
      {"ZAR",      {SouthAfrica(), 2035}},
      {"TRY",      {Turkey(), 2034}},
      {"IDR",      {Indonesia(Indonesia::IDX), 2025}},
      {"SAR",      {SaudiArabia(SaudiArabia::Tadawul), 2029}},
  };
}

// Documented exceptions: dates where the DB deliberately differs from QuantLib, each with its reason.
// (An undocumented mismatch fails the test; these are the ONLY tolerated ones.)
const std::map<std::string, std::map<std::string, std::string>>& exceptions() {
  static const std::map<std::string, std::map<std::string, std::string>> e = {
      {"IDR", {{"2024-05-01", "Labour Day is an Indonesian public holiday (since 2014); QuantLib's Indonesia calendar omits it"},
               {"2025-05-01", "Labour Day (as above)"}}},
  };
  return e;
}
// Calendars with NO usable QuantLib oracle, and why (they stay data-only until a vendor feed is wired):
//   ARS — QuantLib's Argentina is the Merval EXCHANGE calendar: it lacks Carnival, Malvinas Day, Flag Day and
//         Sovereignty Day (national holidays) and closes on half-days; not a settlement oracle.
//   RUB — QuantLib's Russia settlement tabulation ends in 2020.

QuantLib::Date to_ql(const b::Date& d) { return QuantLib::Date(QuantLib::Day(d.day()), QuantLib::Month(d.month()), d.year()); }

}  // namespace

TEST(CalendarQuantLibOracle, EveryDbCalendarMatchesQuantLibDayByDay) {
  const auto ors = oracles();
  std::vector<std::string> unchecked;
  for (const auto& row : cvd::kCalendars) {
    const std::string id(row.id);
    if (id == "NONE") continue;
    const auto it = ors.find(id);
    if (it == ors.end()) { unchecked.push_back(id); continue; }
    const auto& o = it->second;
    std::vector<std::string> mism;
    int n = 0, total = 0;
    for (b::Date d = b::Date::from_iso("2024-01-01"); d.year() <= o.coverage_year; d = d.plus_days(1)) {
      const bool ours = b::is_business_day(id, d);
      const bool ql = o.cal.isBusinessDay(to_ql(d));
      ++n;
      if (ours == ql) continue;
      const auto ex = exceptions().find(id);
      if (ex != exceptions().end() && ex->second.count(b::iso(d))) continue;  // documented, tolerated
      ++total;
      if (mism.size() < 12) mism.push_back(b::iso(d) + (ours ? " ours=open ql=CLOSED" : " ours=CLOSED ql=open"));
    }
    EXPECT_EQ(total, 0) << id << " vs QuantLib " << o.cal.name() << " through " << o.coverage_year << ": " << total
                        << " mismatching days of " << n << "; first: " << [&] {
                             std::string s;
                             for (const auto& m : mism) s += "\n    " + m;
                             return s;
                           }();
  }
  // Calendars with a DOCUMENTED reason for having no oracle (see the comment above); anything else is a failure.
  static const std::set<std::string> documented_no_oracle = {"ARS", "RUB"};
  std::vector<std::string> undocumented;
  for (const auto& u : unchecked) if (!documented_no_oracle.count(u)) undocumented.push_back(u);
  unchecked.swap(undocumented);
  EXPECT_TRUE(unchecked.empty()) << "DB calendars with no QuantLib oracle mapping and no documented reason: " << [&] {
    std::string s;
    for (const auto& u : unchecked) s += u + " ";
    return s;
  }();
}
