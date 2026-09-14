#pragma once
// QuantLib glue for the market-conventions DB. Maps the DB's convention STRINGS (from the codegen'd
// swaps::conventions header) to QuantLib objects, so the reference builders PULL their conventions from the
// database (conventions/conventions.json) instead of hardcoding literals. See conventions/README or the
// CLAUDE.md "pull from the DB" rule.
//
// Tests-only: this depends on QuantLib. The engine core (include/swaps/**) stays QuantLib-free and takes
// only extracted dates + accruals, so the generated swaps::conventions header (which it CAN include) is
// pure data with no QuantLib types.
#include <ql/quantlib.hpp>

#include <stdexcept>
#include <string>
#include <string_view>

#include "swaps/conventions_data.hpp"

namespace swaps::refbuild::conv {

inline QuantLib::DayCounter day_counter(std::string_view dc) {
  using namespace QuantLib;
  if (dc == "ACT/360") return Actual360();
  if (dc == "30E/360") return Thirty360(Thirty360::European);   // Eurobond basis — EUR IRS fixed leg
  if (dc == "30U/360") return Thirty360(Thirty360::BondBasis);  // US bond basis (not used by EUR IRS)
  if (dc == "ACT/365F") return Actual365Fixed();
  throw std::runtime_error("conventions_ql: unknown day_count '" + std::string(dc) + "'");
}

inline QuantLib::BusinessDayConvention bdc(std::string_view b) {
  using namespace QuantLib;
  if (b == "Following") return Following;
  if (b == "ModifiedFollowing") return ModifiedFollowing;
  if (b == "Preceding") return Preceding;
  if (b == "ModifiedPreceding") return ModifiedPreceding;
  throw std::runtime_error("conventions_ql: unknown bdc '" + std::string(b) + "'");
}

// Calendar id (from the DB) -> QuantLib calendar, matching conventions.json "calendars".*.quantlib.
// The USD overnight world splits three ways (desk 2026-07): USD = SIFMA/US-government-securities (bond
// market, used for the EUR/USD FX & xccy joint calendar); USD-SOFR = QuantLib's dedicated SOFR fixing
// calendar (SIFMA incl. Good Friday close); USD-FED = Federal Reserve (Fedwire, for EFFR/Fed Funds).
inline QuantLib::Calendar calendar(std::string_view cid) {
  using namespace QuantLib;
  if (cid == "EUR") return TARGET();
  if (cid == "USD") return UnitedStates(UnitedStates::GovernmentBond);
  if (cid == "USD-SOFR") return Sofr(RelinkableHandle<YieldTermStructure>()).fixingCalendar();
  if (cid == "USD-FED") return UnitedStates(UnitedStates::FederalReserve);
  if (cid == "EURUSD") return JointCalendar(UnitedStates(UnitedStates::GovernmentBond), TARGET());
  if (cid == "USD+USD-FED") return JointCalendar(UnitedStates(UnitedStates::GovernmentBond), UnitedStates(UnitedStates::FederalReserve));
  // The same QuantLib calendars tests/calendar_ql_oracle_test.cpp checks against the DB day by day.
  if (cid == "GBP") return UnitedKingdom(UnitedKingdom::Settlement);  // not ql::Sonia's fixing calendar (Exchange)
  if (cid == "JPY") return Japan();
  if (cid == "AUD") return Australia(Australia::Settlement);
  if (cid == "CAD") return Canada(Canada::Settlement);
  if (cid == "CHF") return Switzerland();
  throw std::runtime_error("conventions_ql: unknown calendar '" + std::string(cid) + "'");
}

// ISO code -> QuantLib Currency: a LABEL for QuantLib's index objects (name / fixing-store key). It carries no market
// convention, so it is not a DB field.
inline QuantLib::Currency currency(std::string_view iso) {
  using namespace QuantLib;
  if (iso == "USD") return USDCurrency();
  if (iso == "EUR") return EURCurrency();
  if (iso == "GBP") return GBPCurrency();
  if (iso == "JPY") return JPYCurrency();
  if (iso == "AUD") return AUDCurrency();
  if (iso == "CAD") return CADCurrency();
  if (iso == "CHF") return CHFCurrency();
  throw std::runtime_error("conventions_ql: unmapped currency '" + std::string(iso) + "'");
}

// Frequency/tenor token ("3M", "6M", "1Y") -> QuantLib Period.
inline QuantLib::Period period(std::string_view tok) {
  using namespace QuantLib;
  if (tok.size() < 2) throw std::runtime_error("conventions_ql: bad period '" + std::string(tok) + "'");
  const int n = std::stoi(std::string(tok.substr(0, tok.size() - 1)));
  switch (tok.back()) {
    case 'D': return Period(n, Days);
    case 'W': return Period(n, Weeks);
    case 'M': return Period(n, Months);
    case 'Y': return Period(n, Years);
  }
  throw std::runtime_error("conventions_ql: bad period '" + std::string(tok) + "'");
}

// DB record accessors that throw (rather than return nullopt) so a typo'd product id fails loudly.
inline swaps::conventions::ProductConv product(std::string_view id) {
  if (auto p = swaps::conventions::product(id)) return *p;
  throw std::runtime_error("conventions_ql: unknown product '" + std::string(id) + "'");
}
inline swaps::conventions::IndexConv index(std::string_view id) {
  if (auto i = swaps::conventions::index(id)) return *i;
  throw std::runtime_error("conventions_ql: unknown index '" + std::string(id) + "'");
}

}  // namespace swaps::refbuild::conv
