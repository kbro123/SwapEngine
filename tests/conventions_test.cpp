// Conformance gate for the market-conventions DB (conventions/conventions.json -> conventions_data.hpp).
//
// The DB is the single source of truth the reference builders + the Python web layer both pull from. This
// test makes it AUTHORITATIVE against QuantLib: for every index/product we assert QuantLib's own encoded
// conventions (day count, fixing calendar, fixing days) agree with the DB. If someone bumps QuantLib and an
// index convention shifts, or edits the DB away from the market, this fails -- the DB can never silently
// drift from the instruments the oracle prices.
//
// NOTE (open item, desk question 2026-07): the DB currently carries ONE "USD" calendar, but QuantLib splits
// the USD overnight world three ways -- SOFR fixes on a dedicated "SOFR fixing calendar", Fed Funds on the
// "Federal Reserve" calendar, and FX/xccy settlement on "US settlement". Until the desk confirms how to
// encode these, we assert the USD *day counts* (unambiguous, ACT/360) but NOT the USD calendars. The EUR
// side (TARGET everywhere) and the EURIBOR fixing lag ARE asserted.

#include <ql/quantlib.hpp>

#include <gtest/gtest.h>

#include "conventions_ql.hpp"

using namespace QuantLib;
namespace conv = swaps::refbuild::conv;
namespace db = swaps::conventions;

namespace {

// A DB day-count string maps to the SAME QuantLib day counter the index carries.
void expect_same_daycount(const std::string& index_id, const DayCounter& ql_dc) {
  const auto idx = conv::index(index_id);
  EXPECT_EQ(conv::day_counter(idx.day_count).name(), ql_dc.name())
      << "day-count drift for " << index_id;
}

}  // namespace

// ---- Indices: day count / calendar / fixing days vs QuantLib's own index objects. -------------------

TEST(Conventions, IndexDayCountsMatchQuantLib) {
  const auto h = RelinkableHandle<YieldTermStructure>();
  expect_same_daycount("USD-SOFR", Sofr(h).dayCounter());
  expect_same_daycount("USD-FEDFUNDS", FedFunds(h).dayCounter());
  expect_same_daycount("EUR-ESTR", Estr(h).dayCounter());
  expect_same_daycount("EUR-EURIBOR-3M", Euribor3M(h).dayCounter());
  expect_same_daycount("EUR-EURIBOR-6M", Euribor6M(h).dayCounter());
  // All five are money-market ACT/360.
  EXPECT_EQ(conv::index("USD-SOFR").day_count, std::string("ACT/360"));
}

TEST(Conventions, EurIndexCalendarsAreTarget) {
  const auto h = RelinkableHandle<YieldTermStructure>();
  // Every EUR index fixes on TARGET; the DB's "EUR" calendar must resolve to it.
  EXPECT_EQ(conv::calendar(conv::index("EUR-ESTR").calendar).name(), Estr(h).fixingCalendar().name());
  EXPECT_EQ(conv::calendar(conv::index("EUR-EURIBOR-3M").calendar).name(), Euribor3M(h).fixingCalendar().name());
  EXPECT_EQ(conv::calendar(conv::index("EUR-EURIBOR-6M").calendar).name(), Euribor6M(h).fixingCalendar().name());
  EXPECT_EQ(TARGET().name(), conv::calendar("EUR").name());
}

TEST(Conventions, EuriborFixingLagMatchesQuantLib) {
  const auto h = RelinkableHandle<YieldTermStructure>();
  // EURIBOR fixes 2 business days before the accrual start; the DB records fixing_lag = 2.
  EXPECT_EQ(conv::index("EUR-EURIBOR-3M").fixing_lag, static_cast<int>(Euribor3M(h).fixingDays()));
  EXPECT_EQ(conv::index("EUR-EURIBOR-6M").fixing_lag, static_cast<int>(Euribor6M(h).fixingDays()));
  EXPECT_EQ(conv::index("EUR-EURIBOR-3M").fixing_lag, 2);
}

// ---- Products: the exact conventions the reference builders now PULL from the DB. --------------------

TEST(Conventions, EuriborIrsFixedLegPullsExpectedQuantLibConventions) {
  // reference_multicurrency.hpp builds the EURIBOR IRS fixed leg from these DB fields; assert they map to
  // the QuantLib objects the code previously hardcoded (30U/360, annual, ModFol, TARGET).
  for (const char* pid : {"EUR-EURIBOR-3M-IRS", "EUR-EURIBOR-6M-IRS"}) {
    const auto p = conv::product(pid);
    EXPECT_EQ(conv::day_counter(p.fixed.day_count).name(), Thirty360(Thirty360::BondBasis).name()) << pid;
    EXPECT_EQ(conv::period(p.fixed.frequency), Period(1, Years)) << pid;
    EXPECT_EQ(conv::bdc(p.bdc), ModifiedFollowing) << pid;
    EXPECT_EQ(conv::calendar(p.calendar).name(), TARGET().name()) << pid;
  }
}

TEST(Conventions, OisProductsAreCompoundedAct360Annual) {
  for (const char* pid : {"USD-SOFR-OIS", "USD-FEDFUNDS-OIS", "EUR-ESTR-OIS"}) {
    const auto p = conv::product(pid);
    EXPECT_EQ(p.floating.day_count, std::string("ACT/360")) << pid;
    EXPECT_EQ(p.floating.compounding, std::string("compounded")) << pid;
    EXPECT_EQ(p.floating.frequency, std::string("1Y")) << pid;
    EXPECT_EQ(p.spot_lag, 2) << pid;      // T+2 spot on the currency calendar
    EXPECT_EQ(p.payment_lag, 2) << pid;   // 2 BD payment delay (desk-confirmed)
  }
}

TEST(Conventions, XccyMtmLegRolesMatchDeskSpec) {
  // EUR/USD MtM xccy: USD SOFR leg is FLAT + resettable; EUR ESTR leg carries the basis spread.
  const auto p = conv::product("XCCY-MTM-EURUSD");
  EXPECT_TRUE(p.floating.flat);            // usd_leg (flat, notional-resetting)
  EXPECT_TRUE(p.floating.notional_resets);
  EXPECT_EQ(p.floating.index, std::string("USD-SOFR"));
  EXPECT_EQ(p.spot_lag, 2);
  EXPECT_EQ(p.payment_lag, 2);
}
