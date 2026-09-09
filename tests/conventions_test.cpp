// @oracle-test — validates the conventions DB against QuantLib's own index conventions. DO NOT DELETE OR
// WEAKEN without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
//
// Conformance gate for the market-conventions DB (conventions/conventions.json -> conventions_data.hpp).
//
// The DB is the single source of truth the reference builders + the Python web layer both pull from. This
// test makes it AUTHORITATIVE against QuantLib: for every index/product we assert QuantLib's own encoded
// conventions (day count, fixing calendar, fixing days) agree with the DB. If someone bumps QuantLib and an
// index convention shifts, or edits the DB away from the market, this fails -- the DB can never silently
// drift from the instruments the oracle prices.
//
// The USD overnight world splits three ways (desk 2026-07, now encoded): SOFR fixes on QuantLib's dedicated
// SOFR calendar (DB "USD-SOFR", SIFMA incl. Good Friday close), Fed Funds on the Federal Reserve calendar
// (DB "USD-FED"), and the EUR/USD FX & xccy joint calendar uses SIFMA government-bond (DB "USD") on the USD
// side. All three are asserted below against QuantLib's own index calendars.

#include <ql/quantlib.hpp>

#include <gtest/gtest.h>

#include "conventions_ql.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/pricing/bond.hpp"

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

TEST(Conventions, UsdOvernightCalendarsMatchQuantLib) {
  const auto h = RelinkableHandle<YieldTermStructure>();
  // SOFR fixes on QuantLib's dedicated SOFR calendar; the DB "USD-SOFR" must resolve to exactly it.
  EXPECT_EQ(conv::calendar(conv::index("USD-SOFR").calendar).name(), Sofr(h).fixingCalendar().name());
  // Fed Funds/EFFR fixes on the Federal Reserve calendar (holiday-identical to UnitedStates::FederalReserve).
  EXPECT_EQ(conv::calendar(conv::index("USD-FEDFUNDS").calendar).name(), FedFunds(h).fixingCalendar().name());
  EXPECT_EQ(conv::calendar("USD-FED").name(), UnitedStates(UnitedStates::FederalReserve).name());
  // The SOFR and Fed calendars genuinely differ -- the split is not cosmetic.
  EXPECT_NE(conv::calendar("USD-SOFR").name(), conv::calendar("USD-FED").name());
  // FX/xccy USD side = SIFMA government-bond (the joint EURUSD calendar).
  EXPECT_EQ(conv::calendar("USD").name(), UnitedStates(UnitedStates::GovernmentBond).name());
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
  // the market EUR convention (30E/360 Eurobond basis, annual, ModFol, TARGET).
  for (const char* pid : {"EUR-EURIBOR-3M-IRS", "EUR-EURIBOR-6M-IRS"}) {
    const auto p = conv::product(pid);
    EXPECT_EQ(conv::day_counter(p.fixed.day_count).name(), Thirty360(Thirty360::European).name()) << pid;
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

// ---------------------------------------------------------------------------------------------------
// BOND conventions. The DB carries the one degree of freedom a per-flow exponent cannot express (the
// stub-discount rule, pricing::YieldConvention), so a bond TYPE is a DB id, not a C++ branch. This pins
// the DB against the named builders in build/bond.hpp AND against QuantLib's two compounding modes, so
// the JSON, the builders and the oracle cannot drift apart.
// ---------------------------------------------------------------------------------------------------
TEST(Conventions, BondConventionsDriveTheNamedBuilders) {
  namespace bld = swaps::build;
  namespace px = swaps::pricing;
  const bld::Date value = bld::Date::ymd(2024, 1, 15), settle = bld::Date::ymd(2024, 1, 16),
                  issue = bld::Date::ymd(2019, 8, 15), maturity = bld::Date::ymd(2029, 8, 15);
  const double coupon = 0.025, y = 0.04;

  // Every catalogued convention resolves, and the DB fields land where the kernel reads them.
  const px::YieldConvention street = bld::yield_convention("US-TREASURY");
  EXPECT_EQ(street.freq, 2.0);
  EXPECT_EQ(street.stub, px::StubDiscount::Compound);
  EXPECT_TRUE(street.final_period_simple);
  const px::YieldConvention tsy = bld::yield_convention("US-TREASURY-TSY");
  EXPECT_EQ(tsy.freq, 2.0);
  EXPECT_EQ(tsy.stub, px::StubDiscount::Simple);
  EXPECT_FALSE(tsy.final_period_simple);
  EXPECT_THROW(bld::yield_convention("NO-SUCH-BOND"), std::invalid_argument);

  // DB-driven construction == the hand-written named builder, to the last bit.
  const auto db_street = bld::bond_from_convention("US-TREASURY", value, settle, issue, maturity, coupon);
  const auto db_tsy = bld::bond_from_convention("US-TREASURY-TSY", value, settle, issue, maturity, coupon);
  EXPECT_EQ(px::bond_dirty_from_yield(db_street.yield, y),
            px::bond_dirty_from_yield(bld::us_treasury(value, settle, issue, maturity, coupon).yield, y));
  EXPECT_EQ(px::bond_dirty_from_yield(db_tsy.yield, y),
            px::bond_dirty_from_yield(bld::us_treasury_tsy(value, settle, issue, maturity, coupon).yield, y));

  // ... and the two conventions are the two DIFFERENT numbers QuantLib gives for the same bond.
  Settings::instance().evaluationDate() = QuantLib::Date(15, January, 2024);
  Schedule sch(QuantLib::Date(15, August, 2019), QuantLib::Date(15, August, 2029), Period(Semiannual),
               NullCalendar(), Unadjusted, Unadjusted, DateGeneration::Backward, false);
  DayCounter dc = ActualActual(ActualActual::ISMA, sch);
  FixedRateBond ql(1, 100.0, sch, std::vector<Rate>{coupon}, dc, Following, 100.0,
                   QuantLib::Date(15, August, 2019));
  const QuantLib::Date s = ql.settlementDate();
  EXPECT_NEAR(px::bond_dirty_from_yield(db_street.yield, y) * 100.0,
              BondFunctions::dirtyPrice(ql, y, dc, Compounded, Semiannual, s), 1e-9);
  EXPECT_NEAR(px::bond_dirty_from_yield(db_tsy.yield, y) * 100.0,
              BondFunctions::dirtyPrice(ql, y, dc, SimpleThenCompounded, Semiannual, s), 1e-9);
}

// ---------------------------------------------------------------------------------------------------
// G20 breadth: every new-currency index resolves to its par product + a known calendar, and the par
// product's float leg points back at the index. Pure-DB assertions (swaps::conventions data accessors,
// no QuantLib mapping) so they hold even before conventions_ql.hpp learns the new calendars -- mirrors
// the USD/EUR par_product wiring the reference builders rely on.
// ---------------------------------------------------------------------------------------------------
TEST(Conventions, G20IndicesResolveToParProductAndCalendar) {
  struct Row { const char* index; const char* par_product; const char* calendar; };
  const Row rows[] = {
      {"JPY-TONA", "JPY-TONA-OIS", "JPY"},
      {"GBP-SONIA", "GBP-SONIA-OIS", "GBP"},
      {"AUD-AONIA", "AUD-AONIA-OIS", "AUD"},
      {"AUD-BBSW-3M", "AUD-BBSW-3M-IRS", "AUD"},
      {"AUD-BBSW-6M", "AUD-BBSW-6M-IRS", "AUD"},
      {"CAD-CORRA", "CAD-CORRA-OIS", "CAD"},
      {"CHF-SARON", "CHF-SARON-OIS", "CHF"},
      {"CNY-FR007", "CNY-FR007-IRS", "CNY"},
      {"CNY-SHIBOR-3M", "CNY-SHIBOR-3M-IRS", "CNY"},
      {"INR-MIBOR-ON", "INR-MIBOR-OIS", "INR"},
      {"BRL-CDI", "BRL-CDI-SWAP", "BRL"},
      {"MXN-TIIE-28", "MXN-TIIE-28-IRS", "MXN"},
      {"MXN-FTIIE", "MXN-FTIIE-OIS", "MXN"},
      {"KRW-KOFR", "KRW-KOFR-OIS", "KRW"},
      {"ZAR-ZARONIA", "ZAR-ZARONIA-OIS", "ZAR"},
      {"ZAR-JIBAR-3M", "ZAR-JIBAR-3M-IRS", "ZAR"},
      {"TRY-TLREF", "TRY-TLREF-OIS", "TRY"},
      {"IDR-INDONIA", "IDR-INDONIA-OIS", "IDR"},
      {"RUB-RUONIA", "RUB-RUONIA-OIS", "RUB"},
      {"SAR-SAIBOR-3M", "SAR-SAIBOR-3M-IRS", "SAR"},
      {"ARS-BADLAR", "ARS-BADLAR-IRS", "ARS"},
  };
  for (const auto& r : rows) {
    const auto ix = db::index(r.index);
    ASSERT_TRUE(ix.has_value()) << "missing index " << r.index;
    EXPECT_EQ(ix->par_product, std::string_view(r.par_product)) << r.index;
    // the index's settlement calendar resolves in the DB ...
    const auto cal = db::calendar(ix->calendar);
    ASSERT_TRUE(cal.has_value()) << "index " << r.index << " -> unknown calendar " << ix->calendar;
    EXPECT_EQ(cal->row.id, std::string_view(r.calendar)) << r.index;
    // ... and its par product exists, in the same currency, with the float leg pointing back at it.
    const auto p = db::product(r.par_product);
    ASSERT_TRUE(p.has_value()) << "missing product " << r.par_product;
    EXPECT_EQ(p->floating.index, std::string_view(r.index)) << r.par_product;
    EXPECT_EQ(p->currency, ix->currency) << r.par_product;
  }
}

// The Saudi calendar carries the Islamic Friday/Saturday weekend, unlike the Sat/Sun majors -- guards
// the per-calendar weekend-mask transcription (bit w set == weekday w is a weekend day, Mon=0..Sun=6).
TEST(Conventions, SaudiWeekendIsFridaySaturday) {
  const auto sar = db::calendar("SAR");
  ASSERT_TRUE(sar.has_value());
  EXPECT_NE(sar->row.weekend_mask & (1 << 4), 0);  // Friday is a weekend day
  EXPECT_NE(sar->row.weekend_mask & (1 << 5), 0);  // Saturday is a weekend day
  EXPECT_EQ(sar->row.weekend_mask & (1 << 6), 0);  // Sunday is a BUSINESS day
  // A Sat/Sun currency for contrast.
  const auto gbp = db::calendar("GBP");
  ASSERT_TRUE(gbp.has_value());
  EXPECT_NE(gbp->row.weekend_mask & (1 << 5), 0);  // Saturday
  EXPECT_NE(gbp->row.weekend_mask & (1 << 6), 0);  // Sunday
  EXPECT_EQ(gbp->row.weekend_mask & (1 << 4), 0);  // Friday is a business day
}
