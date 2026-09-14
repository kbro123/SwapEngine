// E5 taxonomy: T6 regression (fails on the reverted bug)
// P12 REPRODUCTION (2026-09-14; owner decisions "swap_spread matched_maturity = IMPLEMENT" and "the start date of the
// matched maturity needs to match spot for that currency's standard swaps"): spread_type "matched_maturity" is
// ACCEPTED (api/codec.cpp spread_type_from_str) but derive::swap_spread builds the swap of the request's TENOR whatever
// the type says (include/swaps/derive/asset_swap.hpp:286-290) -- the verb returns the HEADLINE swap under the matched
// label (TASKS-ENGINE E7 3.4 finding 1).
//
// The reference is the DEFINITION, written out by hand per currency and checked against the conventions DB rows
// (asserted as premises, nothing restated as a literal convention): the matched swap is the index's DB par product,
// starting at THAT PRODUCT's spot (its spot_lag business days on ITS calendar), TERMINATING on the bond's maturity
// rolled by the product bdc, rolled BACKWARD from the unadjusted maturity (a short FRONT stub, roll day = the maturity's
// day of month). Every accrual is the DB leg day count on the listed dates; every payment is the accrual end + the DB
// payment lag on the product calendar. QuantLib's own swaps are the reference in
// tests/matched_swap_multiccy_oracle_test.cpp; this header-only file is what tools/mutate.py compiles.
//
// Each currency's value date is chosen so its spot CROSSES A HOLIDAY OF THAT CURRENCY ONLY, so a USD T+2 or a US
// calendar gives a different spot (premise-checked below):
//   USD-SOFR    T+2  Fri 2026-09-04 over Labor Day (London open)             -> Wed 2026-09-09
//   EUR-ESTR    T+2  Wed 2027-03-24 over Good Friday + Easter Monday (US open Monday) -> Tue 2027-03-30
//   GBP-SONIA   T+0  Mon 2026-10-12, Columbus Day (SOFR closed, London open) -> Mon 2026-10-12
//   AUD-AONIA   T+1  Fri 2026-10-02 over NSW Labour Day (US open)             -> Tue 2026-10-06
//   AUD-BBSW-6M T+1  same, a SEMIANNUAL IBOR product (both legs 6M): a 20-day front stub
// and each maturity lands on a holiday of that currency only (termination roll on the product calendar).
//
// Run against the unfixed library, the per-currency schedule test, the USD anchor, the weekend-maturity and the month-end
// tests FAILED (the tenor swap was built whatever spread_type said); spot and the holiday-value-date guard passed and are
// kept as pins. The fix is derive::spread_swap (asset_swap.hpp). The only bond rows are US-TREASURY, so the non-USD
// matched swaps are driven through spread_swap -- the bond-free seam -- with a US-TREASURY carrier contributing only its
// maturity. A bond and a swap index of different currencies are NOT refused (owner decision 2026-09-14).
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/derive/asset_swap.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace der = swaps::derive;

namespace {

b::Date d(const char* iso) { return b::Date::from_iso(iso); }

// The par product a swap index names (its conventions DB row, resolved).
b::SwapConv product_of(const char* index) { return b::Index(index).par_convention().resolve(); }

// The CONTROL rows a wrong spot is taken from: another currency's standard swap. Index ids name the rows; every lag and
// calendar is read from them.
const char* const kUsdIndex = "USD-SOFR";
const char* const kGbpIndex = "GBP-SONIA";

// A spot lag that is NOT this product's: USD's (the owner's "USD T+2 applied to another currency") unless it equals
// this product's, then GBP's.
int other_spot_lag(const b::SwapConv& sc) {
  const b::SwapConv usd = product_of(kUsdIndex);
  return usd.spot_lag != sc.spot_lag ? usd.spot_lag : product_of(kGbpIndex).spot_lag;
}
// A spot calendar that is NOT this product's: the USD swap calendar unless it is this product's, then GBP's.
std::string other_calendar(const b::SwapConv& sc) {
  const b::SwapConv usd = product_of(kUsdIndex);
  return usd.calendar != sc.calendar ? usd.calendar : product_of(kGbpIndex).calendar;
}

struct HandPeriod {
  const char* start;
  const char* end;
  const char* pay;
};

struct Case {
  const char* name;
  const char* index;       // the swap index -- its DB par product is the matched swap
  const char* product;     // premise: the par product id
  const char* value_date;
  const char* spot;        // hand: the product's spot
  const char* holiday;     // hand: a date the spot crosses (or, for T+0, the value date) that is closed on ONE of
                           // {this product's calendar, the other calendar} and open on the other
  const char* carrier_issue;  // TODAY ONLY: dated date of the US-TREASURY carrier (maturity - N years)
  const char* maturity;       // the bond's UNADJUSTED maturity
  const char* termination;    // hand: the maturity rolled by the product bdc on the product calendar
  bool maturity_closed;       // hand: the maturity is a holiday of THIS currency only (termination rolls on its calendar)
  const char* today_tenor;    // TODAY ONLY: the tenor the unfixed library requires
  std::vector<HandPeriod> fixed, flt;
};

// ---- hand schedules (dates derived by hand from the DB holiday rules; every one is premise-checked) ----------------
// USD-SOFR-OIS (annual/annual, pay lag 2): spot Wed 2026-09-09; Aug-15 roll; 2027-08-15 is a Sunday.
const std::vector<HandPeriod> kUsd = {
    {"2026-09-09", "2027-08-16", "2027-08-18"},  // 341 days: the short FRONT stub
    {"2027-08-16", "2028-08-15", "2028-08-17"},  // 365 days (rolled from the 15th, not from the 16th)
    {"2028-08-15", "2029-08-15", "2029-08-17"},
    {"2029-08-15", "2030-08-15", "2030-08-19"},
    {"2030-08-15", "2031-08-15", "2031-08-19"},
};
// EUR-ESTR-OIS (annual/annual, pay lag 2, TARGET): spot Tue 2027-03-30; May-1 roll (TARGET Labour Day every year).
const std::vector<HandPeriod> kEur = {
    {"2027-03-30", "2027-05-03", "2027-05-05"},  // 34 days: 2027-05-01 is a Saturday -> Mon 05-03
    {"2027-05-03", "2028-05-02", "2028-05-04"},  // Mon 2028-05-01 Labour Day -> Tue 05-02
    {"2028-05-02", "2029-05-02", "2029-05-04"},
    {"2029-05-02", "2030-05-02", "2030-05-06"},
    {"2030-05-02", "2031-05-02", "2031-05-06"},  // termination: Thu 2031-05-01 Labour Day -> Fri 05-02
};
// GBP-SONIA-OIS (annual/annual, pay lag 0, UK settlement): spot = value date Mon 2026-10-12; Dec-26 roll through the
// Christmas / Boxing Day substitutes (2026: Boxing Sat -> Mon 28; 2027: Christmas Sat -> Mon 27, Boxing Sun -> Tue 28).
const std::vector<HandPeriod> kGbp = {
    {"2026-10-12", "2026-12-29", "2026-12-29"},  // 78 days: Sat 12-26 -> Mon 12-28 (substitute) -> Tue 12-29
    {"2026-12-29", "2027-12-29", "2027-12-29"},  // Sun 12-26 -> Mon 27, Tue 28 substitutes -> Wed 12-29
    {"2027-12-29", "2028-12-27", "2028-12-27"},  // 364 days: Tue 2028-12-26 Boxing Day -> Wed 12-27
    {"2028-12-27", "2029-12-27", "2029-12-27"},
    {"2029-12-27", "2030-12-27", "2030-12-27"},  // termination: Thu 2030-12-26 Boxing Day -> Fri 12-27
};
// AUD-AONIA-OIS (annual/annual, pay lag 2, NSW): spot Tue 2026-10-06; Jan-26 roll (Australia Day, weekend -> Monday).
const std::vector<HandPeriod> kAud = {
    {"2026-10-06", "2027-01-27", "2027-01-29"},  // 113 days: Tue 2027-01-26 Australia Day -> Wed 01-27
    {"2027-01-27", "2028-01-27", "2028-01-31"},
    {"2028-01-27", "2029-01-29", "2029-01-31"},  // 368 days: Fri 2029-01-26 -> Mon 01-29
    {"2029-01-29", "2030-01-29", "2030-01-31"},  // Sat 2030-01-26 -> substitute Mon 28 -> Tue 01-29
    {"2030-01-29", "2031-01-28", "2031-01-30"},  // Sun 2031-01-26 -> substitute Mon 27 -> Tue 01-28
    {"2031-01-28", "2032-01-27", "2032-01-29"},  // termination: Mon 2032-01-26 -> Tue 01-27
};
// AUD-BBSW-6M-IRS (6M/6M, pay lag 0): spot Tue 2026-10-06; Apr/Oct-25 roll (ANZAC Day: no substitute in NSW).
const std::vector<HandPeriod> kAud6m = {
    {"2026-10-06", "2026-10-26", "2026-10-26"},  // 20 days: Sun 2026-10-25 -> Mon 10-26
    {"2026-10-26", "2027-04-26", "2027-04-26"},  // Sun 2027-04-25 -> Mon 04-26
    {"2027-04-26", "2027-10-25", "2027-10-25"},
    {"2027-10-25", "2028-04-26", "2028-04-26"},  // Tue 2028-04-25 ANZAC Day -> Wed 04-26
    {"2028-04-26", "2028-10-25", "2028-10-25"},
    {"2028-10-25", "2029-04-26", "2029-04-26"},  // termination: Wed 2029-04-25 ANZAC Day -> Thu 04-26
};

std::vector<Case> cases() {
  return {
      // USD keeps the original fixture's Friday maturity (no termination roll; the Sunday case is its own TEST).
      {"USD", "USD-SOFR", "USD-SOFR-OIS", "2026-09-04", "2026-09-09", "2026-09-07", "2021-08-15", "2031-08-15",
       "2031-08-15", false, "5Y", kUsd, kUsd},
      {"EUR", "EUR-ESTR", "EUR-ESTR-OIS", "2027-03-24", "2027-03-30", "2027-03-29", "2021-05-01", "2031-05-01",
       "2031-05-02", true, "4Y", kEur, kEur},
      {"GBP", "GBP-SONIA", "GBP-SONIA-OIS", "2026-10-12", "2026-10-12", "2026-10-12", "2020-12-26", "2030-12-26",
       "2030-12-27", true, "4Y", kGbp, kGbp},
      {"AUD", "AUD-AONIA", "AUD-AONIA-OIS", "2026-10-02", "2026-10-06", "2026-10-05", "2022-01-26", "2032-01-26",
       "2032-01-27", true, "5Y", kAud, kAud},
      {"AUD-6M", "AUD-BBSW-6M", "AUD-BBSW-6M-IRS", "2026-10-02", "2026-10-06", "2026-10-05", "2019-04-25", "2029-04-25",
       "2029-04-26", true, "3Y", kAud6m, kAud6m},
  };
}

// ---- the request (today's API) ---------------------------------------------------------------------------------
const char* const kCarrierConvention = "US-TREASURY";  // TODAY ONLY: the one bonds[] row family the DB has

der::SwapSpreadRequest request(const Case& c) {
  der::SwapSpreadRequest r;
  r.value_date = d(c.value_date);
  r.bond = b::BondId{"CARRIER", kCarrierConvention, d(c.carrier_issue), d(c.maturity), 0.04, std::nullopt};
  r.clean = 1.0;
  r.spread = -0.0030;
  r.index = c.index;
  r.swap_curve = 0;
  r.factor_curve = 1;
  r.type = der::SwapSpreadType::MatchedMaturity;
  return r;
}

// The matched swap the library builds for this case.
cal::Instrument matched_swap(const Case& c) {
  // The bond-free, currency-generic seam: the carrier bond contributes only its maturity.
  return der::spread_swap(request(c), product_of(c.index)).swap;
}

// ---- premises: every hand date against the DB row and the DB calendars --------------------------------------------
void assert_premises(const Case& c) {
  SCOPED_TRACE(c.name);
  const b::SwapConv sc = product_of(c.index);
  ASSERT_EQ(sc.product_id, c.product) << "premise: the index's par product";
  ASSERT_FALSE(sc.zero_coupon) << "premise: a coupon-bearing product";
  const b::Date vd = d(c.value_date), spot = d(c.spot);
  ASSERT_TRUE(b::is_business_day(sc.calendar, vd)) << "premise: the value date is a business day of the product";
  ASSERT_EQ(b::spot_date(vd, sc.calendar, sc.spot_lag), spot) << "premise: the product's spot";

  // The case DISCRIMINATES: another currency's spot lag, or the product's lag on another currency's calendar, gives a
  // different start -- and the named holiday is closed on exactly one of the two calendars.
  const std::string oc = other_calendar(sc);
  ASSERT_NE(b::spot_date(vd, sc.calendar, other_spot_lag(sc)), spot) << "premise: another spot lag gives another spot";
  ASSERT_NE(b::spot_date(vd, oc, sc.spot_lag), spot) << "premise: another calendar gives another spot";
  ASSERT_NE(b::is_business_day(sc.calendar, d(c.holiday)), b::is_business_day(oc, d(c.holiday)))
      << "premise: " << c.holiday << " is a holiday of ONE calendar only";

  // The maturity rolls to the listed termination; where flagged, it is a holiday of this currency only.
  const b::Date mat = d(c.maturity), term = d(c.termination);
  if (c.maturity_closed) {
    ASSERT_FALSE(b::is_business_day(sc.calendar, mat)) << "premise: the maturity is closed on the product calendar";
    ASSERT_TRUE(b::is_business_day(oc, mat)) << "premise: ... and open on the other calendar";
  } else {
    ASSERT_TRUE(b::is_business_day(sc.calendar, mat)) << "premise: a business-day maturity";
  }
  ASSERT_EQ(b::adjust(sc.calendar, mat, sc.bdc), term) << "premise: the termination roll";

  for (const auto* leg : {&c.fixed, &c.flt}) {
    ASSERT_FALSE(leg->empty());
    ASSERT_EQ(d(leg->front().start), spot) << "premise: the first accrual starts at spot";
    ASSERT_EQ(d(leg->back().end), term) << "premise: the last accrual ends at the termination";
    for (std::size_t k = 0; k < leg->size(); ++k) {
      const HandPeriod& p = (*leg)[k];
      if (k) ASSERT_EQ(d(p.start), d((*leg)[k - 1].end)) << "premise: contiguous";
      ASSERT_TRUE(b::is_business_day(sc.calendar, d(p.end))) << "premise: " << p.end << " is a business day";
      ASSERT_EQ(b::advance_bd(sc.calendar, d(p.end), sc.pay_lag), d(p.pay)) << "premise: payment of " << p.end;
    }
  }
}

// Both legs of the emitted swap, coupon by coupon, against the hand schedule, on the DB day counts.
void expect_schedule(const Case& c, const cal::Instrument& swap) {
  SCOPED_TRACE(c.name);
  const b::SwapConv sc = product_of(c.index);
  const b::Date vd = d(c.value_date);
  ASSERT_EQ(swap.fixed.coupons.size(), c.fixed.size()) << "fixed coupon count";
  for (std::size_t k = 0; k < c.fixed.size(); ++k) {
    const HandPeriod& p = c.fixed[k];
    EXPECT_DOUBLE_EQ(swap.fixed.coupons[k].pay, b::curve_time(vd, d(p.pay))) << "fixed pay " << k;
    EXPECT_DOUBLE_EQ(swap.fixed.coupons[k].tau, b::year_frac(sc.fixed_dc, d(p.start), d(p.end), sc.calendar))
        << "fixed accrual " << k << " (" << p.start << " -> " << p.end << ")";
  }
  ASSERT_EQ(swap.fwd.coupons.size(), c.flt.size()) << "float coupon count";
  for (std::size_t k = 0; k < c.flt.size(); ++k) {
    const HandPeriod& p = c.flt[k];
    EXPECT_DOUBLE_EQ(swap.fwd.coupons[k].pay, b::curve_time(vd, d(p.pay))) << "float pay " << k;
    EXPECT_DOUBLE_EQ(swap.fwd.coupons[k].tau_pay, b::year_frac(sc.float_dc, d(p.start), d(p.end), sc.calendar))
        << "float accrual " << k;
    EXPECT_DOUBLE_EQ(swap.fwd.coupons[k].accrual_start, b::curve_time(vd, d(p.start))) << "float accrual start " << k;
    EXPECT_DOUBLE_EQ(swap.fwd.coupons[k].accrual_end, b::curve_time(vd, d(p.end))) << "float accrual end " << k;
  }
}

}  // namespace

// ================================================================================================================
// THE BUG, per currency
// ================================================================================================================
TEST(SwapSpreadMatchedMaturityRepro, EachCurrencysMatchedSwapRunsFromItsSpotToTheBondsMaturity) {
  for (const Case& c : cases()) {
    assert_premises(c);
    // TODAY: the tenor swap -- USD 5 coupons ending 2031-09-09 (first 365d); EUR 4 ending 2031-03-31 (first 366d); GBP
    // 4 ending 2030-10-14; AUD 5 ending 2031-10-07; AUD-6M 6 ending 2029-10-08 (first 182d, not 20d).
    expect_schedule(c, matched_swap(c));
  }
}

// ================================================================================================================
// Spot is the CURRENCY's (passes today on the tenor path; pins the owner's clarification for the matched path)
// ================================================================================================================
TEST(SwapSpreadMatchedMaturityRepro, SpotIsTheCurrencysOwnSpot) {
  for (const Case& c : cases()) {
    SCOPED_TRACE(c.name);
    const b::SwapConv sc = product_of(c.index);
    const b::Date vd = d(c.value_date);
    const cal::Instrument swap = matched_swap(c);
    ASSERT_FALSE(swap.fwd.coupons.empty());
    const double start = swap.fwd.coupons.front().accrual_start;
    EXPECT_DOUBLE_EQ(start, b::curve_time(vd, b::spot_date(vd, sc.calendar, sc.spot_lag))) << "the product's spot";
    EXPECT_NE(start, b::curve_time(vd, b::spot_date(vd, sc.calendar, other_spot_lag(sc))))
        << "another currency's spot LAG (for GBP / AUD: USD's T+2) must not be the start";
    EXPECT_NE(start, b::curve_time(vd, b::spot_date(vd, other_calendar(sc), sc.spot_lag)))
        << "the product's lag on another currency's CALENDAR must not be the start";
  }
}

// A T+0 product valued on its own holiday starts on the next business day (build::advance_bd n == 0 rolls Following,
// calendar.hpp; QuantLib MakeOIS adjusts the evaluation date the same way). Passes today; a guard.
TEST(SwapSpreadMatchedMaturityRepro, AZeroLagSpotOnAHolidayRollsToTheNextBusinessDay) {
  const b::SwapConv sc = product_of(kGbpIndex);
  ASSERT_EQ(sc.spot_lag, 0) << "premise: a T+0 product";
  const b::Date vd = d("2026-08-31");  // Summer Bank Holiday (last Monday of August): London closed, New York open
  ASSERT_FALSE(b::is_business_day(sc.calendar, vd)) << "premise";
  ASSERT_TRUE(b::is_business_day(other_calendar(sc), vd)) << "premise";
  ASSERT_EQ(b::spot_date(vd, sc.calendar, sc.spot_lag), d("2026-09-01")) << "premise";
  Case c{"GBP-holiday", kGbpIndex, "GBP-SONIA-OIS", "2026-08-31", "2026-09-01", "2026-08-31", "2020-12-26",
         "2030-12-26", "2030-12-27", true, "4Y", {}, {}};
  const cal::Instrument swap = matched_swap(c);
  ASSERT_FALSE(swap.fwd.coupons.empty());
  EXPECT_DOUBLE_EQ(swap.fwd.coupons.front().accrual_start, b::curve_time(vd, d("2026-09-01")));
}

// ================================================================================================================
// USD through the verb's library entry: the anchor and the headline difference (unchanged by the fix)
// ================================================================================================================
TEST(SwapSpreadMatchedMaturityRepro, TheMatchedAnchorIsTheSwapsTerminationNotTheTenorDate) {
  const Case c = cases().front();
  ASSERT_STREQ(c.name, "USD");
  assert_premises(c);
  const der::SwapSpreadResult res = der::swap_spread(request(c));
  der::SwapSpreadRequest head = request(c);
  head.type = der::SwapSpreadType::HeadlineYield;
  head.tenor = "5Y";
  const der::SwapSpreadResult hr = der::swap_spread(head);

  const b::Date vd = d(c.value_date);
  // TODAY: anchor = curve_time(2031-09-09), the 5Y tenor swap's maturity.
  EXPECT_EQ(res.anchor, b::curve_time(vd, d(c.termination))) << "the anchor is the matched swap's termination";
  EXPECT_EQ(res.swap_maturity, d(c.termination));
  EXPECT_NE(res.anchor, hr.anchor) << "matched_maturity returned the headline anchor";
  // Only the swap differs: one benchmark yield, one spread.
  EXPECT_EQ(res.derived.bond_yield, hr.derived.bond_yield);
  EXPECT_EQ(res.derived.spread, hr.derived.spread);
  EXPECT_EQ(res.derived.rows.pin.market, hr.derived.rows.pin.market);
  EXPECT_EQ(res.derived.rows.asw.market, hr.derived.rows.asw.market);
}

TEST(SwapSpreadMatchedMaturityRepro, AWeekendMaturityRollsItsTerminationButNotItsRollDay) {
  Case c = cases().front();
  c.name = "USD-Sunday";
  c.carrier_issue = "2022-08-15";
  c.maturity = "2032-08-15";  // a Sunday
  c.termination = "2032-08-16";
  c.today_tenor = "6Y";
  c.fixed.push_back({"2031-08-15", "2032-08-16", "2032-08-18"});  // 367 days: the Sunday termination rolled to Monday
  c.flt = c.fixed;
  const b::SwapConv sc = product_of(c.index);
  ASSERT_EQ(d(c.maturity).weekday(), 6) << "premise: a Sunday";
  ASSERT_EQ(b::adjust(sc.calendar, d(c.maturity), sc.bdc), d(c.termination)) << "premise";
  // (assert_premises' "holiday of this currency only" check does not apply to a weekend; the schedule premises do.)
  for (const HandPeriod& p : c.fixed)
    ASSERT_EQ(b::advance_bd(sc.calendar, d(p.end), sc.pay_lag), d(p.pay)) << "premise: payment of " << p.end;
  const der::SwapSpreadResult res = der::swap_spread(request(c));
  EXPECT_EQ(res.anchor, b::curve_time(d(c.value_date), d(c.termination)));
  EXPECT_EQ(res.swap_maturity, d(c.termination));
  // Coupon 1 must be 365 days: anchoring the roll on the ADJUSTED 16th would make it 366 (2028 is a leap year).
  expect_schedule(c, res.derived.rows.asw.combination.at(0).instrument);
}

// ================================================================================================================
// FS1 -- a spot-month roll date that adjusts ONTO spot (found by this design; schedule.hpp front-stub path)
// ================================================================================================================
// A 2-year note maturing on the 31st, valued so that spot is the last business day before a weekend month-end: the
// backward roll's first interior date (Sat 2026-10-31) is after spot but ModifiedFollowing rolls it BACK onto spot
// (Fri 2026-10-30). QuantLib's Schedule drops a boundary that is <= the start (schedule.cpp "final safety checks");
// the engine's front-stub loop (schedule.hpp:192-196) keeps it, emitting a ZERO-LENGTH first period (tau 0), which
// the bundle validator refuses (calibration/problem.hpp:271) and the kernel asserts against (pricing/cashflows.hpp:242).
// Fails today (tenor swap: its last accrual ends Mon 2028-10-30, 367 days) AND on a matched fix without the guard
// (three coupons, the first of zero length).
TEST(SwapSpreadMatchedMaturityRepro, ARollDateThatAdjustsOntoSpotIsNotAZeroLengthPeriod) {
  Case c = cases().front();
  c.name = "USD-month-end";
  c.value_date = "2026-10-28";
  c.spot = "2026-10-30";
  c.carrier_issue = "2026-10-31";
  c.maturity = "2028-10-31";
  c.termination = "2028-10-31";
  c.today_tenor = "2Y";
  c.fixed = {
      {"2026-10-30", "2027-10-29", "2027-11-02"},  // 364 days: Sun 2027-10-31 -> MF back to Fri 10-29
      {"2027-10-29", "2028-10-31", "2028-11-02"},  // 368 days
  };
  c.flt = c.fixed;
  const b::SwapConv sc = product_of(c.index);
  const b::Date vd = d(c.value_date);
  ASSERT_EQ(b::spot_date(vd, sc.calendar, sc.spot_lag), d(c.spot)) << "premise";
  ASSERT_EQ(d("2026-10-31").weekday(), 5) << "premise: a Saturday";
  ASSERT_EQ(b::adjust(sc.calendar, d("2026-10-31"), sc.bdc), d(c.spot)) << "premise: the roll date adjusts ONTO spot";
  ASSERT_EQ(b::adjust(sc.calendar, d("2027-10-31"), sc.bdc), d("2027-10-29")) << "premise";
  ASSERT_TRUE(b::is_business_day(sc.calendar, d(c.maturity))) << "premise";
  for (const HandPeriod& p : c.fixed)
    ASSERT_EQ(b::advance_bd(sc.calendar, d(p.end), sc.pay_lag), d(p.pay)) << "premise: payment of " << p.end;
  const cal::Instrument swap = matched_swap(c);
  for (const auto& cp : swap.fwd.coupons) EXPECT_GT(cp.tau_pay, 0.0) << "no zero-length float period";
  for (const auto& cp : swap.fixed.coupons) EXPECT_GT(cp.tau, 0.0) << "no zero-length fixed period";
  expect_schedule(c, swap);
}

TEST(SwapSpreadMatchedMaturityRepro, AnExplicitAnchorIsCarried) {
  // Control: passes before and after the fix.
  der::SwapSpreadRequest r = request(cases().front());
  r.anchor = 4.2;
  EXPECT_EQ(der::swap_spread(r).anchor, 4.2);
}

// The refusals name their reason (owner decisions 2026-09-14: a matched request carrying a tenor is refused; a bond and
// an index of different currencies are NOT).
TEST(SwapSpreadMatchedMaturityRepro, TheRefusalsNameTheirReason) {
  const std::vector<Case> cs = cases();
  const auto what = [](const der::SwapSpreadRequest& r) -> std::string {
    try {
      (void)der::swap_spread(r);
    } catch (const std::invalid_argument& e) {
      return e.what();
    }
    return "";
  };
  const auto seam_what = [](const der::SwapSpreadRequest& r, const char* index) -> std::string {
    try {
      (void)der::spread_swap(r, product_of(index));
    } catch (const std::invalid_argument& e) {
      return e.what();
    }
    return "";
  };
  // (1) a tenor names a different swap
  der::SwapSpreadRequest with_tenor = request(cs[0]);
  with_tenor.tenor = "5Y";
  EXPECT_NE(what(with_tenor).find("tenor"), std::string::npos) << what(with_tenor);
  // (2) the bond matures on or before THE PRODUCT's spot -- and the boundary is per currency: a GBP maturity on the
  //     business day after the value date is a one-day matched swap under T+0.
  der::SwapSpreadRequest old = request(cs[0]);
  old.bond.maturity = d("2026-09-08");  // after the value date, before USD spot 2026-09-09
  EXPECT_NE(seam_what(old, cs[0].index).find("matures"), std::string::npos) << seam_what(old, cs[0].index);
  der::SwapSpreadRequest next_day = request(cs[2]);
  next_day.bond.maturity = d("2026-10-13");
  EXPECT_EQ(seam_what(next_day, cs[2].index), "") << "T+0: a bond maturing tomorrow still has a matched swap";
  // (3) a day-frequency product is refused by name
  der::SwapSpreadRequest mxn = request(cs[0]);
  mxn.index = "MXN-TIIE-28";
  EXPECT_NE(seam_what(mxn, "MXN-TIIE-28").find(product_of("MXN-TIIE-28").product_id), std::string::npos)
      << seam_what(mxn, "MXN-TIIE-28");
  // (4) a bond and an index of different currencies are accepted (owner decision): a US-TREASURY carrier with EUR-ESTR
  der::SwapSpreadRequest cross = request(cs[1]);
  EXPECT_EQ(what(cross), "") << "a bond against another currency's swap is not refused";
  // (5) control: headline still needs its tenor
  der::SwapSpreadRequest head = request(cs[0]);
  head.type = der::SwapSpreadType::HeadlineYield;
  EXPECT_THROW((void)der::swap_spread(head), std::invalid_argument);
}
