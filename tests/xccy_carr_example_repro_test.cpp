// E5 taxonomy: T5 value pins (hand-derived dates from a published worked example) | T6 regression where it bites
// XCCY EXCHANGE / RESET / PAYMENT DATES PINNED ON THE CARR WORKED EXAMPLE (draft 2026-09-14, the exchange-lag fields
// stage). Source: Bank of Canada / CARR "CORRA-SOFR cross currency swap" recommended terms (November 2021), Example:
//   t   = Fri 5 March 2021   trade date
//   t+2 = Tue 9 March 2021   trade begins, notionals exchanged
//         Mon 7 June 2021    FX reset for mark-to-market (2 business days before the 9 June roll)
//         9 March .. 9 June  interest calculation period
//         Fri 11 June 2021   interest paid (2 business days after the roll)
//         Sat 9 March 2024   final roll (a Saturday) -> notionals exchanged Mon 11 March 2024,
//         Wed 13 March 2024  last interest paid;  interest calculated for 9 December 2023 .. 9 March 2024.
// CARR explicitly did NOT adopt "delaying the notional re-exchange to match the interest payment": exchanges sit on
// the (adjusted) roll dates, coupons pay 2 business days later (owner decision 2026-09-14, keep the engine's placement).
//
// The example is CAD/USD; none of its dates touches a TARGET or SIFMA/New York holiday, so the same dates must hold
// on the XCCY-MTM-EURUSD product's joint calendar (asserted below as premises). Every expected date is the example's.
//
// OWNER DECISION 2026-09-14 (revised): the DB CARRIES fx_reset_fixing_lag = 2 and fx_reset_calendar and the pins below
// check the fixing date, but the resetting notional's FX forward stays at the period START -- a rate fixed 2 business
// days earlier is a spot rate for value on that start. (A first draft moved reset_time to the fixing date: 2 days of
// carry wrong, withdrawn before commit.) Every date / exchange pin already held on f5f20d1.
#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/problem.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;

namespace {

b::Date d(const char* iso) { return b::Date::from_iso(iso); }

struct CarrCase {
  b::XccyConv x = b::xccy_conv("EURUSD");
  b::Date trade = d("2021-03-05");
  b::Date maturity;
  cal::Instrument ins;
  CarrCase() {
    maturity = b::resolve("3Y", trade, x.calendar, x.bdc, x.spot_lag);
    ins = b::xccy_mtm_basis(trade, x, maturity, /*ci=*/0, /*foreign=*/1, /*fund=*/2, /*fx_spot=*/1.25, 0.0);
  }
  double t(const char* iso) const { return b::curve_time(trade, d(iso)); }
};

void assert_premises(const CarrCase& c) {
  ASSERT_EQ(c.x.spot_lag, 2) << "premise: spot (2 business days) start";
  ASSERT_EQ(c.x.pay_lag, 2) << "premise: interest paid 2 business days after the roll";
  ASSERT_EQ(c.x.freq_tok, "3M") << "premise: quarterly";
  for (const char* iso : {"2021-03-08", "2021-03-09", "2021-06-07", "2021-06-08", "2021-06-09", "2021-06-10",
                          "2021-06-11", "2023-12-11", "2024-03-11", "2024-03-12", "2024-03-13"})
    ASSERT_TRUE(b::is_business_day(c.x.calendar, d(iso))) << "premise: " << iso << " is open on " << c.x.calendar;
  ASSERT_FALSE(b::is_business_day(c.x.calendar, d("2024-03-09"))) << "premise: 9 March 2024 is a Saturday";
}

}  // namespace

TEST(XccyCarrExample, NotionalsExchangeOnTheRollDatesAndInterestPaysTwoBusinessDaysLater) {
  const CarrCase c;
  assert_premises(c);
  EXPECT_EQ(b::spot_date(c.trade, c.x.calendar, c.x.spot_lag), d("2021-03-09")) << "trade begins t+2";
  EXPECT_EQ(c.maturity, d("2024-03-11")) << "9 March 2024 is a Saturday: notionals exchanged Monday 11 March";

  // The resetting (funding) leg carries the MtM exchanges on its accrual dates and pays interest 2 BD later.
  const auto& m = c.ins.mtm.coupons;
  ASSERT_EQ(m.size(), 12u) << "3 years of quarterly periods";
  ASSERT_TRUE(m.front().accrual_set && m.back().accrual_set);
  EXPECT_DOUBLE_EQ(m.front().accrual_start, c.t("2021-03-09")) << "notionals exchanged 9 March 2021";
  EXPECT_DOUBLE_EQ(m.front().accrual_end, c.t("2021-06-09")) << "first calculation period ends 9 June 2021";
  EXPECT_DOUBLE_EQ(m.front().pay, c.t("2021-06-11")) << "first interest paid 11 June 2021";
  EXPECT_DOUBLE_EQ(m[1].accrual_start, c.t("2021-06-09")) << "the second period starts on the roll, not the pay date";
  EXPECT_DOUBLE_EQ(m.back().accrual_start, c.t("2023-12-11")) << "9 December 2023 is a Saturday -> Monday 11";
  EXPECT_DOUBLE_EQ(m.back().accrual_end, c.t("2024-03-11")) << "final notionals exchanged Monday 11 March 2024";
  EXPECT_DOUBLE_EQ(m.back().pay, c.t("2024-03-13")) << "last interest paid Wednesday 13 March 2024";

  // The constant-notional leg's exchange pair sits on the same dates (XB1: its self coupons pay on the accrual end).
  const auto& f = c.ins.fwd.coupons;
  ASSERT_EQ(f.size(), m.size());
  EXPECT_DOUBLE_EQ(f.front().accrual_start, c.t("2021-03-09"));
  EXPECT_DOUBLE_EQ(f.back().accrual_end, c.t("2024-03-11"));
  EXPECT_DOUBLE_EQ(f.back().pay, c.t("2024-03-11")) << "an exchange is not delayed to the interest payment date";
}

// The FX fixing: the DB's fx_reset_fixing_lag on fx_reset_calendar puts it 2 business days before the period start
// (CARR: Monday 7 June for the period from Wednesday 9 June), and that fixing is a spot rate for value ON the period
// start -- so the resetting notional's forward stays at the period start (reset_time unset = the accrual start).
TEST(XccyCarrExample, TheFxFixesTwoBusinessDaysBeforeEachPeriodForValueOnThePeriodStart) {
  const CarrCase c;
  assert_premises(c);
  ASSERT_EQ(c.x.fx_reset_lag, 2) << "premise: the DB fixing lag";
  ASSERT_EQ(c.x.fx_reset_calendar, c.x.calendar) << "premise: the pair's joint calendar";
  EXPECT_EQ(b::advance_bd(c.x.fx_reset_calendar, d("2021-06-09"), -c.x.fx_reset_lag), d("2021-06-07"))
      << "FX reset for mark-to-market: Monday 7 June 2021";
  EXPECT_EQ(b::spot_date(d("2021-06-07"), c.x.calendar, c.x.spot_lag), d("2021-06-09"))
      << "a spot rate fixed on 7 June is for value on the 9 June period start";
  const auto& m = c.ins.mtm.coupons;
  ASSERT_GE(m.size(), 2u);
  EXPECT_DOUBLE_EQ(m[1].accrual_start, c.t("2021-06-09"));
  for (const auto& coupon : m) EXPECT_LT(coupon.reset_time, 0.0) << "the notional's forward is read at the accrual start";
}

TEST(XccyCarrExample, ALaggedNotionalExchangeIsRefusedNotIgnored) {
  for (int which = 0; which < 3; ++which) {
    b::XccyConv x = b::xccy_conv("EURUSD");
    (which == 0 ? x.exchange_lag_initial : which == 1 ? x.exchange_lag_intermediate : x.exchange_lag_final) = 2;
    const b::Date vd = d("2021-03-05");
    EXPECT_THROW(b::xccy_mtm_basis(vd, x, b::resolve("1Y", vd, x.calendar, x.bdc, x.spot_lag), 0, 1, 2, 1.25, 0.0),
                 std::invalid_argument) << "exchange lag " << which;
  }
}
