// E5 taxonomy: T6 regression (fails on the reverted bug)
// DB LAG REPRODUCTIONS (2026-09-14; owner decisions after sourced research, TASKS-ENGINE):
//   MXN-FTIIE-OIS spot lag 1 -> 2  (LatAm SEF certification Jan 2024: default T+2 on Mexico City; DTCC 519 trades T+2)
//   BRL-CDI-SWAP  spot lag 1 -> 0  (B3 Manual de Operacoes SWAP "Registrado em D ... com inicio em D"; B3 formula book:
//                                   the start date's CDI accrues; LatAm SEF 2015 T+0; DTCC 10,727 T+0 vs 26 T+1)
//   CAD-CORRA-OIS payment lag 2 -> 1 (CARR 2021 + Term CORRA basis 2023, TP ICAP CAD OIS MET, LCH CDOR conversion
//                                   consultation 2023 "1 CATO business day")
// Expected dates are derived by hand from each market's holiday calendar (the holiday premises are asserted); none is
// computed with a lag taken from the DB row under test.
#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/build/schedule.hpp"

namespace b = swaps::build;

namespace {

b::Date d(const char* iso) { return b::Date::from_iso(iso); }
b::SwapConv product_of(const char* index) { return b::Index(index).par_convention().resolve(); }

}  // namespace

TEST(LagConventionsRepro, MxnFtiieOisSpotIsTwoMexicoCityBusinessDays) {
  const b::SwapConv sc = product_of("MXN-FTIIE");
  ASSERT_EQ(sc.product_id, "MXN-FTIIE-OIS") << "premise";
  ASSERT_FALSE(b::is_business_day(sc.calendar, d("2025-09-16"))) << "premise: Mexican Independence Day";
  // Traded Friday 2025-09-12: Monday 15 is day 1, Tuesday 16 is a holiday, Wednesday 17 is day 2.
  EXPECT_EQ(b::spot_date(d("2025-09-12"), sc.calendar, sc.spot_lag), d("2025-09-17"));
}

TEST(LagConventionsRepro, BrlCdiSwapStartsOnTheTradeDate) {
  const b::SwapConv sc = product_of("BRL-CDI");
  ASSERT_EQ(sc.product_id, "BRL-CDI-SWAP") << "premise";
  ASSERT_TRUE(sc.zero_coupon) << "premise: the DI x Pre swap";
  ASSERT_TRUE(b::is_business_day(sc.calendar, d("2026-09-14"))) << "premise";
  EXPECT_EQ(b::spot_date(d("2026-09-14"), sc.calendar, sc.spot_lag), d("2026-09-14")) << "B3: registered on D, starts on D";
  // Valued on Brazilian Independence Day (Monday 2026-09-07, closed): a zero lag rolls Following to Tuesday 09-08.
  ASSERT_FALSE(b::is_business_day(sc.calendar, d("2026-09-07"))) << "premise";
  EXPECT_EQ(b::spot_date(d("2026-09-07"), sc.calendar, sc.spot_lag), d("2026-09-08"));
}

TEST(LagConventionsRepro, CadCorraOisPaysOneTorontoBusinessDayAfterThePeriodEnd) {
  const b::SwapConv sc = product_of("CAD-CORRA");
  ASSERT_EQ(sc.product_id, "CAD-CORRA-OIS") << "premise";
  ASSERT_FALSE(b::is_business_day(sc.calendar, d("2026-10-12"))) << "premise: Canadian Thanksgiving";
  // A period ending Friday 2026-10-09: Monday 12 is closed, Tuesday 13 is the first Toronto business day.
  EXPECT_EQ(b::advance_bd(sc.calendar, d("2026-10-09"), sc.pay_lag), d("2026-10-13"));
}
