// E5 taxonomy: T6 regression (fails on the reverted bug)
// SC-CAL1 REPRODUCTION (2026-09-14; owner decision "fix SOFR only, no new fields"): USD-SOFR-OIS counted spot, rolled and
// paid on the USD-SOFR FIXING calendar, which closes EVERY Good Friday. SOFR swaps count U.S. Government Securities
// Business Days (SIFMA) AND New York business days (ISDA's SOFR market practice note, 2022; ISDA's Good Friday 2026
// guidance, 27 March 2026). On a first-Friday Good Friday -- when SIFMA only closes early and no SOFR is published --
// the swap's dates are NOT moved by Good Friday; the fixing calendar moved them one business day late. Public DTCC
// trade reports agree: trades executed 2026-04-01 took effective date 2026-04-03 (1305 trades) not 04-06 (46).
//
// Every expected date below is taken from the ISDA documents or derived by hand from the holiday lists named; none is
// computed with the engine's calendar functions. The controls are the ISDA 2022 note's own examples and a SIFMA-only
// close, which must NOT change.
#include <string>

#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/build/schedule.hpp"

namespace b = swaps::build;

namespace {

b::Date d(const char* iso) { return b::Date::from_iso(iso); }

b::SwapConv sofr_ois() { return b::Index("USD-SOFR").par_convention().resolve(); }

void premises(const b::SwapConv& sc) {
  ASSERT_EQ(sc.product_id, "USD-SOFR-OIS") << "premise";
  ASSERT_EQ(sc.spot_lag, 2) << "premise";
  ASSERT_EQ(sc.pay_lag, 2) << "premise";
  ASSERT_EQ(sc.bdc, "ModifiedFollowing") << "premise";
}

}  // namespace

// ---- fail before the fix ---------------------------------------------------------------------------------------------
TEST(SofrCalendarRepro, AFirstFridayGoodFridayDoesNotMoveTheSpotDate) {
  const b::SwapConv sc = sofr_ois();
  premises(sc);
  ASSERT_EQ(d("2026-04-03").weekday(), 4) << "premise: Good Friday 2026 is a Friday";
  // ISDA Good Friday 2026 guidance section 4: traded Wednesday 1 April 2026 -> effective Friday 3 April 2026.
  EXPECT_EQ(b::spot_date(d("2026-04-01"), sc.calendar, sc.spot_lag), d("2026-04-03"));
  // Traded Thursday 2 April: Fri 03 is day 1, Mon 06 day 2.
  EXPECT_EQ(b::spot_date(d("2026-04-02"), sc.calendar, sc.spot_lag), d("2026-04-06"));
}

TEST(SofrCalendarRepro, PaymentsTwoBusinessDaysAfterAPeriodEndCountGoodFriday2026) {
  const b::SwapConv sc = sofr_ois();
  premises(sc);
  // ISDA Good Friday 2026 guidance footnote 8: a period ending Thursday 2 April pays Monday 6 April; one ending Friday
  // 3 April pays Tuesday 7 April.
  EXPECT_EQ(b::advance_bd(sc.calendar, d("2026-04-02"), sc.pay_lag), d("2026-04-06"));
  EXPECT_EQ(b::advance_bd(sc.calendar, d("2026-04-03"), sc.pay_lag), d("2026-04-07"));
}

TEST(SofrCalendarRepro, AOneYearSwapFromThatSpotEndsOnMondayFifthApril2027) {
  const b::SwapConv sc = sofr_ois();
  premises(sc);
  ASSERT_EQ(d("2027-04-03").weekday(), 5) << "premise: a Saturday";
  // spot 2026-04-03 + 1Y = Saturday 2027-04-03 -> Modified Following -> Monday 2027-04-05 (not a US holiday).
  EXPECT_EQ(b::resolve("1Y", d("2026-04-01"), sc.calendar, sc.bdc, sc.spot_lag), d("2027-04-05"));
}

// ---- controls: pass before and after -------------------------------------------------------------------------------
TEST(SofrCalendarRepro, ControlsIsdaExamplesAndASifmaCloseAreUnchanged) {
  const b::SwapConv sc = sofr_ois();
  premises(sc);
  // ISDA SOFR market practice note (2022) examples, around Christmas 2021 and the full-close Good Friday 2022-04-15.
  EXPECT_EQ(b::spot_date(d("2021-12-22"), sc.calendar, sc.spot_lag), d("2021-12-27"));
  EXPECT_EQ(b::spot_date(d("2021-12-23"), sc.calendar, sc.spot_lag), d("2021-12-28"));
  EXPECT_EQ(b::spot_date(d("2022-04-13"), sc.calendar, sc.spot_lag), d("2022-04-18"));
  EXPECT_EQ(b::spot_date(d("2022-04-14"), sc.calendar, sc.spot_lag), d("2022-04-19"));
  // SIFMA closes Friday 2026-07-03 (Independence Day observed): trades on Wednesday 07-01 start Monday 07-06.
  EXPECT_EQ(b::spot_date(d("2026-07-01"), sc.calendar, sc.spot_lag), d("2026-07-06"));
}
