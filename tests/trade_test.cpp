// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// swaps::trade::Trade — the BOOKED DEAL object. A trade is the distinct thing a desk books (an id, a
// direction, a CONTRACT rate struck at execution) as opposed to a market quote (calibration::Instrument,
// which has no contract rate). This test books a vanilla swap and checks it materializes into the fast
// valuation Position the reprice kernel consumes, with the documented payer sign convention.
#include <gtest/gtest.h>

#include "swaps/trade/trade.hpp"

namespace tr = swaps::trade;
namespace b = swaps::build;
namespace pf = swaps::portfolio;

TEST(Trade, VanillaPayerSwapToPosition) {
  const b::Date value_date = b::Date::from_iso("2026-01-15");
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");

  // ~spot effective, 10y maturity (the builders roll their own schedule from spot(value_date); effective
  // is the booked record and, for a spot-start swap, equals that spot).
  const b::Date effective = b::spot_date(value_date, conv.calendar, conv.spot_lag);
  const b::Date maturity = b::resolve("10Y", value_date, "NONE", "Following", 0);

  const tr::Trade t = tr::Trade::vanilla_swap(
      /*id=*/"USD-SOFR-10Y-001", /*notional=*/100e6, /*pay=*/tr::Pay::Fixed,
      /*fixed_rate=*/0.0375, /*currency=*/"USD", /*index=*/"USD-SOFR", effective, maturity,
      /*forecast_role=*/0, /*discount_role=*/0);

  const pf::MultiCurveBook::Position pos = t.to_position(value_date, conv);

  // Booked attributes carried straight onto the Position.
  EXPECT_EQ(pos.kind, pf::MultiCurveBook::Kind::Swap);
  EXPECT_DOUBLE_EQ(pos.notional, 100e6);        // payer of fixed => +notional (documented sign convention)
  EXPECT_DOUBLE_EQ(pos.fixed_rate, 0.0375);     // the CONTRACT rate, a booked attribute a quote lacks
  EXPECT_EQ(pos.fwd_curve, 0);
  EXPECT_EQ(pos.disc_curve, 0);
  EXPECT_EQ(pos.fixed_curve, 0);

  // A real Position with real coupon schedules on both legs.
  EXPECT_FALSE(pos.float_coupons.empty());
  EXPECT_FALSE(pos.fixed_coupons.empty());

  // Last pay: maturity 2036-01-15 (a Tuesday, unadjusted) + the SOFR 2-business-day payment lag =
  // 2036-01-17, i.e. 3654 days from the value date = 10.010958904109590 ACT/365F (hand computation,
  // 2026-09-10). E5: the old ±0.1y bound could not see a dropped pay lag or an unadjusted maturity.
  EXPECT_NEAR(pos.float_coupons.back().pay, 3654.0 / 365.0, 1e-12);
  EXPECT_NEAR(pos.fixed_coupons.back().pay, 3654.0 / 365.0, 1e-12);
}

TEST(Trade, ReceiverFlipsTheNotionalSign) {
  const b::Date value_date = b::Date::from_iso("2026-01-15");
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");
  const b::Date maturity = b::resolve("10Y", value_date, "NONE", "Following", 0);

  // Same deal, but we RECEIVE fixed: NPV = fixed - float = -(float - fixed), so the payer-of-fixed kernel
  // sees a negative notional.
  const tr::Trade t = tr::Trade::vanilla_swap(
      "USD-SOFR-10Y-002", 100e6, tr::Pay::Float, 0.0375, "USD", "USD-SOFR",
      b::spot_date(value_date, conv.calendar, conv.spot_lag), maturity, 0, 0);

  EXPECT_DOUBLE_EQ(t.signed_notional(), -100e6);
  EXPECT_DOUBLE_EQ(t.to_position(value_date, conv).notional, -100e6);
}
