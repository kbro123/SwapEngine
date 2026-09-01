// The first-class market Quote value object (market/quote.hpp) resolves a real two-sided market to the
// engine's calibration target + soft-quote band. These asserts pin the mid/spread arithmetic, the
// two-sided -> soft-band and mid-only -> hard-pin bridges, and source/timestamp round-tripping.
#include <gtest/gtest.h>

#include "swaps/market/quote.hpp"

namespace mkt = swaps::market;

TEST(MarketQuote, TwoSidedMidSpreadAndFlags) {
  const mkt::Quote q = mkt::Quote::bid_ask(3.70, 3.74);
  EXPECT_TRUE(q.is_two_sided());
  EXPECT_DOUBLE_EQ(q.mid(), 3.72);
  EXPECT_NEAR(q.spread(), 0.04, 1e-12);
  EXPECT_DOUBLE_EQ(q.bid(), 3.70);
  EXPECT_DOUBLE_EQ(q.ask(), 3.74);
}

TEST(MarketQuote, TwoSidedResolvesToSoftBand) {
  const double decay = 0.25;
  const mkt::CalibrationTarget t = mkt::Quote::bid_ask(3.70, 3.74).to_target(decay);
  EXPECT_DOUBLE_EQ(t.target, 3.72);      // -> Instrument.market
  EXPECT_DOUBLE_EQ(t.band_lower, 3.70);  // -> Instrument.band_lower
  EXPECT_DOUBLE_EQ(t.band_upper, 3.74);  // -> Instrument.band_upper
  EXPECT_DOUBLE_EQ(t.band_decay, decay); // -> Instrument.band_decay
  EXPECT_GT(t.band_upper, t.band_lower); // a SOFT band (engine treats band_upper > band_lower as soft)
  EXPECT_TRUE(t.is_soft_band());
}

TEST(MarketQuote, MidOnlyResolvesToHardPin) {
  const mkt::Quote q = mkt::Quote::mid(3.72);
  EXPECT_FALSE(q.is_two_sided());
  EXPECT_DOUBLE_EQ(q.mid(), 3.72);
  EXPECT_DOUBLE_EQ(q.spread(), 0.0);
  EXPECT_DOUBLE_EQ(q.bid(), 3.72);  // collapses to mid — no genuine two-sided market
  EXPECT_DOUBLE_EQ(q.ask(), 3.72);

  const mkt::CalibrationTarget t = q.to_target();
  EXPECT_DOUBLE_EQ(t.target, 3.72);        // -> Instrument.market
  EXPECT_LE(t.band_upper, t.band_lower);   // a HARD pin (engine default: band_upper <= band_lower)
  EXPECT_DOUBLE_EQ(t.band_decay, 1.0);     // -> Instrument.band_decay (unit weight => plain residual)
  EXPECT_TRUE(t.is_hard_pin());
}

TEST(MarketQuote, SourceAndTimestampRoundTrip) {
  const mkt::Quote q = mkt::Quote::bid_ask(3.70, 3.74, "BBG", 46000.5);
  EXPECT_EQ(q.source(), "BBG");
  EXPECT_DOUBLE_EQ(q.timestamp(), 46000.5);

  const mkt::Quote m = mkt::Quote::mid(3.72, "ICAP", 46001.0);
  EXPECT_EQ(m.source(), "ICAP");
  EXPECT_DOUBLE_EQ(m.timestamp(), 46001.0);
}
