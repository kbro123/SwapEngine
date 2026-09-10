// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// FxMatrix: the shared, triangulating FX-spot store (market/fx.hpp). Convention under test:
//   (base, quote, rate) means 1 base = rate * quote, so rate(from,to) = units of `to` per 1 unit of `from`.
#include <gtest/gtest.h>

#include <stdexcept>

#include "swaps/market/fx.hpp"

namespace mkt = swaps::market;

TEST(FxMatrix, DirectInverseTriangulateAndConvert) {
  mkt::FxMatrix fx("USD");      // the pivot is explicit (no default currency)
  fx.add("EUR", "USD", 1.09);   // EURUSD = 1.09  (1 EUR = 1.09 USD)
  fx.add("USD", "JPY", 150.0);  // USDJPY = 150   (1 USD = 150 JPY)

  // Direct stored pair.
  EXPECT_DOUBLE_EQ(fx.rate("EUR", "USD"), 1.09);

  // Inverse of a stored pair.
  EXPECT_NEAR(fx.rate("USD", "EUR"), 1.0 / 1.09, 1e-9);

  // Triangulation through the USD pivot: EUR -> USD -> JPY.
  EXPECT_NEAR(fx.rate("EUR", "JPY"), 1.09 * 150.0, 1e-9);

  // Reverse triangulation: JPY -> USD -> EUR.
  EXPECT_NEAR(fx.rate("JPY", "EUR"), 1.0 / (1.09 * 150.0), 1e-9);

  // Identity.
  EXPECT_DOUBLE_EQ(fx.rate("USD", "USD"), 1.0);

  // convert() applies the same resolution.
  EXPECT_NEAR(fx.convert(100.0, "EUR", "JPY"), 100.0 * 1.09 * 150.0, 1e-9);

  // has() probes resolvable paths.
  EXPECT_TRUE(fx.has("EUR", "JPY"));
  EXPECT_TRUE(fx.has("JPY", "EUR"));
  EXPECT_TRUE(fx.has("USD", "USD"));
}

TEST(FxMatrix, MissingPathThrowsAndHasIsFalse) {
  mkt::FxMatrix fx("USD");
  fx.add("EUR", "USD", 1.09);
  fx.add("USD", "JPY", 150.0);

  // An unrelated currency with no quote and no pivot leg has no path.
  EXPECT_FALSE(fx.has("EUR", "GBP"));
  EXPECT_FALSE(fx.has("GBP", "USD"));
  EXPECT_THROW(fx.rate("EUR", "GBP"), std::runtime_error);
  EXPECT_THROW(fx.convert(100.0, "GBP", "USD"), std::runtime_error);
}

TEST(FxMatrix, AddFxRateStructAndInverseImplied) {
  mkt::FxMatrix fx("USD");
  fx.add(mkt::FxRate{"EUR", "USD", 1.09});
  // Only EURUSD was stored, yet USDEUR resolves as its inverse.
  EXPECT_NEAR(fx.rate("USD", "EUR"), 1.0 / 1.09, 1e-9);
  EXPECT_TRUE(fx.empty() == false);
}
