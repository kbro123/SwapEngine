// Market-domain object model: the first-class Currency (market/currency.hpp) is a typed reference object over
// the conventions DB. Its currency set + each currency's discount index and settlement calendar are DERIVED
// from the DB's overnight indices, so these assertions also pin that the derivation stays wired to the DB.
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "swaps/market/currency.hpp"

namespace mkt = swaps::market;

TEST(MarketCurrency, UsdIsDerivedFromTheDb) {
  const mkt::Currency usd = mkt::Currency::of("USD");
  EXPECT_TRUE(usd.valid());
  EXPECT_TRUE(usd.known());
  EXPECT_EQ(usd.code, "USD");
  EXPECT_EQ(usd.minor_units(), 2);

  // Settlement calendar: the currency-level holiday calendar, id non-empty (distinct from an index calendar).
  EXPECT_FALSE(usd.settlement_calendar().id.empty());
  EXPECT_EQ(usd.settlement_calendar().id, "USD");

  // Discount index: the pinned OIS benchmark (SOFR, not FedFunds), an overnight index whose currency is USD.
  EXPECT_EQ(usd.discount_index().id, "USD-SOFR");
  EXPECT_TRUE(usd.discount_index().is_overnight());
  EXPECT_EQ(usd.discount_index().currency(), "USD");
}

TEST(MarketCurrency, EurIsDerivedFromTheDb) {
  const mkt::Currency eur = mkt::Currency::of("EUR");
  EXPECT_TRUE(eur.known());
  EXPECT_EQ(eur.minor_units(), 2);
  EXPECT_EQ(eur.settlement_calendar().id, "EUR");
  EXPECT_EQ(eur.discount_index().id, "EUR-ESTR");
  EXPECT_TRUE(eur.discount_index().is_overnight());
  EXPECT_EQ(eur.discount_index().currency(), "EUR");
}

TEST(MarketCurrency, MinorUnitsTableCoversZeroDecimalCurrencies) {
  // JPY is in the explicit minor-units table (0 decimals) but has NO index in the conventions DB, so it is
  // NOT a known currency: minor_units resolves from the table, while known()==false.
  const mkt::Currency jpy = mkt::Currency::of("JPY");
  EXPECT_EQ(jpy.minor_units(), 0);
  EXPECT_FALSE(jpy.known());
}

TEST(MarketCurrency, UnknownCodeStaysValidButEmpty) {
  const mkt::Currency zzz = mkt::Currency::of("ZZZ");
  EXPECT_TRUE(zzz.valid());          // a usable value object...
  EXPECT_FALSE(zzz.known());         // ...but not recognised in the DB-derived registry
  EXPECT_TRUE(zzz.settlement_calendar().empty());
  EXPECT_FALSE(zzz.discount_index().known());
  EXPECT_EQ(zzz.minor_units(), 2);   // ISO default for anything unlisted

  const mkt::Currency none;
  EXPECT_FALSE(none.valid());        // default-constructed / empty code
}

TEST(MarketCurrency, KnownCodesListComesFromTheDb) {
  const std::vector<std::string> codes = mkt::Currency::known_codes();
  const auto has = [&codes](const std::string& c) {
    return std::find(codes.begin(), codes.end(), c) != codes.end();
  };
  EXPECT_TRUE(has("USD"));
  EXPECT_TRUE(has("EUR"));
  EXPECT_FALSE(has("JPY"));  // no DB index -> not a known currency
}
