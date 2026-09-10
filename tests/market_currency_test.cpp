// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Market-domain object model: the first-class Currency (market/currency.hpp) is a typed reference object over
// the conventions DB. Its currency set + each currency's discount index and settlement calendar are DERIVED
// from the DB's overnight indices, so these assertions also pin that the derivation stays wired to the DB.
#include <gtest/gtest.h>

#include <stdexcept>

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
  // The zero-decimal currencies now carry a DB index (JPY-TONA, KRW-KOFR, IDR-INDONIA), so they are known
  // AND resolve 0 minor units from the explicit table -- the two facts come from different sources and must
  // agree. (A currency without a DB index would still get its table value; here all three are known.)
  for (const char* c : {"JPY", "KRW", "IDR"}) {
    const mkt::Currency ccy = mkt::Currency::of(c);
    EXPECT_EQ(ccy.minor_units(), 0) << c;
    EXPECT_TRUE(ccy.known()) << c;
  }
}

TEST(MarketCurrency, UnknownCodeStaysValidButEmpty) {
  const mkt::Currency zzz = mkt::Currency::of("ZZZ");
  EXPECT_TRUE(zzz.valid());          // a usable value object...
  EXPECT_FALSE(zzz.known());         // ...but not recognised in the DB-derived registry
  // No fallbacks (PRINCIPLES.md P2): every convention accessor throws for a code the DB does not carry.
  EXPECT_THROW(zzz.settlement_calendar(), std::invalid_argument);
  EXPECT_THROW(zzz.discount_index(), std::invalid_argument);
  EXPECT_THROW(zzz.minor_units(), std::invalid_argument);

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
  EXPECT_TRUE(has("JPY"));   // now carries JPY-TONA in the DB -> a known currency
  EXPECT_FALSE(has("SGD"));  // not in currencies[] -> not a known currency (add a DB row, or the conventions verb)
}
