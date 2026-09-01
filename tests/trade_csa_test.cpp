// Trade-domain object model: the first-class CSA (trade/csa.hpp) turns the engine's bare int discount `role`
// into a typed collateral-agreement object. Its discount-curve choice is DERIVED by delegating to
// market::Currency::discount_index(), so these assertions also pin that the derivation stays wired to the
// DB-derived currency registry (SOFR for USD, ESTR for EUR).
#include <gtest/gtest.h>

#include <string>

#include "swaps/trade/csa.hpp"

namespace tr = swaps::trade;

TEST(TradeCsa, UsdCashCsaDiscountsOnSofr) {
  const tr::CSA usd = tr::CSA::cash("USD");
  EXPECT_TRUE(usd.is_cash_collateralized());
  EXPECT_EQ(usd.collateral_currency, "USD");

  // Discount curve is the collateral currency's OIS, resolved through market::Currency.
  EXPECT_EQ(usd.discount_index_id(), "USD-SOFR");
  EXPECT_EQ(usd.discount_index().id, "USD-SOFR");
  EXPECT_TRUE(usd.discount_index().is_overnight());
  EXPECT_EQ(usd.discount_index().currency(), "USD");
  EXPECT_EQ(usd.currency().code, "USD");
}

TEST(TradeCsa, EurCashCsaDiscountsOnEstr) {
  const tr::CSA eur = tr::CSA::cash("EUR");
  EXPECT_TRUE(eur.is_cash_collateralized());
  EXPECT_EQ(eur.discount_index_id(), "EUR-ESTR");
  EXPECT_TRUE(eur.discount_index().is_overnight());
}

TEST(TradeCsa, EconomicTermsRoundTrip) {
  tr::CSA csa = tr::CSA::cash("USD");
  csa.threshold = 1'000'000.0;
  csa.mta = 250'000.0;
  csa.independent_amount = 5'000'000.0;
  csa.rounding = 10'000.0;

  EXPECT_DOUBLE_EQ(csa.threshold, 1'000'000.0);
  EXPECT_DOUBLE_EQ(csa.mta, 250'000.0);
  EXPECT_DOUBLE_EQ(csa.independent_amount, 5'000'000.0);
  ASSERT_TRUE(csa.rounding.has_value());
  EXPECT_DOUBLE_EQ(*csa.rounding, 10'000.0);

  // Terms do not disturb the derived discount choice.
  EXPECT_EQ(csa.discount_index_id(), "USD-SOFR");
}

TEST(TradeCsa, UncollateralizedReportsNotCashCollateralized) {
  const tr::CSA unc = tr::CSA::uncollateralized("USD");
  EXPECT_FALSE(unc.is_cash_collateralized());
  EXPECT_EQ(unc.type, tr::CSA::Type::Uncollateralized);
  EXPECT_EQ(unc.collateral_currency, "USD");
}
