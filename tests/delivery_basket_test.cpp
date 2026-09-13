// E5 taxonomy: T1 oracle (engine number vs an independent number) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// build::analyze_delivery_basket (E7 stage 3.3), the library behind the `bond_future` verb, tested directly so
// tools/mutate.py can reach it. T1: CME's PUBLISHED conversion factors, "Calculating U.S. Treasury Futures
// Conversion Factors" (CME Group IR232, worked examples 1-5: TU / 3-Year / FV month-rounded, TY / US
// quarter-rounded). T5: the IR232 remaining terms by hand, the exchange's 4-dp rounding, and invoice / gross basis
// on the ROUNDED factor. The end-to-end pin through run_json is tests/bond_future_reference_test.cpp.
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "swaps/build/bond_future.hpp"
#include "tolerances.hpp"

namespace b = swaps::build;
namespace px = swaps::pricing;

namespace {
b::Date d(const char* iso) { return b::Date::from_iso(iso); }

b::DeliveryBasketRequest one_bond(const char* contract, const char* value_date, const char* first_delivery,
                                  const char* issue, const char* maturity, double coupon, double clean) {
  b::DeliveryBasketRequest r;
  r.contract = contract;
  r.value_date = d(value_date);
  r.first_delivery = d(first_delivery);
  r.futures_price = 1.24;
  r.repo = 0.005;
  b::Deliverable x;
  x.id = "B";
  x.issue = d(issue);
  x.maturity = d(maturity);
  x.coupon = coupon;
  x.clean = clean;
  r.basket = {x};
  return r;
}
}  // namespace

TEST(DeliveryBasket, ConversionFactorsAreCmesPublishedValues) {
  struct Row {
    const char* contract, *value_date, *first_delivery, *issue, *maturity;
    double coupon, cme;
  };
  const Row rows[] = {
      {"CME-TU", "2008-11-20", "2008-12-01", "2008-10-31", "2010-10-31", 0.015, 0.9229},    // 912828JP6
      {"CME-TU", "2009-02-20", "2009-03-01", "2009-01-15", "2012-01-15", 0.01125, 0.8747},  // 912828KB5, CME 3-Year
                                                                                            // (same CF rows as TU)
      {"CME-FV", "2008-11-20", "2008-12-01", "2008-10-31", "2013-10-31", 0.0275, 0.8653},   // 912828JQ4
      {"CME-TY", "2008-11-20", "2008-12-01", "2008-11-15", "2018-11-15", 0.0375, 0.8357},   // 912828JR2
      {"CME-US", "2008-11-10", "2008-12-01", "2008-05-15", "2038-05-15", 0.045, 0.7943},    // 912810PX0
  };
  for (const Row& row : rows) {
    const b::DeliveryBasketResult res = b::analyze_delivery_basket(
        one_bond(row.contract, row.value_date, row.first_delivery, row.issue, row.maturity, row.coupon, 1.0));
    ASSERT_EQ(res.rows.size(), 1u);
    // Exact: the published k / 1e4 and the rounded factor are the same correctly rounded double.
    EXPECT_EQ(res.rows[0].result.conversion_factor, row.cme) << row.contract << " " << row.maturity;
  }
}

TEST(DeliveryBasket, TheTermCountsWholeMonthsFromTheFirstOfTheDeliveryMonth) {
  // IR232: 10-Year 3-3/4s of Nov-15-2018 into Dec-2008 is 9y 11m 14d -> quarters -> 9y 9m.
  const b::ConversionTerm ty = b::conversion_term(d("2008-12-01"), d("2018-11-15"), 3);
  EXPECT_EQ(ty.years, 9);
  EXPECT_EQ(ty.months, 9);
  // IR232: Bond 4-1/2s of May-15-2038 into Dec-2008 is 29y 5m 14d -> 29y 3m.
  const b::ConversionTerm us = b::conversion_term(d("2008-12-01"), d("2038-05-15"), 3);
  EXPECT_EQ(us.years, 29);
  EXPECT_EQ(us.months, 3);
  // IR232: 5-Year 2-3/4s of Oct-31-2013 into Dec-2008 is 4y 10m 30d -> whole months 4y 10m.
  const b::ConversionTerm fv = b::conversion_term(d("2008-12-01"), d("2013-10-31"), 1);
  EXPECT_EQ(fv.years, 4);
  EXPECT_EQ(fv.months, 10);
  // A first_delivery given as the first BUSINESS day (Mon 2009-03-02) still counts from the 1st: a maturity on
  // Jun-01-2014 is 5y 3m out, not 5y 2m.
  const b::ConversionTerm first = b::conversion_term(d("2009-03-02"), d("2014-06-01"), 1);
  EXPECT_EQ(first.years, 5);
  EXPECT_EQ(first.months, 3);
  EXPECT_THROW(b::conversion_term(d("2008-12-01"), d("2018-11-15"), 5), std::invalid_argument);
  EXPECT_THROW(b::conversion_term(d("2008-12-01"), d("2018-11-15"), 0), std::invalid_argument);
}

TEST(DeliveryBasket, TheFactorIsRoundedToTheContractsDecimals) {
  EXPECT_EQ(b::round_conversion_factor(0.83565054249792992, 4), 0.8357);
  EXPECT_EQ(b::round_conversion_factor(0.83565054249792992, 6), 0.835651);
  EXPECT_EQ(swaps::conventions::require_bond_future("CME-TY").conversion_factor_decimals, 4);
}

TEST(DeliveryBasket, InvoiceAndBasisUseTheExchangeFactor) {
  // TY 3-3/4s of Nov-15-2018 bought 2008-11-20 for delivery 2008-12-01. Accrued at delivery by hand, ACT/ACT ICMA:
  // 16 days of the 181-day Nov-15-2008 -> May-15-2009 period.
  const b::DeliveryBasketResult res =
      b::analyze_delivery_basket(one_bond("CME-TY", "2008-11-20", "2008-12-01", "2008-11-15", "2018-11-15", 0.0375, 1.03));
  const px::DeliverableResult<double>& x = res.rows[0].result;
  const double accrued_delivery = 0.0375 / 2.0 * 16.0 / 181.0;
  EXPECT_NEAR(x.invoice_price, 1.24 * 0.8357 + accrued_delivery, swaps::tol::literal);
  EXPECT_NEAR(x.gross_basis, 1.03 - 1.24 * 0.8357, swaps::tol::literal);
  ASSERT_TRUE(res.ctd.has_value());
  EXPECT_EQ(*res.ctd, 0u);
}

TEST(DeliveryBasket, TheContractSuppliesDefaultsAndRequestFieldsOverride) {
  b::DeliveryBasketRequest r =
      one_bond("CME-TY", "2008-11-20", "2008-12-01", "2008-11-15", "2018-11-15", 0.0375, 1.03);
  const double by_default = b::analyze_delivery_basket(r).rows[0].result.conversion_factor;
  r.basket[0].convention = "US-TREASURY";  // the contract's deliverable convention, named explicitly
  EXPECT_EQ(b::analyze_delivery_basket(r).rows[0].result.conversion_factor, by_default);

  r.notional_coupon = 0.04;
  EXPECT_EQ(b::analyze_delivery_basket(r).rows[0].result.conversion_factor,
            b::round_conversion_factor(px::cme_conversion_factor<double>(0.0375, 9, 9, 0.04), 4));
  r.notional_coupon.reset();
  r.round_months = 1;  // whole months: 9y 11m
  EXPECT_EQ(b::analyze_delivery_basket(r).rows[0].result.conversion_factor,
            b::round_conversion_factor(px::cme_conversion_factor<double>(0.0375, 9, 11, 0.06), 4));

  r.contract.clear();
  EXPECT_THROW(b::analyze_delivery_basket(r), std::invalid_argument);
  r.contract = "NO-SUCH";
  EXPECT_THROW(b::analyze_delivery_basket(r), std::invalid_argument);
  r.contract = "CME-TY";
  r.basket.clear();
  EXPECT_FALSE(b::analyze_delivery_basket(r).ctd.has_value());
}
