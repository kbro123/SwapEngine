// trade::Book (portfolio tree) and trade::NettingSet (exposure unit under a CSA) — the objects that compose
// Trades. Proves the tree flattens + materializes into the fast MultiCurveBook, and the netting set carries
// its discounting via the CSA (an object, not the hard-coded "whole book = one netting set").
#include <gtest/gtest.h>

#include "swaps/build/conventions.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/trade/book.hpp"

namespace tr = swaps::trade;
namespace b = swaps::build;

namespace {
tr::Trade usd_swap(const std::string& id, double notional, tr::Pay pay, double rate, const char* maturity) {
  const b::Date vd = b::Date::from_iso("2026-09-01");
  return tr::Trade::vanilla_swap(id, notional, pay, rate, "USD", "USD-SOFR",
                                 b::resolve("0d", vd), b::resolve(maturity, vd),
                                 /*forecast_role=*/0, /*discount_role=*/0);
}
}  // namespace

TEST(TradeBook, TreeFlattensAndMaterializesToTheFastBook) {
  const b::Date vd = b::Date::from_iso("2026-09-01");
  const b::SwapConv conv = b::swap_conv("USD", 1.0, "USD-SOFR");

  tr::Book desk("USD-Rates");
  desk.add(usd_swap("t1", 100e6, tr::Pay::Fixed, 0.0375, "10y"))   // payer of fixed
      .add(usd_swap("t2", 50e6, tr::Pay::Float, 0.0360, "5y"));    // receiver of fixed
  tr::Book strat("relative-value");
  strat.add(usd_swap("t3", 25e6, tr::Pay::Fixed, 0.0345, "2y"));
  desk.add_subbook(strat);

  EXPECT_EQ(desk.count(), 3);                 // recursive count across the tree
  EXPECT_EQ(desk.trades().size(), 2u);        // booked directly on the desk
  EXPECT_EQ(desk.all_trades().size(), 3u);    // flattened (incl. the sub-book)

  const auto mb = desk.to_book(vd, conv);     // materialize -> the fast valuation book
  ASSERT_EQ(mb.positions.size(), 3u);
  EXPECT_DOUBLE_EQ(mb.positions[0].notional, 100e6);   // payer -> +notional (direction in the sign)
  EXPECT_DOUBLE_EQ(mb.positions[1].notional, -50e6);   // receiver -> -notional
  EXPECT_DOUBLE_EQ(mb.positions[0].fixed_rate, 0.0375);
  EXPECT_FALSE(mb.positions[0].float_coupons.empty());
  EXPECT_FALSE(mb.positions[0].fixed_coupons.empty());
}

TEST(TradeNettingSet, GroupsTradesAndDiscountsViaTheCsa) {
  tr::NettingSet ns("CPTY-A", tr::CSA::cash("USD"));
  ns.add(usd_swap("t1", 100e6, tr::Pay::Fixed, 0.0375, "10y"))
      .add(usd_swap("t2", 50e6, tr::Pay::Float, 0.0360, "5y"));

  EXPECT_EQ(ns.count(), 2);
  EXPECT_FALSE(ns.empty());
  EXPECT_EQ(ns.id(), "CPTY-A");
  EXPECT_TRUE(ns.csa().is_cash_collateralized());
  EXPECT_EQ(ns.discount_index_id(), "USD-SOFR");   // discounting is the CSA's collateral-ccy OIS
}
