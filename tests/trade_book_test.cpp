// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// trade::NettingSet (exposure unit under a CSA) — the grouping object that composes Trades for the `exposure`
// verb: the netting set carries its discounting via the CSA (an object, not the hard-coded "whole book = one
// netting set"). (The trade::Book tree and its tests were deleted in E6.1, 2026-09-10: no consumer.)
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
                                 b::resolve("0d", vd, "NONE", "Following", 0), b::resolve(maturity, vd, "NONE", "Following", 0),
                                 /*forecast_role=*/0, /*discount_role=*/0);
}
}  // namespace

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

// The per-trade-convention and CSA-override materializations added with the API wiring: each trade rolls
// under ITS OWN index's conventions (to_position(vd)), and NettingSet::to_book(vd,
// csa_role) forces every trade's discounting onto the CSA's collateral-OIS role.
TEST(TradeBook, PerTradeConventionsAndCsaDiscountRole) {
  namespace tr = swaps::trade;
  namespace bld = swaps::build;
  const bld::Date vd = bld::Date::from_iso("2026-09-04");
  const bld::Date eff = bld::Date::from_iso("2026-09-08");
  const bld::Date mat = bld::Date::from_iso("2031-09-08");

  tr::Trade t = tr::Trade::vanilla_swap("T1", 1e6, tr::Pay::Fixed, 0.03, "USD", "USD-SOFR", eff, mat,
                                        /*forecast*/ 0, /*discount*/ 0);
  // Resolving from the trade's OWN index equals resolving its convention by hand — one source of truth.
  const auto own = t.to_position(vd);
  const auto handed = t.to_position(vd, bld::Index("USD-SOFR").par_convention().resolve());
  ASSERT_EQ(own.float_coupons.size(), handed.float_coupons.size());
  EXPECT_EQ(own.float_coupons.back().pay, handed.float_coupons.back().pay);
  EXPECT_EQ(own.fixed_coupons.size(), handed.fixed_coupons.size());

  // NettingSet::to_book(vd, csa_role): the CSA's role OVERRIDES the trade's booked discount roles.
  tr::NettingSet ns("CP-1", tr::CSA::cash("USD"));
  t.discount_curve = 7;  // booked against some other role on purpose
  ns.add(t);
  const auto book = ns.to_book(vd, /*csa_discount_role=*/2);
  ASSERT_EQ(book.positions.size(), 1u);
  EXPECT_EQ(book.positions[0].disc_curve, 2) << "the CSA decides discounting, not the booked int";
  EXPECT_EQ(book.positions[0].fixed_curve, 2);
  EXPECT_EQ(book.positions[0].fwd_curve, 0) << "forecasting stays the trade's own index role";
}
