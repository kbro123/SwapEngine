// QuantLib-FREE gate for the bond-future / cheapest-to-deliver kernel (pricing/bond_future.hpp). These
// properties are pinned against references OTHER than QuantLib (which has no first-class Treasury-future
// conversion-factor object), in the spirit of tests/bond_reference_test.cpp:
//
//   1. The CME 6% conversion factor is, BY CONSTRUCTION, the deliverable's clean price at the 6% notional
//      yield. So cme_conversion_factor() must equal build+bond_clean_from_yield(0.06) on a bond that is an
//      exact whole (and half) number of years from a coupon date — a completely independent code path.
//   2. Hull's textbook worked factor (10% coupon, 20y, CF = 1.4623) and the par self-check (a 6% bond an
//      exact number of half-years out has CF = 1).
//   3. Invoice price = futures·CF + accrued, gross basis = clean − futures·CF (definitional).
//   4. Net basis is zero exactly at repo == implied repo (the cash-and-carry break-even identity), and the
//      CTD is the max-implied-repo bond in a hand-built basket.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/build/bond.hpp"
#include "swaps/pricing/bond.hpp"
#include "swaps/pricing/bond_future.hpp"

namespace px = swaps::pricing;
namespace bld = swaps::build;

namespace {

// A bond built to settle ON a coupon-grid date (zero accrued), `half_years` semiannual periods to maturity.
bld::BuiltBond on_grid_bond(int half_years, double coupon) {
  const bld::Date issue = bld::Date::ymd(2020, 2, 15);  // a coupon-grid anchor
  bld::FixedBondTerms t;
  t.value_date = issue;
  t.settle = issue;  // on a coupon date => zero accrued, first coupon a full period away
  t.issue = issue;
  t.maturity = issue.plus_months(6 * half_years);
  t.coupon = coupon;
  t.freq = 2;
  return bld::fixed_rate_bond(t);
}

}  // namespace

// (1) The conversion factor IS the 6% clean price — cross-checked against the yield kernel for z=0 and z=6.
TEST(BondFuture, ConversionFactorEqualsSixPercentCleanPrice) {
  struct Case { int n; int z; int half_years; double coupon; };
  for (const Case& c : {Case{10, 0, 20, 0.05}, Case{10, 6, 21, 0.05}, Case{7, 0, 14, 0.0375},
                        Case{5, 6, 11, 0.08}}) {
    const double cf = px::cme_conversion_factor<double>(c.coupon, c.n, c.z);
    const bld::BuiltBond bb = on_grid_bond(c.half_years, c.coupon);
    const double clean_6pct = px::bond_clean_from_yield(bb.yield, 0.06);
    EXPECT_NEAR(cf, clean_6pct, 1e-10) << "n=" << c.n << " z=" << c.z;
  }
}

// (2) A 6% bond an exact number of half-years out has CF = 1; Hull's 10% 20y worked example is 1.4623.
TEST(BondFuture, ConversionFactorKnownValues) {
  EXPECT_NEAR(px::cme_conversion_factor<double>(0.06, 15, 0), 1.0, 1e-12);
  EXPECT_NEAR(px::cme_conversion_factor<double>(0.06, 8, 0), 1.0, 1e-12);
  EXPECT_NEAR(px::cme_conversion_factor<double>(0.10, 20, 0), 1.4623, 5e-5);  // Hull, OFOD
}

// (3) Definitional identities: invoice price and gross basis.
TEST(BondFuture, InvoiceAndGrossBasisIdentities) {
  const double futures = 0.985, cf = 0.9123, accrued_del = 0.0142, clean = 0.9015;
  EXPECT_NEAR(px::bond_future_invoice_price<double>(futures, cf, accrued_del), futures * cf + accrued_del,
              1e-15);
  EXPECT_NEAR(px::bond_future_gross_basis<double>(clean, futures, cf), clean - futures * cf, 1e-15);
}

// (4a) Net basis vanishes exactly at repo == implied repo (cash-and-carry break-even), even with an interim
// coupon in the delivery window.
TEST(BondFuture, NetBasisZeroAtImpliedRepo) {
  px::DeliverableInput in;
  in.clean = 0.9740;
  in.conversion_factor = 0.9105;
  in.accrued_now = 0.0110;
  in.accrued_delivery = 0.0021;
  in.interim_coupons = {{0.0225, 45.0}};  // a 4.5% semi coupon paid 45 days before delivery
  const double futures = 1.0685, days = 120.0;
  const double irr = px::bond_future_implied_repo<double>(in, futures, days);
  EXPECT_NEAR(px::bond_future_net_basis<double>(in, futures, irr, days), 0.0, 1e-12);
  // And net basis is monotone increasing in the funding repo (higher financing cost => less net carry
  // => costlier to hold the CTD to delivery). It is symmetric about zero at the implied repo.
  EXPECT_LT(px::bond_future_net_basis<double>(in, futures, irr - 0.01, days),
            px::bond_future_net_basis<double>(in, futures, irr + 0.01, days));
}

// (4b) CTD selection: the max-implied-repo bond, and it is also (here) the min net basis at a common repo.
TEST(BondFuture, CtdIsMaxImpliedRepo) {
  const double futures = 0.9850, repo = 0.050, days = 90.0;
  // Three deliverables; #1 is deliberately the richest to carry (highest implied repo).
  std::vector<px::DeliverableInput> basket(3);
  basket[0] = {0.9910, 0.9880, 0.0130, 0.0090, {}};
  basket[1] = {0.9600, 0.9950, 0.0155, 0.0110, {}};  // CTD: priced cheap vs its CF
  basket[2] = {0.9990, 1.0060, 0.0100, 0.0060, {}};
  std::vector<px::DeliverableResult<double>> results;
  for (const auto& in : basket) results.push_back(px::analyze_deliverable<double>(in, futures, repo, days));

  const std::size_t ctd = px::select_ctd(results);
  EXPECT_EQ(ctd, 1u);
  for (std::size_t i = 0; i < results.size(); ++i)
    if (i != ctd) {
      EXPECT_GE(results[ctd].implied_repo, results[i].implied_repo);
      EXPECT_LE(results[ctd].net_basis, results[i].net_basis);  // min net basis coincides here
    }
}
