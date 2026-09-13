// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// build::bond_from_terms and pricing::street_analytics (E7 stage 3.2): the library functions the `bonds` verb now
// calls in one line. The terms builder must reproduce the named convention builders it replaces for seasoned and
// when-issued bonds, a frequency override must equal hand-filled FixedBondTerms, and the analytics must be the
// kernel's price<->yield identities with exactly one quote. The verb's rateslib pins (api_test
// BondsVerbSelectsConventionAndHandlesWhenIssued) still pin the numbers end to end.
#include <optional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "swaps/build/bond.hpp"

namespace b = swaps::build;
namespace px = swaps::pricing;

namespace {
b::Date d(const char* iso) { return b::Date::from_iso(iso); }

void expect_same_bond(const b::BuiltBond& x, const b::BuiltBond& y) {
  EXPECT_EQ(x.accrued, y.accrued);
  ASSERT_EQ(x.yield.flows.size(), y.yield.flows.size());
  for (std::size_t i = 0; i < x.yield.flows.size(); ++i) {
    EXPECT_EQ(x.yield.flows[i].amount, y.yield.flows[i].amount) << i;
    EXPECT_EQ(x.yield.flows[i].exponent, y.yield.flows[i].exponent) << i;
  }
  EXPECT_EQ(x.yield.conv.freq, y.yield.conv.freq);
  EXPECT_EQ(x.yield.conv.stub, y.yield.conv.stub);
  EXPECT_EQ(x.yield.conv.final_period_simple, y.yield.conv.final_period_simple);
}
}  // namespace

TEST(BondTerms, ReproduceTheNamedConventionBuilders) {
  const b::Date vd = d("2024-01-16");
  b::BondTerms seasoned;
  seasoned.convention = "US-TREASURY";
  seasoned.settle = vd;
  seasoned.issue = d("2019-08-15");
  seasoned.maturity = d("2029-08-15");
  seasoned.coupon = 0.025;
  expect_same_bond(b::bond_from_terms(seasoned, vd),
                   b::bond_from_convention("US-TREASURY", vd, vd, d("2019-08-15"), d("2029-08-15"), 0.025));

  b::BondTerms wi;
  wi.convention = "US-TREASURY-TSY";
  wi.settle = d("2024-06-15");
  wi.dated = d("2024-06-15");
  wi.first_coupon = d("2024-11-15");
  wi.maturity = d("2034-11-15");
  wi.coupon = 0.045;
  expect_same_bond(b::bond_from_terms(wi, wi.settle),
                   b::wi_bond_from_convention("US-TREASURY-TSY", wi.settle, d("2024-06-15"), d("2024-11-15"),
                                              d("2034-11-15"), 0.045, d("2024-06-15")));
}

TEST(BondTerms, AFrequencyOverrideEqualsHandFilledTerms) {
  const b::Date vd = d("2024-01-16");
  b::BondTerms t;
  t.convention = "US-TREASURY";  // semiannual in the DB; overridden to annual below
  t.settle = vd;
  t.issue = d("2019-08-15");
  t.maturity = d("2029-08-15");
  t.coupon = 0.025;
  t.freq = 1;
  const px::YieldConvention yc = b::yield_convention("US-TREASURY");
  expect_same_bond(b::bond_from_terms(t, vd),
                   b::fixed_rate_bond(b::FixedBondTerms{vd, vd, d("2019-08-15"), d("2029-08-15"), 0.025, 1, yc.stub,
                                                        yc.final_period_simple}));
}

TEST(BondTerms, IncompleteTermsThrow) {
  b::BondTerms t;
  t.settle = d("2024-01-16");
  t.issue = d("2019-08-15");
  t.maturity = d("2029-08-15");
  EXPECT_THROW(b::bond_from_terms(t, t.settle), std::invalid_argument) << "no convention";
  t.convention = "US-TREASURY";
  t.issue.reset();
  EXPECT_THROW(b::bond_from_terms(t, t.settle), std::invalid_argument) << "neither issue nor dated";
  t.dated = d("2024-06-15");
  EXPECT_THROW(b::bond_from_terms(t, t.settle), std::invalid_argument) << "dated without first_coupon";
}

TEST(StreetAnalytics, IsThePriceYieldIdentityWithExactlyOneQuote) {
  const b::BuiltBond bb = b::bond_from_convention("US-TREASURY", d("2024-01-16"), d("2024-01-16"), d("2019-08-15"),
                                                  d("2029-08-15"), 0.025);
  const px::StreetAnalytics from_yield = px::street_analytics(bb.yield, std::nullopt, 0.04);
  EXPECT_EQ(from_yield.yield, 0.04);
  EXPECT_EQ(from_yield.dirty, px::bond_dirty_from_yield(bb.yield, 0.04));
  EXPECT_EQ(from_yield.accrued, bb.yield.accrued);
  EXPECT_EQ(from_yield.clean, from_yield.dirty - from_yield.accrued);
  const px::BondRisk r = px::bond_risk(bb.yield, 0.04);
  EXPECT_EQ(from_yield.modified_duration, r.modified_duration);
  EXPECT_EQ(from_yield.macaulay_duration, r.macaulay_duration);
  EXPECT_EQ(from_yield.convexity, r.convexity);

  // Quoting the clean price that yield implies recovers the yield (Newton to 1e-14 on price).
  const px::StreetAnalytics from_clean = px::street_analytics(bb.yield, from_yield.clean, std::nullopt);
  EXPECT_NEAR(from_clean.yield, 0.04, 1e-12);
  EXPECT_NEAR(from_clean.clean, from_yield.clean, 1e-14);

  EXPECT_THROW(px::street_analytics(bb.yield, 0.99, 0.04), std::invalid_argument);
  EXPECT_THROW(px::street_analytics(bb.yield, std::nullopt, std::nullopt), std::invalid_argument);
}
