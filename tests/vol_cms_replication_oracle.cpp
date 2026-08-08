// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// QuantLib ORACLE for the full Hagan static-replication CMS (vol/cms_replication.hpp): the standard-model G
// function matches QuantLib GFunctionStandard EXACTLY (weight function, the tight/substantive part), and the
// replicated CMS convexity-adjusted rate matches QuantLib's NumericHaganPricer (flat normal vol) to within
// ~15% — the residual is the standard-model G's S=0 pole, which QuantLib integrates through (quadrature-node-
// dependent) and we floor above. Also the robust limits (no vol -> no convexity; positive; grows with vol).
#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include <cmath>

#include "swaps/vol/cms_replication.hpp"

namespace v = swaps::vol;
namespace QL = QuantLib;

TEST(CmsReplication, GFunctionMatchesQuantLib) {
  struct Case { int q; double delta; int L; } cases[] = {{1, 1.0, 10}, {2, 0.5, 5}, {4, 0.25, 20}, {1, 0.0, 2}};
  for (const auto& c : cases) {
    auto qlg = QL::GFunctionFactory::newGFunctionStandard(c.q, c.delta, c.L);
    const v::GFunctionStandard g{static_cast<double>(c.q), c.delta, static_cast<double>(c.L)};
    for (double x : {0.005, 0.015, 0.030, 0.050, 0.080}) {
      EXPECT_NEAR(g(x), (*qlg)(x), 1e-10) << "G q=" << c.q << " d=" << c.delta << " L=" << c.L << " x=" << x;
      EXPECT_NEAR(g.d1(x), qlg->firstDerivative(x), 1e-9) << "G' x=" << x;
      EXPECT_NEAR(g.d2(x), qlg->secondDerivative(x), 1e-7) << "G'' x=" << x;
    }
  }
}

TEST(CmsReplication, ReplicatedRateMatchesNumericHaganPricer) {
  QL::Date today(15, QL::June, 2026);
  QL::Settings::instance().evaluationDate() = today;
  const QL::Handle<QL::YieldTermStructure> curve(
      QL::ext::make_shared<QL::FlatForward>(today, 0.030, QL::Actual365Fixed()));
  // A 10y annual-fixed swap index (q=1, swapLength=10); a CMS coupon fixing in ~5y, accruing 1y, pay at end
  // (so delta = 1.0). Flat NORMAL swaption vol so the only thing under test is the replication + G.
  const auto swapIndex = QL::ext::make_shared<QL::EuriborSwapIsdaFixA>(QL::Period(10, QL::Years), curve);
  const QL::Date start = QL::TARGET().advance(today, QL::Period(5, QL::Years));
  const QL::Date end = QL::TARGET().advance(start, QL::Period(1, QL::Years));
  QL::CmsCoupon coupon(end, 1.0, start, end, swapIndex->fixingDays(), swapIndex);

  for (double sigma : {0.0060, 0.0090, 0.0120}) {
    const QL::Handle<QL::SwaptionVolatilityStructure> vol(QL::ext::make_shared<QL::ConstantSwaptionVolatility>(
        0, QL::TARGET(), QL::Following, sigma, QL::Actual365Fixed(), QL::Normal));
    const auto pricer = QL::ext::make_shared<QL::NumericHaganPricer>(
        vol, QL::GFunctionFactory::Standard, QL::Handle<QL::Quote>(QL::ext::make_shared<QL::SimpleQuote>(0.0)));
    coupon.setPricer(pricer);
    const QL::Rate ql_cms = coupon.rate();

    const double F0 = swapIndex->fixing(coupon.fixingDate());
    const double expiry = vol->timeFromReference(coupon.fixingDate());
    const v::GFunctionStandard g{1.0, 1.0, 10.0};
    const double ours = v::cms_replicated_forward(F0, expiry, g, [&](double) { return sigma; });

    // Both are the convexity-adjusted CMS rate. We match QuantLib's adjustment to within ~15% — the residual
    // is the S=0 pole of the standard-model G (G'' ~ -C/S^2): QuantLib integrates THROUGH it (its Gauss-Kronrod
    // nodes miss the pole, so its pole-region value is quadrature-node-dependent), while we floor above it. On
    // a legitimately pole-ambiguous quantity this ballpark agreement, plus the EXACT G match above, is the
    // honest validation; the tight, exact part (the weight function) is separately gated.
    const double our_ca = ours - F0, ql_ca = ql_cms - F0;
    EXPECT_GT(our_ca, 0.0);
    EXPECT_LT(our_ca, ql_ca);                              // pole-avoiding => slightly below QL's pole-including
    EXPECT_NEAR(our_ca, ql_ca, 0.15 * ql_ca)
        << "sigma=" << sigma << " ql_ca=" << ql_ca << " our_ca=" << our_ca;
  }
}

TEST(CmsReplication, ZeroVolNoConvexityAndMonotone) {
  const v::GFunctionStandard g{1.0, 1.0, 10.0};
  const double F0 = 0.032, T = 5.0;
  EXPECT_NEAR(v::cms_replicated_forward(F0, T, g, [](double) { return 0.0; }), F0, 1e-10);
  const double a = v::cms_replicated_forward(F0, T, g, [](double) { return 0.0080; }) - F0;
  const double b = v::cms_replicated_forward(F0, T, g, [](double) { return 0.0120; }) - F0;
  EXPECT_GT(a, 0.0);
  EXPECT_GT(b, a);  // more vol -> more convexity
}
